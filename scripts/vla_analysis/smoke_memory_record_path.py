"""Smoke-test the ``lerobot-memory-record`` inference path for a memory-augmented pi05.

Why this exists (2 Sep 26): the on-robot script (``lerobot_memory_record.py``) was written
on 11 May and last deployed on 24 May; the model code beneath it gained ~3.3k lines since
(frozen-base dual pass, frozen pre-pass, VLM text-span memory, pooled/anchored routing,
expert anchors, shared-pair tables). This script drives the record script's OWN loader
(``_load_policy``, imported — CPU load -> lm_head trim -> ``.to(device)``) and then
``select_action`` under ``torch.inference_mode()`` with a batch of 1, exactly as
``_select_action`` does, so a change that breaks live inference fails HERE, not on the arm.

The pi05 processor pipeline (normalizer + gated PaliGemma tokenizer) is NOT exercised:
tokens are hand-built with pi05's prompt structure ``Task: <instr>, State: <bins>;\\nAction:``
so the ``"▁State"`` marker (id 3040) sits where the pooled-router / anchor code expects it.

Usage
-----
  # (default) build the E65 merged-6x2 real-world cell on top of local pi05_base weights
  # (memory random-init, tables shrunk via N_KEYS so it fits any card):
  python scripts/vla_analysis/smoke_memory_record_path.py
  # full-size tables (n=256, ~19 GB on a 24 GB card):
  N_KEYS=256 python scripts/vla_analysis/smoke_memory_record_path.py
  # a real memory checkpoint (config read from its config.json, no overrides):
  CKPT=outputs/train/<run>/checkpoints/last/pretrained_model python scripts/vla_analysis/smoke_memory_record_path.py
  # + save/reload round trip through a memory checkpoint, and the two negative tests
  # (offload override on shared tables -> config-time ValueError; a shape-changing
  # override -> the record script's post-load verification raises):
  MODE=roundtrip python scripts/vla_analysis/smoke_memory_record_path.py

Env: BASE (default outputs/pi05_base), CKPT, N_KEYS (default 64), MODE (main|roundtrip),
DEVICE (default cuda).
"""

import dataclasses
import io
import os
import shutil
import tempfile
import time
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_memory_record import _load_policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODE = os.environ.get("MODE", "main")
BASE = os.environ.get("BASE", "outputs/pi05_base")
CKPT = os.environ.get("CKPT")
N_KEYS = int(os.environ.get("N_KEYS", "64"))
DEVICE = os.environ.get("DEVICE", "cuda")

# E65 merged-6x2 real-world cell (rw_merged6x2_full_chain.sh + rw_rwarmup_common.sh, with the
# A-phase/sequential-stage overrides) — the paper cell deployed on the WidowX AI.
RW_MEM = {
    "enabled": True,
    "memory_only": False,
    "layers": [4, 6, 8, 10, 14, 16],
    "mem_n_keys": N_KEYS,
    "lora_rank": 2,
    "mem_knn": 36,
    "routing_loss_topk": 36,
    "vlm_layers": [5, 7, 9, 11, 13, 15],
    "vlm_mem_n_keys": N_KEYS,
    "vlm_lora_rank": 2,
    "vlm_mem_knn": 16,
    "vlm_text_span": 200,
    "vlm_router_pool": "anchored",
    "vlm_router_pool_weights": [1.0, 0.5],
    "vlm_route_once": True,
    "vlm_image_regions": 0,
    "vlm_image_pool_weights": [1.0, 0.5],
    "router_only_fast": False,
    "use_frozen_base_input_features": True,
    "frozen_prepass": True,
    "share_groups": [[4, 6], [8, 10]],
    "vlm_share_groups": [[5, 7], [9, 11], [13, 15]],
    "log_usage": True,
    "aggregate_usage": False,
    "mem_heads": 4,
    "mem_k_dim": 512,
    "value_fixed_lr": 0.001,
    "memory_lr": 0.001,
    "lang_to_query": False,
    "expert_anchor_pool": "text",
    "expert_anchor_weight": 0.40,
    "fuse_method": "film",
    "embedding_model": "all-mpnet-base-v2",
    "value_type": "lora",
    "contrastive_method": "sample",
    "contrastive_loss_weight": 0.05,
    "contrastive_margin": 0.0,
    "contrastive_query_queue": 512,
    "routing_intra_task_locality_weight": 0,
    "routing_inter_task_separation_weight": 8.0,
    "routing_query_queue": 512,
}


def build_policy_cfg():
    if CKPT:
        cfg = PreTrainedConfig.from_pretrained(CKPT, cli_overrides=[])
        cfg.pretrained_path = CKPT
        cfg.device = DEVICE
        return cfg
    valid = {f.name for f in dataclasses.fields(MemoryLayerConfig)}
    dropped = [k for k in RW_MEM if k not in valid]
    assert not dropped, f"MemoryLayerConfig no longer has fields {dropped}"
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.pretrained_path = BASE
    cfg.device = DEVICE
    cfg.dtype = "bfloat16"
    cfg.empty_cameras = 1
    cfg.memory_layers = True
    cfg.memory_layer = MemoryLayerConfig(**RW_MEM)
    cfg.train_memory_only = True  # as saved in the sequential-stage checkpoint config
    cfg.freeze_memory_router = True
    cfg.train_router_only = False
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    # RW dataset features after rename_map: 2 real cams (480x640) + 7-D state/action.
    cfg.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return cfg


def load_via_record_script(policy_cfg):
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        policy = _load_policy(SimpleNamespace(policy=policy_cfg, device=DEVICE))
    log = buf.getvalue()
    print(log)
    print(f"[smoke] _load_policy took {time.time() - t0:.1f}s")
    return policy, log


def make_batch(policy_cfg, dev):
    """Batch of 1 with pi05's prompt token structure; images/state random."""
    max_len = policy_cfg.tokenizer_max_length
    g = torch.Generator().manual_seed(0)

    def rnd(n):
        return torch.randint(1000, 20000, (n,), generator=g)

    toks = torch.cat(
        [
            torch.tensor([2]),
            rnd(2),  # <bos> Task :
            rnd(10),  # instruction tokens (positions 3..12)
            torch.tensor([235269]),  # ","  -> instr_len boundary (position 13)
            torch.tensor([3040]),  # "▁State" marker
            rnd(1),  # ":"
            rnd(32),  # discretised state bins
            rnd(4),  # ";" "\n" "Action" ":"
        ]
    )
    n = toks.numel()
    tokens = torch.zeros(1, max_len, dtype=torch.long)
    tokens[0, :n] = toks
    masks = torch.zeros(1, max_len, dtype=torch.bool)
    masks[0, :n] = True
    batch = {
        OBS_LANGUAGE_TOKENS: tokens.to(dev),
        OBS_LANGUAGE_ATTENTION_MASK: masks.to(dev),
        "observation.state": torch.randn(
            1, policy_cfg.input_features["observation.state"].shape[0], device=dev
        ),
        "task": "pick up the red brick",
        "robot_type": "widowxai_follower_robot",
    }
    for k, feat in policy_cfg.input_features.items():
        if feat.type == FeatureType.VISUAL and "empty_camera" not in k:
            batch[k] = torch.rand(1, *feat.shape, device=dev)
    return batch


def run_inference_checks(policy, policy_cfg):
    dev = torch.device(DEVICE)
    autocast_ctx = torch.autocast(device_type=dev.type) if policy.config.use_amp else nullcontext()
    for i in range(3):
        if i == 2:
            policy.reset()  # new episode
        t1 = time.time()
        with torch.inference_mode(), autocast_ctx:
            a = policy.select_action(make_batch(policy_cfg, dev))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        assert torch.isfinite(a).all(), "non-finite action"
        print(f"[smoke] select_action #{i}: shape={tuple(a.shape)} finite=True dt={time.time() - t1:.2f}s")
    # memory engaged? compare against every memory module bypassed (frozen-capture = plain MLP)
    with torch.inference_mode():
        policy.reset()
        ref = policy.select_action(make_batch(policy_cfg, dev)).float().cpu()
        mods = [m for m in policy.modules() if isinstance(m, MLPPlusMemory)]
        for m in mods:
            m._frozen_capture = True
        policy.reset()
        nomem = policy.select_action(make_batch(policy_cfg, dev)).float().cpu()
        for m in mods:
            m._frozen_capture = False
            m._frozen_stash = []
    print(f"[smoke] |action(mem) - action(no-mem)| max = {float((ref - nomem).abs().max()):.3e} (expect > 0)")
    if dev.type == "cuda":
        print(f"[smoke] peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return ref


def main():
    policy_cfg = build_policy_cfg()
    policy, log = load_via_record_script(policy_cfg)
    assert "Could not load state dict" not in log
    pwe = policy.model.paligemma_with_expert
    exp_mem = [
        i for i, layer in enumerate(pwe.gemma_expert.model.layers) if isinstance(layer.mlp, MLPPlusMemory)
    ]
    vlm_mem = [
        i
        for i, layer in enumerate(pwe.paligemma.model.language_model.layers)
        if isinstance(layer.mlp, MLPPlusMemory)
    ]
    print(
        f"[smoke] expert memory layers {exp_mem} | vlm memory layers {vlm_mem} | "
        f"prepass={pwe._frozen_prepass_enabled()} frozen_routing={pwe._frozen_routing_enabled()}"
    )
    if not CKPT:
        # fresh memory: give slot_up non-zero values so the value path is numerically live
        with torch.no_grad():
            seen = set()
            for m in policy.modules():
                if isinstance(m, MLPPlusMemory) and id(m.mem.slot_up) not in seen:
                    seen.add(id(m.mem.slot_up))
                    torch.nn.init.normal_(m.mem.slot_up, std=0.02)
        print(f"[smoke] randomised {len(seen)} distinct slot_up tables")
    if DEVICE == "cuda":
        print(f"[smoke] GPU after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    a_before = run_inference_checks(policy, policy_cfg)
    print("[smoke] MAIN OK")
    if MODE != "roundtrip":
        return

    # ---------------- round trip through a memory checkpoint ----------------
    tmp = tempfile.mkdtemp(prefix="smoke_record_ckpt_")
    try:
        torch.manual_seed(123)
        with torch.inference_mode():
            policy.reset()
            a_before = policy.select_action(make_batch(policy_cfg, torch.device(DEVICE))).float().cpu()
        policy.save_pretrained(tmp)
        del policy
        torch.cuda.empty_cache()
        cfg2 = PreTrainedConfig.from_pretrained(tmp, cli_overrides=[])
        cfg2.pretrained_path = tmp
        cfg2.device = DEVICE
        policy, log2 = load_via_record_script(cfg2)
        assert "Could not load state dict" not in log2
        # The policy was saved AFTER the record loader replaced both lm_heads with Identity, so
        # those two keys are legitimately absent from this checkpoint (a trainer-saved checkpoint
        # keeps them). Everything else — every memory tensor, shared tables under either name —
        # must load, and the alias-aware diagnostic must have recognised the shared storage.
        assert "Unexpected keys" not in log2, log2
        assert "memory param keys initialized from scratch" not in log2, "memory tensors did not load"
        assert "shared-storage alias keys already loaded" in log2, "alias diagnostic missing"
        missing_lines = [line.strip() for line in log2.splitlines() if line.strip().startswith("- ")]
        assert all(line.endswith("lm_head.weight") for line in missing_lines), missing_lines
        pwe = policy.model.paligemma_with_expert
        exp_layers = pwe.gemma_expert.model.layers
        vlm_layers = pwe.paligemma.model.language_model.layers
        for a, b in cfg2.memory_layer.share_groups:
            assert exp_layers[b].mlp.mem.slot_up is exp_layers[a].mlp.mem.slot_up
            assert exp_layers[b].mlp.mem.keys is exp_layers[a].mlp.mem.keys
        for a, b in cfg2.memory_layer.vlm_share_groups:
            assert vlm_layers[b].mlp.mem.slot_up is vlm_layers[a].mlp.mem.slot_up
        torch.manual_seed(123)
        with torch.inference_mode():
            policy.reset()
            a_after = policy.select_action(make_batch(cfg2, torch.device(DEVICE))).float().cpu()
        diff = float((a_before - a_after).abs().max())
        print(f"[rt] action before save vs after reload: max|diff| = {diff:.3e}")
        assert diff == 0.0, "reloaded memory checkpoint is not numerically identical"
        del policy
        torch.cuda.empty_cache()

        # negative 1: offload override on a shared-table checkpoint must fail at CONFIG time
        try:
            PreTrainedConfig.from_pretrained(tmp, cli_overrides=["--memory_layer.offload_slots_to_cpu=true"])
            raise AssertionError("offload + share_groups did not raise at config time")
        except AssertionError:
            raise
        except Exception as e:  # draccus wraps the dataclass ValueError in a ParsingError
            chain, cur = [], e
            while cur is not None:
                chain.append(str(cur))
                cur = cur.__cause__ or cur.__context__
            assert any("offload_slots_to_cpu is unsupported with share_groups" in m for m in chain), chain
            print(f"[neg] offload+share_groups -> rejected at config time ({type(e).__name__}): ok")

        # negative 2: a shape-changing override makes load_state_dict fail inside from_pretrained's
        # try/except; the record script's verification must refuse the policy.
        cfg3 = PreTrainedConfig.from_pretrained(
            tmp, cli_overrides=[f"--memory_layer.mem_n_keys={N_KEYS * 2}"]
        )
        cfg3.pretrained_path = tmp
        cfg3.device = DEVICE
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _load_policy(SimpleNamespace(policy=cfg3, device=DEVICE))
            raise AssertionError("record loader accepted a policy whose weights did not load")
        except RuntimeError as e:
            print(f"[neg] shape-changing override -> record loader refused: {str(e)[:110]}...")
        print("[rt] ROUNDTRIP OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
