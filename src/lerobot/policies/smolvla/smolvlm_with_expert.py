# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForImageTextToText,
        AutoProcessor,
        SmolVLMForConditionalGeneration,
    )
else:
    AutoConfig = None
    AutoModel = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    SmolVLMForConditionalGeneration = None

from lerobot.policies.modules.memory_lite import (
    MLPPlusMemory,
    attach_memory_to_layer_list,
    checkpoint_recompute_context_fn,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
    ):
        super().__init__()
        require_package("transformers", extra="smolvla")
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self._gradient_checkpointing = False
        self.set_requires_grad()

        # Memory state, filled by attach_memory (empty = no memory attached).
        self._mem_cfg = None
        self._mem_layer_indices: list[int] = []  # expert layers with memory
        self._vlm_mem_layer_indices: list[int] = []  # VLM text layers with memory
        self._anchor_hooks: list = []
        self._anchor_lock = False
        self._anchor_pos_mask = None
        self._frozen_prefix_kv = None
        self._frozen_prefix_kv_bs = -1
        self._newline_token_id = None

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()
            # Memory banks attached inside the frozen VLM still follow `mode`: their routing
            # losses, query queues and usage/TF-IDF statistics only run in training mode.
            for m in self.vlm.modules():
                if isinstance(m, MLPPlusMemory):
                    m.train(mode)

    # ---------------------------------------------------------------------------------------
    # Memory: the SmolVLA side of what pi05's PaliGemmaWithExpertModel does. Every knob is a
    # memory_layer.* config field with the same meaning as on pi05.
    # ---------------------------------------------------------------------------------------
    def _vlm_layer_of_expert(self, expert_idx: int) -> int:
        """The VLM layer an expert layer runs alongside (see get_model_layers)."""
        return expert_idx * (self.num_vlm_layers // self.num_expert_layers)

    def attach_memory(self, cfg) -> None:
        """Attach memory to the expert (cfg.layers) and VLM text (cfg.vlm_layers) MLPs.

        Mirrors pi05's attach_memory_to_expert: the expert tower reads cfg directly; the VLM
        tower gets a derived cfg from the vlm_* fields (bank, rank, knn, share groups, span).
        Idempotent, so the from_pretrained path can call it before and after loading.
        """
        text_layers = self.get_vlm_model().text_model.layers
        vlm_dim = int(self.config.text_config.hidden_size)
        anchor = str(getattr(cfg, "expert_anchor_pool", "") or "")
        frozen = bool(getattr(cfg, "use_frozen_base_input_features", False))
        prepass = bool(getattr(cfg, "frozen_prepass", False))
        if anchor and int(getattr(cfg, "expert_anchor_src_dim", 2048)) != vlm_dim:
            raise ValueError(
                f"memory_layer.expert_anchor_src_dim={getattr(cfg, 'expert_anchor_src_dim', 2048)} must equal "
                f"the SmolVLM text hidden size ({vlm_dim})"
            )
        if prepass and not frozen:
            raise ValueError("memory_layer.frozen_prepass requires use_frozen_base_input_features=true (as on pi05)")
        if frozen and getattr(cfg, "memory_only", False):
            raise ValueError("use_frozen_base_input_features is incompatible with memory_layer.memory_only")

        attach_memory_to_layer_list(self.lm_expert.layers, dim=self.expert_hidden_size, cfg=cfg, label="EXPERT")
        self._mem_cfg = cfg
        self._mem_layer_indices = sorted(
            i for i, layer in enumerate(self.lm_expert.layers) if isinstance(layer.mlp, MLPPlusMemory)
        )
        self.mem_target_layers = list(self._mem_layer_indices)

        # ---- VLM tower ----
        vlm_layers = list(getattr(cfg, "vlm_layers", []) or [])
        span = int(getattr(cfg, "vlm_text_span", 0) or 0)
        if vlm_layers:
            # Placement guard (pi05): VLM memory at or below an expert memory layer changes the
            # prefix KV the expert reads, so live expert routing would see memory content.
            exp_max = max((self._vlm_layer_of_expert(e) for e in self._mem_layer_indices), default=-1)
            if min(vlm_layers) <= exp_max and not prepass:
                if getattr(cfg, "allow_nonstationary_routing", False):
                    logging.warning(
                        f"LIVE ADDRESSING (allow_nonstationary_routing): interleaved placement expert "
                        f"{self._mem_layer_indices} / VLM {vlm_layers} without the frozen pre-pass."
                    )
                else:
                    raise ValueError(
                        f"vlm_layers {vlm_layers} must all sit above the highest expert memory layer "
                        f"(VLM depth {exp_max}) unless memory_layer.frozen_prepass=true (as on pi05)"
                    )
        vlm_cfg = dataclasses.replace(
            cfg,
            layers=vlm_layers,
            mem_n_keys=int(getattr(cfg, "vlm_mem_n_keys", 256)),
            lora_rank=int(getattr(cfg, "vlm_lora_rank", 2)),
            mem_knn=int(getattr(cfg, "vlm_mem_knn", 16)),
            routing_loss_topk=int(getattr(cfg, "vlm_mem_knn", 16)),
            layer_ranks=[],
            lang_to_query=False,
            use_frozen_base_input_features=False,
            frozen_prepass=False,
            # > 0 turns on the instruction + state span; the length itself is set per batch
            # by set_memory_context. 0 = memory on every prefix position (images included).
            text_span=span,
            image_regions=0,
            value_input_noise_sigma=list(getattr(cfg, "vlm_value_input_noise_sigma", []) or []),
            vlm_value_input_noise_sigma=[],
            vlm_layers=[],
            expert_anchor_pool="",
            share_groups=list(getattr(cfg, "vlm_share_groups", []) or []),
            vlm_share_groups=[],
        )
        if int(getattr(cfg, "vlm_image_regions", 0) or 0):
            raise NotImplementedError("memory_layer.vlm_image_regions is not supported on SmolVLA")
        attach_memory_to_layer_list(text_layers, dim=vlm_dim, cfg=vlm_cfg, label="VLM")
        self._vlm_mem_layer_indices = sorted(
            i for i, layer in enumerate(text_layers) if isinstance(layer.mlp, MLPPlusMemory)
        )
        self.vlm_mem_target_layers = list(self._vlm_mem_layer_indices)
        for i in self._vlm_mem_layer_indices:
            text_layers[i].mlp._span_gather_only = span > 0

        # ---- expert anchor: expert layer e routes on the pooled instruction hidden of the VLM
        # layer it runs alongside (pi05 pairs expert layer j with LM layer j). The hook sits on
        # that VLM layer's MLP (or its memory wrapper) and sees the post-attention-norm input.
        for h in self._anchor_hooks:
            h.remove()
        self._anchor_hooks = []
        if anchor:
            for e in self._mem_layer_indices:
                wrapper = self.lm_expert.layers[e].mlp
                vlm_mlp = text_layers[self._vlm_layer_of_expert(e)].mlp
                self._anchor_hooks.append(vlm_mlp.register_forward_pre_hook(self._make_anchor_hook(wrapper)))
        logging.info(
            f"SmolVLA memory: expert layers {self._mem_layer_indices}, VLM layers {self._vlm_mem_layer_indices} "
            f"(span={'instruction+state' if span > 0 else 'all positions'}), anchor={anchor or 'off'}, "
            f"frozen routing={'pre-pass' if frozen else 'off'}"
        )

    def _make_anchor_hook(self, wrapper):
        def hook(mod, args):
            if self._anchor_lock:
                return  # captured by the memory-free pre-pass; the live pass must not overwrite it
            x = args[0]
            pm = self._anchor_pos_mask
            if pm is None:
                raise RuntimeError("expert anchor: no instruction mask (set_memory_context not called)")
            if x.dim() != 3 or tuple(x.shape[:2]) != tuple(pm.shape):
                raise RuntimeError(f"expert anchor: prefix {tuple(x.shape)} vs instruction mask {tuple(pm.shape)}")
            with torch.no_grad():
                m = pm.to(device=x.device).unsqueeze(-1).float()
                pooled = (x.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
                valid = pm.to(device=x.device).sum(dim=1) > 0
            wrapper.set_expert_anchor(pooled.detach(), valid)

        return hook

    def _get_newline_token_id(self) -> int:
        if self._newline_token_id is None:
            tok = self.processor.tokenizer
            ids = tok.convert_tokens_to_ids(tok.tokenize("\n"))
            if len(ids) != 1:
                raise RuntimeError(f"expected a single newline token, got {ids}")
            self._newline_token_id = int(ids[0])
        return self._newline_token_id

    def set_memory_context(
        self, lang_tokens: torch.Tensor, lang_masks: torch.Tensor, lang_start: int, n_state: int, prefix_len: int
    ) -> None:
        """Per-batch positions for the VLM span memory and the expert anchors.

        SmolVLA's prefix is [images | instruction tokens, right-padded | state token(s) | pad].
        The instruction always ends with the processor's newline token. The layout handed to
        the span memory mirrors pi05's language field: instruction tokens (routed per token),
        then the state region = the newline marker + the state token(s), which share one pooled
        key under vlm_router_pool; the pools skip the newline, as pi05's skip its markers.
        The expert anchor pools the same instruction tokens.
        """
        span_wrappers = [
            m for _, m in self._memory_wrappers()[1] if int(getattr(m, "text_span", 0) or 0) > 0
        ]
        if not span_wrappers and not self._anchor_hooks:
            return
        lm = lang_masks.bool()
        B, L = lm.shape
        dev = lm.device
        n_valid = lm.sum(dim=1)
        pos = torch.arange(L, device=dev).unsqueeze(0)
        if not bool((lm == (pos < n_valid.unsqueeze(1))).all()):
            raise ValueError("memory span: language tokens must be right-padded")
        last = lang_tokens.gather(1, (n_valid - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        has_nl = (n_valid > 0) & (last == self._get_newline_token_id())
        instr_len = n_valid - has_nl.long()
        if self._anchor_hooks:
            pm = torch.zeros(B, prefix_len, dtype=torch.bool, device=dev)
            pm[:, lang_start : lang_start + L] = pos < instr_len.unsqueeze(1)
            self._anchor_pos_mask = pm
        if span_wrappers:
            state_start = lang_start + L
            t = torch.arange(L + n_state, device=dev).unsqueeze(0)
            nv = n_valid.unsqueeze(1)
            idx = torch.where(t < nv, lang_start + t, state_start + (t - nv).clamp(0, n_state - 1))
            valid = t < nv + n_state
            skips = (0, has_nl.long(), 0)  # (instruction head, state-region head, state-region tail)
            for m in span_wrappers:
                m._ctx_span_index = idx
                m._ctx_valid_mask = valid
                m._ctx_instr_len = instr_len
                m._ctx_span_skips = skips

    def _frozen_routing_on(self) -> bool:
        cfg = self._mem_cfg
        return bool(
            cfg is not None
            and getattr(cfg, "use_frozen_base_input_features", False)
            and (self._mem_layer_indices or self._vlm_mem_layer_indices)
        )

    def _memory_wrappers(self):
        """([(loop layer index, expert wrapper)], [(loop layer index, VLM wrapper)])."""
        text_layers = self.get_vlm_model().text_model.layers
        exp = [(self._vlm_layer_of_expert(e), self.lm_expert.layers[e].mlp) for e in self._mem_layer_indices]
        vlm = [(i, text_layers[i].mlp) for i in self._vlm_mem_layer_indices]
        return exp, vlm

    def _capture(self, wrappers, run):
        """Run `run()` memory-free with `wrappers` stashing their MLP inputs; returns its result."""
        for _, m in wrappers:
            m.begin_frozen_capture()
        try:
            with torch.no_grad():
                out = run()
        except Exception:
            for _, m in wrappers:
                m._frozen_capture = False
                m._frozen_stash = []
            self._anchor_lock = False
            raise
        for _, m in wrappers:
            m.end_frozen_capture()
        return out

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce activation memory during training."""
        self._gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for SmolVLMWithExpertModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for SmolVLMWithExpertModel")

    def _compute_layer(
        self,
        layer_idx,
        inputs_embeds,
        model_layers,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache,
        fill_kv_cache,
        past_key_values,
        task_emb,
        task_ids,
        router_xs=None,
    ):
        """Compute a single transformer layer (attention + post-attention) for all active streams.

        router_xs: optional (vlm_router_x, expert_router_x) for this layer: the memory-free MLP
        inputs from the frozen pre-pass, handed to memory wrappers as their routing features.
        """
        # --- Attention ---
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, past_key_values = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )

        # --- Post-attention: o_proj, residual, layernorm, MLP/memory, residual ---
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = (
                att_outputs[i] if i < len(att_outputs) else att_outputs[0]
            )  # in case of self_attn
            if hidden_states is not None:
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)

                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                out_emb = layer.post_attention_layernorm(out_emb)
                if isinstance(layer.mlp, MLPPlusMemory):
                    rx = router_xs[i] if router_xs is not None else None
                    out_emb = layer.mlp(out_emb, lang_emb=task_emb, task_ids=task_ids, router_x=rx)
                else:
                    out_emb = layer.mlp(out_emb)

                out_emb += after_first_residual

                outputs_embeds.append(out_emb)

                start = end if len(att_outputs) == 1 else 0
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def _compute_layer_checkpointable(
        self,
        layer_idx,
        vlm_hidden,
        expert_hidden,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        task_emb,
        task_ids,
        model_layers=None,
        vlm_router_x=None,
        expert_router_x=None,
    ):
        """Checkpoint-friendly wrapper around _compute_layer (training only, no cache).

        Router inputs are explicit arguments so they re-thread through the recompute."""
        inputs_embeds = [vlm_hidden, expert_hidden]
        router_xs = None
        if vlm_router_x is not None or expert_router_x is not None:
            router_xs = (vlm_router_x, expert_router_x)
        outputs_embeds, _ = self._compute_layer(
            layer_idx, inputs_embeds, model_layers,
            position_ids, attention_mask, batch_size, head_dim,
            use_cache=False, fill_kv_cache=False, past_key_values=None,
            task_emb=task_emb, task_ids=task_ids, router_xs=router_xs,
        )
        return outputs_embeds[0], outputs_embeds[1]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        task_emb: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim

        use_gradient_checkpointing = (
            self._gradient_checkpointing and self.training and not use_cache
        )

        def run_layers(embeds, pkv, use_cache_, fill_kv_cache_):
            # Memory-free pass over every layer; used by the frozen pre-pass (wrappers in capture mode).
            for li in range(num_layers):
                embeds, pkv = self._compute_layer(
                    li, embeds, model_layers, position_ids, attention_mask, batch_size, head_dim,
                    use_cache_, fill_kv_cache_, pkv, task_emb, task_ids,
                )
            return pkv

        # Frozen-base routing (memory_layer.use_frozen_base_input_features): every memory wrapper
        # routes (query + gate, and the anchors) on the features the network has WITHOUT memory.
        # They come from one memory-bypassed pre-pass per forward, as pi05's frozen_prepass does.
        frozen = self._frozen_routing_on()
        if not frozen:
            self._anchor_lock = False  # anchors follow the live stream
        prefix_only = inputs_embeds[1] is None
        suffix_only = inputs_embeds[0] is None
        vlm_rx: dict[int, torch.Tensor] = {}
        exp_rx: dict[int, torch.Tensor] = {}
        live_popped = []  # wrappers whose stash the live pass must consume (inference paths)
        exp_wr, vlm_wr = self._memory_wrappers()
        if frozen and not prefix_only and not suffix_only:
            # Training / joint path: drain the stashes into explicit router_x arguments, which
            # re-thread safely through gradient-checkpoint recompute.
            self._anchor_lock = False
            self._capture(exp_wr + vlm_wr, lambda: run_layers(list(inputs_embeds), None, False, False))
            vlm_rx = {li: m.drain_frozen_stash() for li, m in vlm_wr}
            exp_rx = {li: m.drain_frozen_stash() for li, m in exp_wr}
            self._anchor_lock = True
        elif frozen and prefix_only:
            # Inference prefix pass. With VLM memory, a memory-free prefix pass stashes the VLM
            # routing features, captures the anchors (then locked) and keeps its KV cache: the
            # expert's own memory-free pass (denoise steps) must attend that, not the live KV.
            self._anchor_lock = False
            self._frozen_prefix_kv = None
            self._frozen_prefix_kv_bs = -1
            if vlm_wr:
                fkv = self._capture(vlm_wr, lambda: run_layers(list(inputs_embeds), None, True, True))
                self._frozen_prefix_kv = fkv
                self._frozen_prefix_kv_bs = int(inputs_embeds[0].shape[0])
                self._anchor_lock = True
                live_popped = vlm_wr
        elif frozen and suffix_only and exp_wr:
            # Denoise step: the expert's memory-free pass, attending the memory-free prefix KV.
            kv_src = past_key_values
            if vlm_wr:
                if self._frozen_prefix_kv is None or self._frozen_prefix_kv_bs != int(inputs_embeds[1].shape[0]):
                    raise RuntimeError(
                        "frozen routing: no memory-free prefix KV for this batch; the prefix pass "
                        "must run before the denoise steps"
                    )
                kv_src = self._frozen_prefix_kv
            self._capture(exp_wr, lambda: run_layers(list(inputs_embeds), kv_src, use_cache, False))
            live_popped = exp_wr

        for layer_idx in range(num_layers):
            vrx = vlm_rx.get(layer_idx)
            erx = exp_rx.get(layer_idx)
            if use_gradient_checkpointing:
                vlm_out, expert_out = torch.utils.checkpoint.checkpoint(
                    self._compute_layer_checkpointable,
                    layer_idx, inputs_embeds[0], inputs_embeds[1],
                    position_ids, attention_mask,
                    batch_size, head_dim, task_emb, task_ids,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    context_fn=checkpoint_recompute_context_fn,
                    model_layers=model_layers,
                    vlm_router_x=vrx,
                    expert_router_x=erx,
                )
                inputs_embeds = [vlm_out, expert_out]
            else:
                router_xs = (vrx, erx) if (vrx is not None or erx is not None) else None
                inputs_embeds, past_key_values = self._compute_layer(
                    layer_idx, inputs_embeds, model_layers,
                    position_ids, attention_mask, batch_size, head_dim,
                    use_cache, fill_kv_cache, past_key_values,
                    task_emb, task_ids, router_xs=router_xs,
                )
        for _, m in live_popped:
            m.assert_frozen_stash_consumed()

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
