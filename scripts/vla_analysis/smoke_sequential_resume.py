#!/usr/bin/env python
"""Smokes for cross-task sequential state persistence / resume (task-boundary recovery).

Context: an 11h run died at task 2/5 on 27 Jul 26 (logind RemoveIPC swept the DataLoader's
/dev/shm segments — see CLAUDE.md 9.5.1). Restarting was the only valid option because the
prior-usefulness protection store and the online-IDF accumulators live only in module-level
globals and were never checkpointed: a naive resume would have run the remaining tasks
unprotected and produced a plausible but meaningless number.

These smokes cover the mechanism that fixes that. Run:
    python scripts/vla_analysis/smoke_sequential_resume.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import lerobot.scripts.lerobot_sequential_train as T

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def _populate(n_slots=1024, n_mod=3, seed=0):
    """Fill the module globals with synthetic per-module state."""
    g = torch.Generator().manual_seed(seed)
    T._protect_usefulness_by_module.clear()
    T._online_idf_df_by_module.clear()
    T._online_idf_total_batches.clear()
    idf = {}
    for m in range(n_mod):
        k = f"expert.layers.{m}.mlp.mem"
        T._protect_usefulness_by_module[k] = torch.rand(n_slots, generator=g)
        T._online_idf_df_by_module[k] = torch.randint(0, 500, (n_slots,), generator=g).float()
        T._online_idf_total_batches[k] = 500 + m
        idf[k] = torch.rand(n_slots, generator=g)
    return idf


def main():
    tmp = Path(tempfile.mkdtemp(prefix="seqresume_"))
    try:
        print("\n=== S1: round-trip fidelity ===")
        idf = _populate()
        ref_u = {k: v.clone() for k, v in T._protect_usefulness_by_module.items()}
        ref_df = {k: v.clone() for k, v in T._online_idf_df_by_module.items()}
        ref_tb = dict(T._online_idf_total_batches)
        ref_idf = {k: v.clone() for k, v in idf.items()}
        ckpt = tmp / "checkpoints" / "010000"
        dt = T._save_sequential_state(
            ckpt, task_pos=1, online_task_ids=[0, 1, 2, 3, 4], global_step=10000,
            idf_by_module=idf, seen_env_task_ids=[4, 6],
            eval_bar_history=[{"trained_task_idx": 0, "per_task": {4: 55.0}}],
            loss_eval_history=[], loss_baseline={4: 0.11},
        )
        check("state file written", (ckpt / T.SEQUENTIAL_STATE_FILENAME).is_file())
        check("no .tmp leftovers", not list(ckpt.glob("*.tmp")))
        check("save is fast (<60s preemption window)", dt < 60, f"{dt:.3f}s")

        # wipe the globals as a fresh process would have them
        T._protect_usefulness_by_module.clear()
        T._online_idf_df_by_module.clear()
        T._online_idf_total_batches.clear()
        idf2 = {}
        st = T._load_sequential_state(ckpt, idf2, [0, 1, 2, 3, 4])
        check("task_pos restored", st["task_pos"] == 1)
        check("global_step restored", st["global_step"] == 10000)
        check("protection store keys", set(T._protect_usefulness_by_module) == set(ref_u))
        check("protection store EXACT",
              all(torch.equal(T._protect_usefulness_by_module[k], ref_u[k]) for k in ref_u))
        check("idf DF EXACT", all(torch.equal(T._online_idf_df_by_module[k], ref_df[k]) for k in ref_df))
        check("idf batch totals EXACT", T._online_idf_total_batches == ref_tb)
        check("idf vectors EXACT", all(torch.equal(idf2[k], ref_idf[k]) for k in ref_idf))
        check("eval histories restored",
              st["seen_env_task_ids"] == [4, 6] and st["eval_bar_history"][0]["per_task"][4] == 55.0)
        check("loss_baseline restored", st["loss_baseline"] == {4: 0.11})

        print("\n=== S2: refuses to resume without the state file (the silent-corruption guard) ===")
        empty = tmp / "checkpoints" / "no_state"
        empty.mkdir(parents=True, exist_ok=True)
        try:
            T._load_sequential_state(empty, {}, [0, 1, 2, 3, 4])
            check("missing file raises", False, "no exception")
        except FileNotFoundError as e:
            check("missing file raises FileNotFoundError", True)
            check("error explains the risk", "EMPTY prior-usefulness store" in str(e))

        print("\n=== S3: refuses a different task order ===")
        try:
            T._load_sequential_state(ckpt, {}, [0, 1, 2])
            check("task_ids mismatch raises", False, "no exception")
        except ValueError as e:
            check("task_ids mismatch raises ValueError", True)
            check("error names both orders", "[0, 1, 2]" in str(e))

        print("\n=== S4: overwrite is atomic (old file intact until replace) ===")
        before = (ckpt / T.SEQUENTIAL_STATE_FILENAME).read_bytes()
        _populate(seed=99)
        T._save_sequential_state(
            ckpt, task_pos=2, online_task_ids=[0, 1, 2, 3, 4], global_step=15000,
            idf_by_module={}, seen_env_task_ids=[4, 6, 9],
            eval_bar_history=[], loss_eval_history=[], loss_baseline={},
        )
        after = (ckpt / T.SEQUENTIAL_STATE_FILENAME).read_bytes()
        check("file replaced, not appended", before != after)
        check("no .tmp leftovers after overwrite", not list(ckpt.glob("*.tmp")))
        st2 = T._load_sequential_state(ckpt, {}, [0, 1, 2, 3, 4])
        check("overwritten state reads back", st2["task_pos"] == 2 and st2["global_step"] == 15000)

        print("\n=== S5: RNG restoration ===")
        torch.manual_seed(1234)
        np.random.seed(1234)
        ck2 = tmp / "checkpoints" / "rng"
        T._save_sequential_state(
            ck2, task_pos=0, online_task_ids=[0], global_step=5000, idf_by_module={},
            seen_env_task_ids=[], eval_bar_history=[], loss_eval_history=[], loss_baseline={},
        )
        expect_t = torch.randn(4)
        expect_n = np.random.rand(4)
        torch.manual_seed(999)
        np.random.seed(999)
        T._load_sequential_state(ck2, {}, [0])
        check("torch RNG stream resumes", torch.equal(torch.randn(4), expect_t))
        check("numpy RNG stream resumes", np.allclose(np.random.rand(4), expect_n))

        print("\n=== S6: config default is legacy-safe ===")
        check("resume_sequential defaults False",
              T.SequentialOnlineConfig.__dataclass_fields__["resume_sequential"].default is False)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
