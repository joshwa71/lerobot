#!/usr/bin/env python3
"""Helpers for build_rw_split.sh (real-world pretrain / sequential split from the 20-task pool).

  patch-tasks <per_task_root> <task_index>
      split_dataset_by_task.py writes the FULL pool task table into every per-task dataset;
      merge_datasets.py builds the merged vocabulary by first appearance over each source's
      table, so merging such parts keeps the pool's 20-entry vocabulary (15 empty entries in
      the pretrain split, non-contiguous ids in the sequential split). Rewriting each part's
      table to its single task makes the merge renumber contiguously IN MERGE ORDER — which
      is how the sequential order is fixed (task_index 0..4 = training order).
  verify <dataset_root> [expected_n_tasks]
      prints the task table, per-task episode/frame counts (episodes meta) and the set of
      task_index values actually present in the data parquet files; exits 1 on any
      inconsistency (non-contiguous ids, data ids not in the table, count mismatch).
  manifest <out_json> <pool_root> <pretrain_root> <seq_root> <heldout_csv_in_order>
      writes the split manifest (pool ids <-> split ids, names, eps, frames).
"""
import glob
import json
import os
import sys

import pandas as pd


def _episodes_df(root):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _tasks_df(root):
    return pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))


def _data_task_ids(root):
    ids = set()
    for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
        ids |= set(int(x) for x in pd.read_parquet(f, columns=["task_index"])["task_index"].unique())
    return ids


def per_task_counts(root):
    e = _episodes_df(root)
    e["task"] = e["tasks"].apply(lambda x: list(x)[0])
    g = e.groupby("task")["length"].agg(["count", "sum"])
    return {t: (int(r["count"]), int(r["sum"])) for t, r in g.iterrows()}


def cmd_patch_tasks(root, task_index):
    t = _tasks_df(root)
    ti = int(task_index)
    keep = t[t["task_index"].astype(int) == ti]
    if len(keep) != 1:
        raise SystemExit(f"patch-tasks: expected exactly one row with task_index={ti} in {root}, got {len(keep)}")
    data_ids = _data_task_ids(root)
    if data_ids != {ti}:
        raise SystemExit(f"patch-tasks: data task_index values {sorted(data_ids)} != {{{ti}}} in {root}")
    keep.to_parquet(os.path.join(root, "meta", "tasks.parquet"))
    print(f"[patch-tasks] {root}: table -> 1 row ({keep.index[0]!r} -> {ti})")


def cmd_verify(root, expected_n=None):
    t = _tasks_df(root)
    names = {int(r["task_index"]): str(n) for n, r in t.iterrows()}
    ids = sorted(names)
    data_ids = _data_task_ids(root)
    counts = per_task_counts(root)
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    ok = True
    print(f"[verify] {root}")
    print(f"  info: episodes={info.get('total_episodes')} frames={info.get('total_frames')} tasks={info.get('total_tasks')} fps={info.get('fps')}")
    for i in ids:
        c = counts.get(names[i], (0, 0))
        print(f"  task_index {i:2d}  eps {c[0]:3d}  frames {c[1]:6d}  {names[i]}")
    if ids != list(range(len(ids))):
        print(f"  FAIL: task ids not contiguous from 0: {ids}"); ok = False
    if data_ids != set(ids):
        print(f"  FAIL: data task_index values {sorted(data_ids)} != table ids {ids}"); ok = False
    if expected_n is not None and len(ids) != int(expected_n):
        print(f"  FAIL: expected {expected_n} tasks, table has {len(ids)}"); ok = False
    tot_eps = sum(c[0] for c in counts.values()); tot_fr = sum(c[1] for c in counts.values())
    if tot_eps != int(info.get("total_episodes", -1)) or tot_fr != int(info.get("total_frames", -1)):
        print(f"  FAIL: episodes meta totals ({tot_eps}, {tot_fr}) != info.json"); ok = False
    print("  OK" if ok else "  VERIFY FAILED")
    sys.exit(0 if ok else 1)


def cmd_manifest(out, pool, pre, seq, heldout_csv):
    pool_t = _tasks_df(pool)
    pool_names = {int(r["task_index"]): str(n) for n, r in pool_t.iterrows()}
    pool_counts = per_task_counts(pool)
    heldout = [int(x) for x in heldout_csv.split(",")]
    pre_t = _tasks_df(pre); seq_t = _tasks_df(seq)
    pre_names = {int(r["task_index"]): str(n) for n, r in pre_t.iterrows()}
    seq_names = {int(r["task_index"]): str(n) for n, r in seq_t.iterrows()}
    name_to_pool = {v: k for k, v in pool_names.items()}
    m = {
        "pool": pool, "pretrain_root": pre, "seq_root": seq,
        "heldout_pool_ids_in_seq_order": heldout,
        "seq": [{"seq_task_index": i, "pool_task_index": name_to_pool[n], "task": n,
                 "eps": pool_counts[n][0], "frames": pool_counts[n][1]} for i, n in sorted(seq_names.items())],
        "pretrain": [{"pretrain_task_index": i, "pool_task_index": name_to_pool[n], "task": n,
                      "eps": pool_counts[n][0], "frames": pool_counts[n][1]} for i, n in sorted(pre_names.items())],
    }
    m["seq_frames"] = sum(x["frames"] for x in m["seq"]); m["seq_eps"] = sum(x["eps"] for x in m["seq"])
    m["pretrain_frames"] = sum(x["frames"] for x in m["pretrain"]); m["pretrain_eps"] = sum(x["eps"] for x in m["pretrain"])
    assert [x["pool_task_index"] for x in m["seq"]] == heldout, "sequential order does not match HELDOUT order"
    json.dump(m, open(out, "w"), indent=1)
    print(f"[manifest] {out}: pretrain {len(m['pretrain'])} tasks / {m['pretrain_eps']} eps / {m['pretrain_frames']} frames; "
          f"seq {len(m['seq'])} tasks / {m['seq_eps']} eps / {m['seq_frames']} frames; order {heldout}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "patch-tasks":
        cmd_patch_tasks(sys.argv[2], sys.argv[3])
    elif cmd == "verify":
        cmd_verify(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "manifest":
        cmd_manifest(*sys.argv[2:7])
    else:
        raise SystemExit(__doc__)
