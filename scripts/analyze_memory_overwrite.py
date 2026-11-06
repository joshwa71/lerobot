#!/usr/bin/env python3
"""Read-time overwrite / forgetting analysis from per-task memory_usage JSONs.

For each sequential run we compute, per task and per layer:
  - read footprint  (slots with total_accesses>0, total read mass)
  - update footprint (slots with total_updates>0, total update mass)
  - read-through-overwrite: fraction of a task's READ weight that lands on slots
    UPDATED BY LATER tasks (the dominant forgetting channel)
  - pairwise overwrite matrix M[X<-Y] = frac of X's read weight on Y's updates
  - mean pairwise weighted-read Jaccard (validates the logged memory_iou)
"""
import sys, os, json, gc
import numpy as np

LAYERS = [8, 10, 12, 14]
PREFIX = "model.paligemma_with_expert.gemma_expert.model.layers."
TASK_IDX = list(range(10, 20))                 # dataset task indices
# ds_to_env_map_json from the job script (dataset task_index -> libero_goal env id)
ENV = {10: 8, 11: 9, 12: 3, 13: 6, 14: 2, 15: 5, 16: 7, 17: 1, 18: 4, 19: 0}
# online_task_ids order == training order: sequential in task_index 10..19
ORDER_TIDX = list(range(10, 20))               # ord k == task_index (10+k)
ORD = {t: i for i, t in enumerate(ORDER_TIDX)} # ord position of each task index

def load_task(path):
    """Return {layer: (reads dict slot->acc, updates dict slot->upd)}."""
    d = json.load(open(path))
    pm = d["per_module"]
    out = {}
    for L in LAYERS:
        node = pm[PREFIX + str(L)]
        # single task key inside
        tkey = next(iter(node))
        slots = node[tkey]
        reads = {}
        updates = {}
        for sk, st in slots.items():
            sid = int(sk.rsplit("_", 1)[1])
            acc = st["total_accesses"]
            upd = st["total_updates"]
            if acc:
                reads[sid] = acc
            if upd:
                updates[sid] = upd
        out[L] = (reads, updates)
    del d
    return out

def analyze(run_dir, label):
    mbt = os.path.join(run_dir, "memory_by_task")
    data = {}  # tidx -> {L:(reads,updates)}
    for t in TASK_IDX:
        p = os.path.join(mbt, f"memory_usage_task_{t}.json")
        data[t] = load_task(p)
        gc.collect()

    print("=" * 96)
    print(f"RUN: {label}")
    print("=" * 96)

    # ---- per-task footprint ----
    print("\nPer-task footprint (summed over 4 layers):")
    print("  ord env tidx | read_slots  read_mass | upd_slots   upd_mass | L14_read_slots L14_upd_slots")
    for t in ORDER_TIDX:
        rs = sum(len(data[t][L][0]) for L in LAYERS)
        rm = sum(sum(data[t][L][0].values()) for L in LAYERS)
        us = sum(len(data[t][L][1]) for L in LAYERS)
        um = sum(sum(data[t][L][1].values()) for L in LAYERS)
        l14rs = len(data[t][14][0]); l14us = len(data[t][14][1])
        print(f"  {ORD[t]:>3d} {ENV[t]:>3d} {t:>4d} | {rs:>10d} {rm:>10d} | {us:>9d} {um:>10d} | {l14rs:>13d} {l14us:>13d}")

    # ---- read-through-overwrite: frac of task X reads on union of LATER updates ----
    print("\nRead-through-overwrite  (frac of task's READ weight on slots updated by LATER tasks):")
    print("  ord env | L8     L10    L12    L14   | mean4")
    rto_mean = {}
    for t in ORDER_TIDX:
        later = [u for u in ORDER_TIDX if ORD[u] > ORD[t]]
        per_layer = []
        for L in LAYERS:
            reads, _ = data[t][L]
            tot = sum(reads.values())
            if tot == 0:
                per_layer.append(0.0); continue
            later_upd = set()
            for u in later:
                later_upd |= set(data[u][L][1].keys())
            hit = sum(acc for s, acc in reads.items() if s in later_upd)
            per_layer.append(hit / tot)
        m = float(np.mean(per_layer))
        rto_mean[t] = m
        print(f"  {ORD[t]:>3d} {ENV[t]:>3d} | " + " ".join(f"{x:5.1%}" for x in per_layer) + f" | {m:5.1%}")
    print(f"  MEAN over tasks (excl. last):  {np.mean([rto_mean[t] for t in ORDER_TIDX[:-1]]):.1%}")

    # ---- pairwise overwrite matrix, mean over 4 layers: M[X<-Y] ----
    print("\nPairwise overwrite  M[X<-Y] = frac of X(row) reads on Y(col) updates  (mean over 4 layers).")
    print("  Rows/cols are ENV ids in TRAINING ORDER. Only Y trained AFTER X is meaningful.")
    hdr = "        " + "".join(f"{ENV[u]:>6d}" for u in ORDER_TIDX)
    print(hdr)
    big_pairs = []
    for t in ORDER_TIDX:
        row = []
        for u in ORDER_TIDX:
            if ORD[u] <= ORD[t]:
                row.append(None); continue
            vals = []
            for L in LAYERS:
                reads, _ = data[t][L]
                tot = sum(reads.values())
                upd = set(data[u][L][1].keys())
                hit = sum(acc for s, acc in reads.items() if s in upd) if tot else 0
                vals.append(hit / tot if tot else 0.0)
            mv = float(np.mean(vals))
            row.append(mv)
            if mv >= 0.12:
                big_pairs.append((ENV[t], ENV[u], mv, ORD[t], ORD[u]))
        cells = "".join("     ." if v is None else f"{v:6.1%}" for v in row)
        print(f"  e{ENV[t]:<2d}o{ORD[t]:<2d}{cells}")
    print("\n  Biggest overwrite channels (>=12%, X<-Y, Y later):")
    for x, y, v, ox, oy in sorted(big_pairs, key=lambda z: -z[2]):
        print(f"    env{x} (ord{ox}) <- env{y} (ord{oy}):  {v:.1%}")

    # ---- mean pairwise weighted-read Jaccard (validate logged IoU) ----
    def wjacc(a, b):
        keys = set(a) | set(b)
        inter = sum(min(a.get(s, 0), b.get(s, 0)) for s in (set(a) & set(b)))
        union = sum(max(a.get(s, 0), b.get(s, 0)) for s in keys)
        return inter / union if union else 0.0
    jac = []
    for i, t in enumerate(TASK_IDX):
        for u in TASK_IDX[i+1:]:
            per_layer = [wjacc(data[t][L][0], data[u][L][0]) for L in LAYERS]
            jac.append(np.mean(per_layer))
    print(f"\n  Mean pairwise weighted-read Jaccard (4-layer mean): {np.mean(jac):.4f}")

    # write-set binary IoU
    def bjacc(a, b):
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A | B) else 0.0
    wj = []
    for i, t in enumerate(TASK_IDX):
        for u in TASK_IDX[i+1:]:
            per_layer = [bjacc(data[t][L][1], data[u][L][1]) for L in LAYERS]
            wj.append(np.mean(per_layer))
    print(f"  Mean pairwise write-set binary IoU (4-layer mean):  {np.mean(wj):.4f}")
    del data
    gc.collect()

if __name__ == "__main__":
    BASE = "/home/josh/lerobot/outputs/train"
    RUNS = {
        "seq_t512":  f"{BASE}/libero_goal_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k",
        "seq_t1536": f"{BASE}/libero_goal_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k_top_t_1536",
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for name, d in RUNS.items():
        if which in (name, "both"):
            analyze(d, name)
            print()
