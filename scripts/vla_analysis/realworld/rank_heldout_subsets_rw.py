#!/usr/bin/env python3
"""Rank K-task held-out subsets of the real-world pool from a task_geometry_*.json
(output of probe_task_geometry_rw.py). CPU only.

Similarity: for each DEPLOYED router-input space (expert MLP inputs at the 6 expert sites,
instruction pools at the 6 expert-anchor source layers, the (1.0,0.5) VLM keys at the 6 VLM
sites) the off-diagonal task-centroid cosines are z-scored within the space (absolute cosine
levels differ by space); Z = mean z over spaces. Per group (expert / anchor / vlm-key) z-means
are reported too.

Per subset H (|H| = K):
  coll_max  = max_{a<b in H} Z(a,b)          collision risk inside the sequential set (LOW is good)
  coll_mean = mean pair Z inside H
  support   = mean_{h in H} max_{p not in H} Z(h,p)   each held-out task's closest pretrain relative (HIGH is good)
  length    = mean episode length (s); nmulti = # of tasks flagged multi-step (MULTISTEP env, default 9,13,14,15)

Prints: the most-similar pairs (sanity: the known red-bowl <-> blender-lid collision must be
near the top), per-task nearest neighbour, top subsets under three filters, and where the
historical v1/v3/v4 splits rank.
Env: K (5), MULTISTEP ("9,13,14,15"), KEEP (comma list forced into H, ""), EXCLUDE (""),
MAXLEN (s, 15), TOPN (20), SPLITS (json dict name->list, default v1/v3/v4).
"""
import itertools
import json
import os
import sys

import numpy as np

J = json.load(open(sys.argv[1]))
K = int(os.environ.get("K", "5"))
MULTISTEP = {int(x) for x in os.environ.get("MULTISTEP", "9,13,14,15").split(",") if x}
KEEP = {int(x) for x in os.environ.get("KEEP", "").split(",") if x}
EXCLUDE = {int(x) for x in os.environ.get("EXCLUDE", "").split(",") if x}
MAXLEN = float(os.environ.get("MAXLEN", "15"))
TOPN = int(os.environ.get("TOPN", "20"))
SPLITS = json.loads(os.environ.get("SPLITS", '{"v1":[0,1,4,10,12],"v3":[0,1,10,14,19],"v4":[0,1,10,11,19]}'))

names = {int(t): n for t, n in J["tasks"].items()}
tasks = sorted(names)
ti = {t: i for i, t in enumerate(tasks)}
fps = float(J.get("config", {}).get("fps") or 30.0)
ep_len = {int(t): v / fps for t, v in J.get("ep_len_mean_frames", {}).items()}
for t in tasks:
    ep_len.setdefault(t, float("nan"))

DEPLOYED = ([f"expert_L{L}" for L in (4, 6, 8, 10, 14, 16)]
            + [f"instr_L{L}" for L in (4, 6, 8, 10, 14, 16)]
            + [f"key_L{L}" for L in (5, 7, 9, 11, 13, 15)])
SPACES = [s for s in DEPLOYED if s in J["spaces"]]
missing = [s for s in DEPLOYED if s not in J["spaces"]]
if missing:
    print(f"[rank] WARNING missing spaces: {missing}")


def mat(name):
    d = J["spaces"][name]
    ts = [int(x) for x in d["tasks"]]
    M = np.array(d["cos"], dtype=np.float64)
    idx = {t: i for i, t in enumerate(ts)}
    out = np.full((len(tasks), len(tasks)), np.nan)
    for a in tasks:
        for b in tasks:
            if a in idx and b in idx:
                out[ti[a], ti[b]] = M[idx[a], idx[b]]
    return out


def zscore(M):
    iu = np.triu_indices(len(tasks), 1)
    v = M[iu]
    Z = (M - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    np.fill_diagonal(Z, np.nan)
    return Z


raw = {s: mat(s) for s in SPACES}
Zs = {s: zscore(raw[s]) for s in SPACES}
Z = np.nanmean(np.stack([Zs[s] for s in SPACES]), axis=0)
groups = {
    "expert": [s for s in SPACES if s.startswith("expert")],
    "anchor": [s for s in SPACES if s.startswith("instr")],
    "vlmkey": [s for s in SPACES if s.startswith("key")],
}
Zg = {g: np.nanmean(np.stack([Zs[s] for s in ss]), axis=0) for g, ss in groups.items() if ss}
Rg = {g: np.nanmean(np.stack([raw[s] for s in ss]), axis=0) for g, ss in groups.items() if ss}


def short(t, n=34):
    return names[t][:n]


def P(a, b, M):
    return float(M[ti[a], ti[b]])


print(f"[rank] spaces used: {len(SPACES)}  tasks: {len(tasks)}  K={K}")
print(f"[rank] usage per task: " + " ".join(f"{t}:{J['used'].get(str(t), J['used'].get(t, '?'))}" for t in tasks))

# --- most similar pairs overall
iu = np.triu_indices(len(tasks), 1)
order = np.argsort(-Z[iu])
print("\n=== most similar task pairs (Z = mean z over deployed spaces; raw mean cos per group) ===")
print(f"{'a':>2} {'b':>2} {'Z':>6} {'exp':>6} {'anc':>6} {'vlm':>6}  pair")
for o in order[:18]:
    a, b = tasks[iu[0][o]], tasks[iu[1][o]]
    print(f"{a:2d} {b:2d} {P(a,b,Z):6.2f} {P(a,b,Rg['expert']):6.3f} {P(a,b,Rg['anchor']):6.3f} {P(a,b,Rg['vlmkey']):6.3f}  {short(a)} | {short(b)}")

print("\n=== per-task nearest neighbour (Z) ===")
for t in tasks:
    row = Z[ti[t]].copy()
    row[ti[t]] = -np.inf
    j = int(np.nanargmax(row))
    print(f"{t:2d} {short(t,40):40s} nn={tasks[j]:2d} Z={row[j]:5.2f}  len={ep_len[t]:5.1f}s{'  MULTI' if t in MULTISTEP else ''}")

# --- subsets
rows = []
for H in itertools.combinations(tasks, K):
    if EXCLUDE & set(H):
        continue
    pairs = list(itertools.combinations(H, 2))
    zs = [P(a, b, Z) for a, b in pairs]
    Pset = [t for t in tasks if t not in H]
    support = float(np.mean([max(P(h, p, Z) for p in Pset) for h in H]))
    worst = pairs[int(np.argmax(zs))]
    rows.append({
        "H": H, "coll_max": max(zs), "coll_mean": float(np.mean(zs)), "support": support,
        "worst_pair": worst,
        "len": float(np.nanmean([ep_len[t] for t in H])),
        "nmulti": sum(t in MULTISTEP for t in H),
        **{f"max_{g}": max(P(a, b, Zg[g]) for a, b in pairs) for g in Zg},
    })
rows.sort(key=lambda r: (r["coll_max"], -r["support"]))
rank_of = {r["H"]: i + 1 for i, r in enumerate(rows)}


def show(title, filt, n=TOPN):
    print(f"\n=== {title} ===")
    print(f"{'rank':>5} {'coll_max':>8} {'coll_mean':>9} {'support':>7} {'len':>5} {'mult':>4} {'exp':>5} {'anc':>5} {'vlm':>5}  subset  (worst pair)")
    k = 0
    for r in rows:
        if not filt(r):
            continue
        k += 1
        print(f"{rank_of[r['H']]:5d} {r['coll_max']:8.2f} {r['coll_mean']:9.2f} {r['support']:7.2f} {r['len']:5.1f} {r['nmulti']:4d} "
              f"{r['max_expert']:5.2f} {r['max_anchor']:5.2f} {r['max_vlmkey']:5.2f}  {list(r['H'])}  {list(r['worst_pair'])}")
        if k >= n:
            break


show("A: pure geometry (lowest max within-subset similarity)", lambda r: True)
show(f"B: + no multi-step task, mean episode <= {MAXLEN:.0f}s", lambda r: r["nmulti"] == 0 and r["len"] <= MAXLEN)
if KEEP:
    show(f"C: B + forced {sorted(KEEP)}", lambda r: r["nmulti"] == 0 and r["len"] <= MAXLEN and KEEP <= set(r["H"]))

print("\n=== historical splits ===")
for nm, H in SPLITS.items():
    H = tuple(sorted(int(x) for x in H))
    r = next((r for r in rows if r["H"] == H), None)
    if r is None:
        print(f"{nm}: {list(H)} not scored (excluded)")
        continue
    print(f"{nm}: {list(H)} rank {rank_of[H]}/{len(rows)}  coll_max {r['coll_max']:.2f} (worst {list(r['worst_pair'])})  "
          f"coll_mean {r['coll_mean']:.2f}  support {r['support']:.2f}  len {r['len']:.1f}s  multi {r['nmulti']}")
print("\n[rank] task ids: " + "; ".join(f"{t}={short(t,28)}" for t in tasks))
