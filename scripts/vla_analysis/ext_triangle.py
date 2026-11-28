#!/usr/bin/env python
"""Summarise a sequential run's retention triangle and in-run instruments as markdown (research_log2 Entry 5 onwards).

Reads local copies of the small result files of a run (rsync them from runs/<NAME>/ on the cluster):
  <mem>/tri/s<seed>/b<row>.json             triseed rows: results[env][seed] = {pc, successes}
  <mem>/seq/eval/current_task_rollouts.jsonl in-training rollouts of the just-trained task
  <mem>/state/sequential_state_latest.pt     loss_baseline (just-trained MSE) and loss_eval_history (MSE per boundary)
and, optionally, LoRA specialists (<spec>/t<task>/eval.json) and a naive-sequential arm (<naive>, the same layout as
<mem>, with eval/seeds_<seed>.json for its final row).

  python scripts/vla_analysis/ext_triangle.py --mem outputs/analysis/ext20/memory --order '[26,59,...]' \
      [--spec outputs/analysis/ext20/spec] [--naive outputs/analysis/ext20/naive]
"""

import argparse
import glob
import json
import os
import re
from statistics import mean


def load_rows(tri_dir):
    """{row: {env: {seed: pc}}} from tri/s<seed>/b<row>.json."""
    rows = {}
    for f in glob.glob(os.path.join(tri_dir, "s*", "b*.json")):
        row = int(re.search(r"b(\d+)\.json$", f).group(1))
        d = json.load(open(f))
        for env, by_seed in d["results"].items():
            for seed, r in by_seed.items():
                rows.setdefault(row, {}).setdefault(int(env), {})[int(seed)] = float(r["pc"])
    return rows


def m(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def fmt(x, w=5):
    return f"{x:{w}.1f}" if x is not None else " " * (w - 2) + "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", required=True)
    ap.add_argument("--order", required=True, help="JSON list of env ids in training order")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--naive", default=None)
    a = ap.parse_args()
    order = json.loads(a.order)
    K = len(order)
    rows = load_rows(os.path.join(a.mem, "tri"))
    done = sorted(rows)
    seeds_per_row = {r: len({s for e in rows[r].values() for s in e}) for r in done}

    def cell(r, env):
        by_seed = rows.get(r, {}).get(env)
        return mean(by_seed.values()) if by_seed else None

    print(
        f"## Retention triangle ({len(done)} of {K} rows; seeds per row: "
        + ", ".join(f"{r}:{seeds_per_row[r]}" for r in done)
        + ")\n"
    )
    print(
        "Row k = the checkpoint after task k; columns = tasks in training order; success %, mean over seeds.\n"
    )
    print("| row | " + " | ".join(f"{j + 1}:{e}" for j, e in enumerate(order)) + " | prior mean |")
    print("|---" * (K + 2) + "|")
    for r in done:
        vals = [cell(r, e) if j < r else None for j, e in enumerate(order)]
        prior = m(vals[: r - 1]) if r > 1 else None
        print(
            f"| {r} | "
            + " | ".join(fmt(v).strip() if v is not None else "" for v in vals)
            + f" | {fmt(prior).strip() if prior is not None else '--'} |"
        )

    last = max((r for r in done if seeds_per_row[r] == max(seeds_per_row.values())), default=None)
    diag = {e: cell(j + 1, e) for j, e in enumerate(order)}
    print(
        "\n## Per task: just trained (diagonal) and latest complete row "
        f"(row {last}, {seeds_per_row.get(last, 0)} seeds)\n"
    )
    inrun = {}
    p = os.path.join(a.mem, "seq", "eval", "current_task_rollouts.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            inrun[int(d["env_task"])] = float(d["pc_success"])
    spec = {}
    if a.spec:
        for f in glob.glob(os.path.join(a.spec, "t*", "eval.json")):
            d = json.load(open(f))
            for env, by_seed in d["results"].items():
                spec[int(env)] = mean(float(r["pc"]) for r in by_seed.values())
    drift = {}
    sp = os.path.join(a.mem, "state", "sequential_state_latest.pt")
    if os.path.exists(sp):
        import torch

        s = torch.load(sp, map_location="cpu", weights_only=False)
        base = {int(k): float(v) for k, v in (s.get("loss_baseline") or {}).items()}
        hist = s.get("loss_eval_history") or []
        if hist:
            latest = {int(k): float(v) for k, v in hist[-1]["per_task"].items()}
            drift = {e: (100 * (latest[e] - base[e]) / base[e]) for e in latest if e in base and base[e] > 0}
    print(
        "| k | task | in-run (20 eps, seed 1000) | diagonal | latest row | latest - diagonal | specialist "
        "| loss drift % |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for j, e in enumerate(order):
        lr = cell(last, e) if last and j < last else None
        dg = diag.get(e)
        dd = (lr - dg) if (lr is not None and dg is not None and j + 1 < (last or 0)) else None
        print(
            f"| {j + 1} | {e} | {fmt(inrun.get(e)).strip()} | {fmt(dg).strip()} | {fmt(lr).strip()} | "
            f"{fmt(dd).strip()} | {fmt(spec.get(e)).strip()} | {fmt(drift.get(e)).strip()} |"
        )
    n_prev = [e for j, e in enumerate(order) if last and j + 1 < last]
    print("\n## Summary\n")
    print(f"- diagonal mean (tasks 1..{max(done) if done else 0}): {fmt(m([diag[e] for e in order]))}")
    if last:
        print(
            f"- latest complete row ({last}): mean over its {last} tasks {fmt(m([cell(last, e) for e in order[:last]]))}; "
            f"prior tasks (1..{last - 1}) {fmt(m([cell(last, e) for e in n_prev]))} against their diagonal "
            f"{fmt(m([diag[e] for e in n_prev]))}"
        )
    if inrun:
        both = [e for e in inrun if diag.get(e) is not None]
        print(
            f"- in-run rollouts mean {fmt(m(list(inrun.values())))} over {len(inrun)} tasks; on the {len(both)} with a "
            f"diagonal: in-run {fmt(m([inrun[e] for e in both]))} vs diagonal {fmt(m([diag[e] for e in both]))}"
        )
    if spec:
        sh = [e for e in spec if diag.get(e) is not None]
        print(
            f"- specialists mean {fmt(m(list(spec.values())))} over {len(spec)} tasks; on the {len(sh)} with a diagonal: "
            f"specialist {fmt(m([spec[e] for e in sh]))} vs diagonal {fmt(m([diag[e] for e in sh]))}"
        )
    if drift:
        print(
            f"- loss drift (latest MSE vs just-trained), mean over earlier tasks: "
            f"{fmt(m([v for e, v in drift.items() if e != order[len(drift) - 1]]))}%"
        )
    if a.naive:
        nv = {}
        for f in glob.glob(os.path.join(a.naive, "eval", "seeds_*.json")):
            d = json.load(open(f))
            for env, by_seed in d["results"].items():
                nv.setdefault(int(env), []).extend(float(r["pc"]) for r in by_seed.values())
        nin = {}
        q = os.path.join(a.naive, "seq", "eval", "current_task_rollouts.jsonl")
        if os.path.exists(q):
            for line in open(q):
                d = json.loads(line)
                nin[int(d["env_task"])] = float(d["pc_success"])
        if nv or nin:
            print(
                f"- naive sequential: final row mean {fmt(m([mean(v) for v in nv.values()]))} over {len(nv)} tasks; "
                f"in-run just-trained mean {fmt(m(list(nin.values())))} over {len(nin)} tasks"
            )


if __name__ == "__main__":
    main()
