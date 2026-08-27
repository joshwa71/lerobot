#!/usr/bin/env python3
"""Forgetting-matrix report for the real-world sequential run — RAW NUMBERS, no verdicts.

  inrun  <run_dir> [steps_per_task]
      reads <run_dir>/eval/loss_results.jsonl (the in-run --eval.type=loss instrument, rows
      {"step", "task_t", "forget_t"}; forget_t = task_t - just-trained baseline) and prints the
      step x task matrix, the just-trained baseline per task, the final value, delta and rel %.
  msemat <jsonl> <steps_per_task>
      reads an mse_matrix_rw.py / mse_matrix2.py output ({"run","ckpt","per_task"}) and prints
      the same table; just-trained for the i-th task = the row at ckpt (i+1)*steps_per_task.
OUT=<json> writes the tables as JSON as well.
"""
import json
import os
import sys


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def fmt_matrix(rows, tasks):
    lines = [f"{'ckpt/step':>10} " + " ".join(f"{'t' + str(t):>9}" for t in tasks)]
    for lab, d in rows:
        cells = []
        for t in tasks:
            v = d.get(t)
            cells.append(f"{v:9.5f}" if isinstance(v, (int, float)) else f"{'-':>9}")
        lines.append(f"{str(lab):>10} " + " ".join(cells))
    return "\n".join(lines)


def report(rows, tasks, just_trained, title):
    print(f"\n==== {title}")
    print(fmt_matrix(rows, tasks))
    _, final = rows[-1]
    print(f"\n{'task':>5} {'just-trained':>13} {'final':>10} {'delta':>10} {'rel%':>8}")
    out = {}
    for t in tasks:
        jt, fv = just_trained.get(t), final.get(t)
        if jt is None or fv is None:
            print(f"{t:>5} {'-':>13} {'-':>10}")
            continue
        d = fv - jt
        rel = 100.0 * d / jt if jt else float("nan")
        print(f"{t:>5} {jt:13.5f} {fv:10.5f} {d:+10.5f} {rel:+8.2f}")
        out[str(t)] = {"just_trained": jt, "final": fv, "delta": d, "rel_pct": rel}
    return out


def cmd_inrun(run_dir, spt):
    p = os.path.join(run_dir, "eval", "loss_results.jsonl")
    recs = sorted(load_jsonl(p), key=lambda r: r["step"])
    tasks = sorted({int(k.split("_")[1]) for r in recs for k in r if k.startswith("task_")})
    rows = [(r["step"], {t: r.get(f"task_{t}") for t in tasks}) for r in recs]
    jt = {}
    for r in recs:
        for t in tasks:
            if f"task_{t}" in r and f"forget_{t}" in r:
                jt[t] = r[f"task_{t}"] - r[f"forget_{t}"]
    print(f"[inrun] {p}: {len(recs)} rows, tasks {tasks}, steps_per_task {spt}")
    summ = report(rows, tasks, jt, "in-run --eval.type=loss matrix (paired-noise MSE over eval_loss_n_batches)")
    final = recs[-1]
    print("final-row forget_t (raw): " + "  ".join(
        f"t{t}:{final[f'forget_{t}']:+.5f}" if f"forget_{t}" in final else f"t{t}:-" for t in tasks))
    return {"source": p, "steps_per_task": spt, "rows": recs, "summary": summ}


def cmd_msemat(path, spt):
    recs = load_jsonl(path)
    tasks = sorted({int(t) for r in recs for t in r["per_task"]})
    rows = [(r["ckpt"], {int(t): v for t, v in r["per_task"].items()}) for r in recs]
    by_ckpt = {r["ckpt"]: {int(t): v for t, v in r["per_task"].items()} for r in recs}
    jt = {}
    for i, t in enumerate(tasks):
        ck = f"{(i + 1) * spt:06d}"
        if ck in by_ckpt and t in by_ckpt[ck]:
            jt[t] = by_ckpt[ck][t]
    print(f"[msemat] {path}: {len(recs)} ckpts, tasks {tasks}, steps_per_task {spt}")
    summ = report(rows, tasks, jt, "mse_matrix (paired-noise MSE, seed 0, slot swap per checkpoint)")
    return {"source": path, "steps_per_task": spt, "rows": recs, "summary": summ}


if __name__ == "__main__":
    cmd = sys.argv[1]
    spt = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    if cmd == "inrun":
        res = cmd_inrun(sys.argv[2], spt)
    elif cmd == "msemat":
        res = cmd_msemat(sys.argv[2], spt)
    else:
        raise SystemExit(__doc__)
    if os.environ.get("OUT"):
        json.dump(res, open(os.environ["OUT"], "w"), indent=1)
        print("wrote", os.environ["OUT"])
