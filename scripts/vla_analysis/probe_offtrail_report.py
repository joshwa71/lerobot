#!/usr/bin/env python3
"""Off-trail probe, stage 3 (E56): join the two models' paired chunks, the retrieval
traces, the written-slot sets, and the excursion distances into the pre-registered reads.
Pure CPU — no GPU, no model.

Reads (per E56 discussion):
  1. Degradation curves: cross-model chunk disagreement D(s) vs excursion distance, by
     population (demo control / success / fail, per harvest tag).
  2. Retrieval composition: fraction of retrieval mass on slots the task's sequential
     block actually WROTE (self-written mass) vs distance, success vs fail. The
     fallback-to-A-content hypothesis predicts collapse tracking read 1.
  3. Churn vs drift: per-episode consecutive-call weighted-Jaccard of retrieved sets.
  4. Divergence-point table: per failed episode, first call where D(s) exceeds the
     success-population P90; composition/churn around it.
Anchor: per-model chunk-vs-demo-gt MSE on demo-control rows must reproduce the known
own-chunk numbers (B ~0.032, spec ~0.033) — instrument zero-point.

Env knobs:
  REPORT_DIR       dir with chunks_{TAG}.npz / features_{TAG}.npz (from stage 2)
  REPORT_TAG_A     model A tag (the memory model whose traces we analyse, e.g. B)
  REPORT_TAG_B     model B tag (reference oracle, e.g. spec_e7)
  REPORT_HARVESTS  comma list of harvest dirs (for traces + index.json)
  REPORT_MBT       memory_usage_task_{T}.json of model A's sequential run (written sets)
  REPORT_TASKKEY   task key inside the MBT json (default task_4)
  REPORT_OUT       output jsonl path (rows: one per state)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_written_sets(mbt_path, task_key):
    per_module = json.load(open(mbt_path))["per_module"]
    written = {}
    for mod, tasks in per_module.items():
        if task_key not in tasks:
            continue
        w = {int(k.split("_")[-1]) for k, st in tasks[task_key].items()
             if st.get("total_updates", 0) > 0}
        written[mod] = np.array(sorted(w), dtype=np.int64)
    return written


def _tower(mod):
    return "vlm" if "language_model" in mod else "expert"


def _match_written(written, trace_key):
    """Trace keys and MBT keys are both '<path>.layers.N' — match by suffix to be safe."""
    for mod, w in written.items():
        if trace_key.endswith(mod) or mod.endswith(trace_key):
            return w
    return None


def _wjac(s1, m1, s2, m2):
    """Weighted Jaccard between two (slots, mass) sparse vectors."""
    d1 = dict(zip(s1.tolist(), m1.tolist()))
    d2 = dict(zip(s2.tolist(), m2.tolist()))
    mn = mx = 0.0
    for k in set(d1) | set(d2):
        a, b = d1.get(k, 0.0), d2.get(k, 0.0)
        mn += min(a, b)
        mx += max(a, b)
    return mn / mx if mx > 0 else 0.0


def _quart_table(rows, xkey, ykey, title, fh, pops=None):
    xs = np.array([r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None])
    if xs.size < 8:
        return
    qs = np.quantile(xs, [0.25, 0.5, 0.75])
    print(f"\n-- {title} (x={xkey} quartiles: {qs.round(4).tolist()}) --", file=fh)
    groups = sorted(set(r["pop"] for r in rows)) if pops is None else pops
    print(f"{'pop':<22}" + "".join(f"Q{i + 1:<9}" for i in range(4)) + " n", file=fh)
    for g in groups:
        sub = [r for r in rows if r["pop"] == g and r.get(xkey) is not None and r.get(ykey) is not None]
        if not sub:
            continue
        cells = []
        for qi in range(4):
            lo = -np.inf if qi == 0 else qs[qi - 1]
            hi = np.inf if qi == 3 else qs[qi]
            ys = [r[ykey] for r in sub if lo < r[xkey] <= hi] if qi else \
                 [r[ykey] for r in sub if r[xkey] <= hi]
            cells.append(f"{np.mean(ys):.4f}" if ys else "-")
        print(f"{g:<22}" + "".join(f"{c:<10}" for c in cells) + f" {len(sub)}", file=fh)


def main():
    rdir = Path(os.environ["REPORT_DIR"])
    tag_a = os.environ.get("REPORT_TAG_A", "B")
    tag_b = os.environ.get("REPORT_TAG_B", "spec_e7")
    harvests = [h for h in os.environ["REPORT_HARVESTS"].split(",") if h]
    mbt = os.environ["REPORT_MBT"]
    task_key = os.environ.get("REPORT_TASKKEY", "task_4")
    out_path = os.environ["REPORT_OUT"]
    fh = open(out_path.replace(".jsonl", ".txt"), "w")

    A = np.load(rdir / f"chunks_{tag_a}.npz", allow_pickle=True)
    B = np.load(rdir / f"chunks_{tag_b}.npz", allow_pickle=True)
    uids = A["uids"].tolist()
    assert uids == B["uids"].tolist(), "state lists differ between models — rerun scoring"
    ca, cb = A["chunks"].astype(np.float32), B["chunks"].astype(np.float32)

    # ---------- anchor: per-model demo-gt error ----------
    demo_rows = [i for i, u in enumerate(uids) if u.startswith("demo:")]
    gt = A["demo_gt"].astype(np.float32)
    for tag, c in ((tag_a, ca), (tag_b, cb)):
        errs = [((c[i][k] - gt[j]) ** 2).mean() for j, i in enumerate(demo_rows)
                for k in range(c.shape[1])]
        print(f"[anchor] {tag} demo chunk-vs-gt MSE = {np.mean(errs):.4f} "
              f"(expected ~ the known own-chunk numbers)", file=fh)

    # ---------- distances ----------
    feats = None
    fpath = rdir / f"features_{tag_a}.npz"
    if fpath.exists():
        F = np.load(fpath, allow_pickle=True)
        assert F["uids"].tolist() == uids
        feats = F["feats"].astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    ref_rows = demo_rows[0::2]  # even demo rows = kNN reference
    states = A["states"].astype(np.float32)
    demo_states = states[demo_rows]
    st_std = demo_states.std(axis=0) + 1e-6

    # ---------- traces (model A only) ----------
    trace = {}  # uid -> {modkey: (slots, mass)}
    succ_by_ep = {}
    for hdir in harvests:
        hdir = Path(hdir)
        idx = json.load(open(hdir / "index.json"))
        for m in idx["episodes"]:
            succ_by_ep[f"{hdir.name}:ep{m['ep']:03d}"] = bool(m["success"])
        for m in idx["episodes"]:
            tp = hdir / f"trace_ep{m['ep']:03d}.npz"
            if not tp.exists():
                continue
            d = np.load(tp)
            per_call = defaultdict(dict)
            for k in d.files:
                c, mod, kind = k.split("__")
                per_call[c].setdefault(mod, [None, None])
                per_call[c][mod][0 if kind == "slots" else 1] = d[k]
            for c, mods in per_call.items():
                uid = f"{hdir.name}:ep{m['ep']:03d}:{c}"
                trace[uid] = {mod: (v[0], v[1]) for mod, v in mods.items()}

    written = _load_written_sets(mbt, task_key)
    print(f"[report] written sets: " +
          ", ".join(f"{m.split('.')[-1]}:{len(w)}" for m, w in written.items()), file=fh)

    # ---------- per-state rows ----------
    rows = []
    prev_by_ep = {}
    for i, uid in enumerate(uids):
        if uid.startswith("demo:"):
            pop, ep_key, call = "demo", None, None
        else:
            htag, ep, c = uid.split(":")
            ep_key = f"{htag}:{ep}"
            call = int(c[1:])
            pop = f"{htag}/{'succ' if succ_by_ep.get(ep_key) else 'fail'}"
        D = float(((ca[i] - cb[i]) ** 2).mean())
        d_feat = None
        if feats is not None:
            sims = feats[i] @ feats[ref_rows].T
            if uid.startswith("demo:") and i in ref_rows:
                sims[ref_rows.index(i)] = -np.inf  # leave-one-out for reference rows
            d_feat = float(1.0 - sims.max())
        d_state = float(np.min(np.linalg.norm((states[i] - demo_states) / st_std, axis=1)))
        row = {"uid": uid, "pop": pop, "call": call, "D": D,
               "d_feat": d_feat, "d_state": d_state}
        tr = trace.get(uid)
        if tr:
            sm, churn = {}, {}
            for mod, (slots, mass) in tr.items():
                w = _match_written(written, mod)
                tot = float(mass.sum())
                if w is not None and tot > 0:
                    sm.setdefault(_tower(mod), []).append(
                        float(mass[np.isin(slots, w)].sum()) / tot)
                pv = prev_by_ep.get((ep_key, mod))
                if pv is not None:
                    churn.setdefault(_tower(mod), []).append(
                        1.0 - _wjac(slots, mass, pv[0], pv[1]))
                prev_by_ep[(ep_key, mod)] = (slots, mass)
            for t in ("expert", "vlm"):
                if t in sm:
                    row[f"selfmass_{t}"] = float(np.mean(sm[t]))
                if t in churn:
                    row[f"churn_{t}"] = float(np.mean(churn[t]))
        rows.append(row)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---------- reads ----------
    _quart_table(rows, "d_feat", "D", "READ 1: cross-model disagreement D vs feature distance", fh)
    _quart_table(rows, "d_state", "D", "READ 1b: D vs proprio distance", fh)
    _quart_table(rows, "d_feat", "selfmass_expert", "READ 2: expert self-written mass vs distance", fh)
    _quart_table(rows, "d_feat", "selfmass_vlm", "READ 2b: vlm self-written mass vs distance", fh)
    _quart_table(rows, "d_feat", "churn_expert", "READ 3: expert churn vs distance", fh)

    # per-call trajectories
    print("\n-- per-call means (D | selfmass_expert | selfmass_vlm | churn_expert) --", file=fh)
    for pop in sorted(set(r["pop"] for r in rows if r["pop"] != "demo")):
        sub = [r for r in rows if r["pop"] == pop]
        line = [pop]
        for c in range(0, 11):
            cc = [r for r in sub if r["call"] == c]
            if not cc:
                continue
            line.append(f"c{c}:{np.mean([r['D'] for r in cc]):.3f}/"
                        f"{np.mean([r.get('selfmass_expert', np.nan) for r in cc]):.2f}/"
                        f"{np.mean([r.get('selfmass_vlm', np.nan) for r in cc]):.2f}/"
                        f"{np.mean([r.get('churn_expert', np.nan) for r in cc]):.2f}")
        print("  ".join(line), file=fh)

    # divergence points: first call with D above success-P90, per failed episode of model A's harvest
    a_pops = [p for p in set(r["pop"] for r in rows) if p != "demo"]
    for htag in sorted(set(p.split("/")[0] for p in a_pops)):
        succ_D = [r["D"] for r in rows if r["pop"] == f"{htag}/succ"]
        if not succ_D:
            continue
        thr = float(np.quantile(succ_D, 0.9))
        print(f"\n-- READ 4: divergence points ({htag}, thr=succ-P90={thr:.4f}) --", file=fh)
        eps = sorted(set(r["uid"].rsplit(":", 1)[0] for r in rows if r["pop"] == f"{htag}/fail"))
        for ek in eps:
            er = sorted((r for r in rows if r["uid"].startswith(ek + ":")), key=lambda r: r["call"])
            first = next((r for r in er if r["D"] > thr), None)
            if first:
                print(f"{ek}: first D>{thr:.3f} at call {first['call']} "
                      f"(D={first['D']:.3f}, selfmass_e={first.get('selfmass_expert', float('nan')):.2f}, "
                      f"churn_e={first.get('churn_expert', float('nan')):.2f}, d_feat={first.get('d_feat')})",
                      file=fh)
            else:
                print(f"{ek}: never crosses (max D={max(r['D'] for r in er):.3f})", file=fh)

    fh.close()
    print(open(out_path.replace(".jsonl", ".txt")).read())
    print(f"[report] rows -> {out_path}")


if __name__ == "__main__":
    main()
