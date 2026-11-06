#!/usr/bin/env python3
"""REAL-WORLD duplicate of e62_slots.py (slot autopsy for the merged 6x2 layout).

Reads one sequential run's memory_by_task/memory_usage_task_{t}.json and reports:
1. per-task effnum per module (footprint sanity);
2. WITHIN-TABLE SITE-BLEED on the shared pairs (victim site's read mass on slots the OTHER
   member updated, same task) — sim bands: E61 17-43%, E62 14-51%;
3. PRIOR-CORE WRITE EVENTS (later tasks' update events into each earlier task's core50, per
   module) — sim pre-registration 0 at the protection set; solo E14/E16 = depth-lever integrity.
IDENTICAL numerics to e62_slots.py. Deltas: one run from SLOTS_RUN_DIR (no hardcoded RUNS
table), task labels from SLOTS_LABELS (seq id -> pool id; there is no LIBERO env map), shared
pairs from SLOTS_PAIRS, and a JSON twin of the .out for the research log.

Env: SLOTS_RUN_DIR (required), SLOTS_RUN_NAME (basename), SLOTS_NTASKS (5),
SLOTS_LABELS ("0:p0,1:p10,2:p16,3:p7,4:p1" = split v5 seq order), SLOTS_PAIRS
("E4-6,E8-10,V5-7,V9-11,V13-15"), SLOTS_OUT_DIR, SLOTS_TAG.
"""
import json
import math
import os

NT = int(os.environ.get("SLOTS_NTASKS", "5"))
RUN_DIR = os.environ["SLOTS_RUN_DIR"].rstrip("/")
NAME = os.environ.get("SLOTS_RUN_NAME", os.path.basename(RUN_DIR))
LABELS = {int(k): v for k, v in (kv.split(":") for kv in
          os.environ.get("SLOTS_LABELS", "0:p0,1:p10,2:p16,3:p7,4:p1").split(","))}
TOWER = {"E": "gemma_expert", "V": "language_model"}
PAIRS = []
for tok in os.environ.get("SLOTS_PAIRS", "E4-6,E8-10,V5-7,V9-11,V13-15").split(","):
    tok = tok.strip()
    if tok:
        a, b = tok[1:].split("-")
        PAIRS.append((TOWER[tok[0]], int(a), int(b)))
OUT_DIR = os.environ.get("SLOTS_OUT_DIR", "/home/josh/lerobot/outputs/analysis/realworld/e65")
TAG = os.environ.get("SLOTS_TAG", "rw_merged6x2")
os.makedirs(OUT_DIR, exist_ok=True)


def load_run(run_dir):
    reads, upds = {}, {}
    for t in range(NT):
        d = json.load(open(f"{run_dir}/memory_by_task/memory_usage_task_{t}.json"))["per_module"]
        reads[t], upds[t] = {}, {}
        for mod, tasks in d.items():
            key = f"task_{t}"
            if key not in tasks:
                continue
            r, u = {}, {}
            for slot, st in tasks[key].items():
                sid = int(slot.rsplit("_", 1)[-1])
                if st.get("total_accesses", 0):
                    r[sid] = st["total_accesses"]
                if st.get("total_updates", 0):
                    u[sid] = st["total_updates"]
            reads[t][mod] = r
            upds[t][mod] = u
    return reads, upds


def effnum(read):
    tot = sum(read.values())
    if not tot:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in read.values() if c)
    return math.exp(h)


def core50(read):
    tot = sum(read.values())
    if not tot:
        return set()
    acc, out = 0.0, set()
    for sid, c in sorted(read.items(), key=lambda kv: -kv[1]):
        out.add(sid)
        acc += c
        if acc >= 0.5 * tot:
            break
    return out


def lab(t):
    return f"t{t}/{LABELS.get(t, '?')}"


reads, upds = load_run(RUN_DIR)
mods = sorted(reads[0], key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
short = lambda m: ("E" if "gemma_expert" in m else "V") + m.rsplit(".", 1)[-1]  # noqa: E731
summary = {"run": NAME, "run_dir": RUN_DIR, "n_tasks": NT,
           "labels": {str(k): v for k, v in LABELS.items()}, "modules": [short(m) for m in mods],
           "effnum": {}, "site_bleed": {}, "prior_core_events": {}}
out_path = f"{OUT_DIR}/slots_{TAG}.out"
with open(out_path, "w") as fh:
    print(f"\n== RUN {NAME} ==", file=fh)
    for t in range(NT):
        row = [f"{short(m)}:eff{effnum(reads[t][m]):.0f}" for m in mods]
        print(f" {lab(t)}: " + "  ".join(row), file=fh)
        summary["effnum"][str(t)] = {short(m): effnum(reads[t][m]) for m in mods}
    print("\n-- WITHIN-TABLE SITE-BLEED (victim reads on other-member-updated slots, same task; "
          "sim bands E61 17-43% / E62 14-51%) --", file=fh)
    for tower, a, b in PAIRS:
        ma = next((m for m in mods if tower in m and m.endswith(f".{a}")), None)
        mb = next((m for m in mods if tower in m and m.endswith(f".{b}")), None)
        if ma is None or mb is None:
            print(f" pair ({tower} {a},{b}): not present in this run - skipped", file=fh)
            continue
        cells, sb = [], {}
        for t in range(NT):
            ra, ub = reads[t][ma], set(upds[t][mb])
            rb, ua = reads[t][mb], set(upds[t][ma])
            ba = sum(c for s, c in ra.items() if s in ub) / max(sum(ra.values()), 1)
            bb = sum(c for s, c in rb.items() if s in ua) / max(sum(rb.values()), 1)
            cells.append(f"t{t}: {short(ma)}<-{short(mb)} {ba*100:.0f}% | {short(mb)}<-{short(ma)} {bb*100:.0f}%")
            sb[str(t)] = {f"{short(ma)}<-{short(mb)}": ba, f"{short(mb)}<-{short(ma)}": bb}
        print(f" pair ({short(ma)},{short(mb)}): " + "  ".join(cells), file=fh)
        summary["site_bleed"][f"{short(ma)},{short(mb)}"] = sb
    print("\n-- PRIOR-CORE WRITE EVENTS (later tasks' update events into task t's core50; "
          "sim pre-reg 0 at the protection set) --", file=fh)
    for t in range(NT - 1):
        cells, ev_d = [], {}
        for m in mods:
            c50 = core50(reads[t][m])
            ev = sum(cnt for lt in range(t + 1, NT) for s, cnt in upds[lt][m].items() if s in c50)
            cells.append(f"{short(m)}:{ev}")
            ev_d[short(m)] = ev
        print(f" victim {lab(t)}: " + "  ".join(cells), file=fh)
        summary["prior_core_events"][str(t)] = ev_d
json_path = f"{OUT_DIR}/slots_{TAG}.json"
json.dump(summary, open(json_path, "w"), indent=1)
print(open(out_path).read())
print("wrote", out_path, "and", json_path)
