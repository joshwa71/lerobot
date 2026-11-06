#!/usr/bin/env python3
"""E61 slot autopsy: sharepairs vs interleave + the WITHIN-TABLE SITE-BLEED read
(pre-registered <= ~15%): for each shared pair, victim site A's read mass landing
on slots the OTHER member updated (same task), per task per pair."""
import json, math, os

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "sharepairs": f"{BASE}/libero_10_seq5_jw_sharepairs_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k",
    "interleave": f"{BASE}/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k",
}
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
PAIRS = {"sharepairs": [("gemma_expert", 6, 8), ("gemma_expert", 10, 12), ("language_model", 7, 9), ("language_model", 11, 13)]}
OUT_DIR = "/home/josh/lerobot/outputs/analysis/e61"
os.makedirs(OUT_DIR, exist_ok=True)


def load_run(run_dir):
    reads, upds = {}, {}
    for t in range(5):
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


with open(f"{OUT_DIR}/slots_e61.out", "w") as fh:
    for name, rd in RUNS.items():
        reads, upds = load_run(rd)
        mods = sorted(reads[0], key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
        short = lambda m: ("E" if "gemma_expert" in m else "V") + m.rsplit(".", 1)[-1]
        print(f"\n== RUN {name} ==", file=fh)
        for t in range(5):
            row = [f"{short(m)}:eff{effnum(reads[t][m]):.0f}" for m in mods]
            print(f" t{t}/e{ENV[t]}: " + "  ".join(row), file=fh)
        if name in PAIRS:
            print("\n-- WITHIN-TABLE SITE-BLEED (victim reads on other-member-updated slots, same task) --", file=fh)
            for tower, a, b in PAIRS[name]:
                ma = next(m for m in mods if tower in m and m.endswith(f".{a}"))
                mb = next(m for m in mods if tower in m and m.endswith(f".{b}"))
                cells = []
                for t in range(5):
                    ra, ub = reads[t][ma], set(upds[t][mb])
                    rb, ua = reads[t][mb], set(upds[t][ma])
                    ba = sum(c for s, c in ra.items() if s in ub) / max(sum(ra.values()), 1)
                    bb = sum(c for s, c in rb.items() if s in ua) / max(sum(rb.values()), 1)
                    cells.append(f"t{t}: {short(ma)}<-{short(mb)} {ba*100:.0f}% | {short(mb)}<-{short(ma)} {bb*100:.0f}%")
                print(f" pair ({short(ma)},{short(mb)}): " + "  ".join(cells), file=fh)
print("wrote", OUT_DIR + "/slots_e61.out")
