#!/usr/bin/env python3
"""E62 slot autopsy: merged 6x2 (share (4,6)+(8,10) expert, solo 14/16; share
(5,7)+(9,11)+(13,15) VLM) vs interleave-8 + bigsearch-12. Reads:
1. per-task effnum per module (footprint sanity);
2. WITHIN-TABLE SITE-BLEED on the 5 shared pairs (victim site's read mass on
   slots the OTHER member updated, same task) — E61 band was 17-43%;
3. PRIOR-CORE WRITE EVENTS (later tasks' update events into each earlier
   task's core50, per module) — pre-registered 0 at the protection set;
   solo E14/E16 cells are the depth-lever integrity read."""
import json, math, os

NT = int(os.environ.get("SLOTS_NTASKS", "5"))  # 5 or 10 (E63)

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "seq10_merged6x2": f"{BASE}/libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k",
    "merged6x2": f"{BASE}/libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k",
    "interleave": f"{BASE}/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k",
}
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}
PAIRS = {"seq10_merged6x2": [("gemma_expert", 4, 6), ("gemma_expert", 8, 10),
                            ("language_model", 5, 7), ("language_model", 9, 11),
                            ("language_model", 13, 15)],
         "merged6x2": [("gemma_expert", 4, 6), ("gemma_expert", 8, 10),
                       ("language_model", 5, 7), ("language_model", 9, 11),
                       ("language_model", 13, 15)]}
OUT_DIR = os.environ.get("SLOTS_OUT_DIR", "/home/josh/lerobot/outputs/analysis/e62")
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


with open(f"{OUT_DIR}/slots_{os.environ.get('SLOTS_TAG','e62')}.out", "w") as fh:
    for name, rd in RUNS.items():
        try:
            reads, upds = load_run(rd)
        except FileNotFoundError as e:
            print(f"== RUN {name}: SKIPPED ({e}) ==", file=fh)
            continue
        mods = sorted(reads[0], key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
        short = lambda m: ("E" if "gemma_expert" in m else "V") + m.rsplit(".", 1)[-1]
        print(f"\n== RUN {name} ==", file=fh)
        for t in range(NT):
            row = [f"{short(m)}:eff{effnum(reads[t][m]):.0f}" for m in mods]
            print(f" t{t}/e{ENV[t]}: " + "  ".join(row), file=fh)
        if name in PAIRS:
            print("\n-- WITHIN-TABLE SITE-BLEED (victim reads on other-member-updated slots, same task; E61 band 17-43%) --", file=fh)
            for tower, a, b in PAIRS[name]:
                ma = next(m for m in mods if tower in m and m.endswith(f".{a}"))
                mb = next(m for m in mods if tower in m and m.endswith(f".{b}"))
                cells = []
                for t in range(NT):
                    ra, ub = reads[t][ma], set(upds[t][mb])
                    rb, ua = reads[t][mb], set(upds[t][ma])
                    ba = sum(c for s, c in ra.items() if s in ub) / max(sum(ra.values()), 1)
                    bb = sum(c for s, c in rb.items() if s in ua) / max(sum(rb.values()), 1)
                    cells.append(f"t{t}: {short(ma)}<-{short(mb)} {ba*100:.0f}% | {short(mb)}<-{short(ma)} {bb*100:.0f}%")
                print(f" pair ({short(ma)},{short(mb)}): " + "  ".join(cells), file=fh)
        print("\n-- PRIOR-CORE WRITE EVENTS (later tasks' update events into task t's core50; pre-reg 0 at the protection set) --", file=fh)
        for t in range(NT - 1):
            cells = []
            for m in mods:
                c50 = core50(reads[t][m])
                ev = sum(cnt for lt in range(t + 1, NT) for s, cnt in upds[lt][m].items() if s in c50)
                cells.append(f"{short(m)}:{ev}")
            print(f" victim t{t}/e{ENV[t]}: " + "  ".join(cells), file=fh)
print("wrote", OUT_DIR + f"/slots_{os.environ.get('SLOTS_TAG','e62')}.out")
