#!/usr/bin/env python3
"""E59 slot autopsy: interleave (E[6,8,10,12]+V[7,9,11,13]) vs B (E[2,4,6,8]+V[10,12,14,16]).
Per module x task: read mass (total_accesses), update events (total_updates),
core50 set/size, effnum, self-coverage; per victim<writer pair: RTO (victim read
mass on writer-updated slots), events into victim core50. Output: text + json."""
import json, math, os, sys
from collections import defaultdict

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "bigsearch":  f"{BASE}/libero_10_seq5_jw_bigsearch_e4to16_v5to13_prepass_beta4corefrac_topt3072_lr2x_steps5k",
    "interleave": f"{BASE}/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k",
    "B":          f"{BASE}/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k",
}
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
OUT_DIR = "/home/josh/lerobot/outputs/analysis/e60"
os.makedirs(OUT_DIR, exist_ok=True)


def short_mod(m):
    l = m.rsplit(".", 1)[-1]
    if "gemma_expert" in m:
        return f"E{l}"
    return f"V{l}"


def load_run(run_dir):
    """returns reads[t][mod] = {slot:int}, upds[t][mod] = {slot:int}"""
    reads, upds = {}, {}
    for t in range(5):
        p = os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")
        d = json.load(open(p))["per_module"]
        reads[t], upds[t] = {}, {}
        for mod, tasks in d.items():
            key = f"task_{t}"
            if key not in tasks:
                continue
            r, u = {}, {}
            for slot, st in tasks[key].items():
                sid = int(slot.rsplit("_", 1)[-1])
                a = st.get("total_accesses", 0)
                up = st.get("total_updates", 0)
                if a:
                    r[sid] = a
                if up:
                    u[sid] = up
            reads[t][mod] = r
            upds[t][mod] = u
    return reads, upds


def core50(read):
    tot = sum(read.values())
    s, acc, out = sorted(read.items(), key=lambda kv: -kv[1]), 0, set()
    for sid, c in s:
        out.add(sid); acc += c
        if acc >= 0.5 * tot:
            break
    return out


def effnum(read):
    tot = sum(read.values())
    if not tot:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in read.values() if c)
    return math.exp(h)


def analyze(name, run_dir, fh):
    reads, upds = load_run(run_dir)
    mods = sorted(reads[0].keys(), key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
    res = {"mods": [short_mod(m) for m in mods], "tasks": {}}
    print(f"\n{'='*100}\nRUN {name}\n{'='*100}", file=fh)
    cores = {t: {m: core50(reads[t][m]) for m in mods} for t in range(5)}

    # per-task footprint + self-coverage
    print(f"\n-- footprints (core50 size / effnum) + self-coverage + n_updated --", file=fh)
    for t in range(5):
        row = []
        tr = res["tasks"].setdefault(t, {})
        for m in mods:
            r, u = reads[t][m], upds[t][m]
            tot = sum(r.values())
            selfcov = sum(c for s, c in r.items() if s in u) / max(tot, 1)
            tr[short_mod(m)] = {
                "core50": len(cores[t][m]), "effnum": round(effnum(r), 1),
                "selfcov": round(selfcov, 3), "n_upd": len(u),
                "upd_events": sum(u.values()),
            }
            row.append(f"{short_mod(m)}:{len(cores[t][m])}/{effnum(r):.0f} sc={selfcov:.2f} nU={len(u)}")
        print(f" t{t}/e{ENV[t]}: " + "  ".join(row), file=fh)

    # victim<-writer: RTO + events into victim core
    print(f"\n-- victim<-writer bleed (per module: victim read-mass %% on writer-updated slots | writer events into victim core50) --", file=fh)
    res["pairs"] = {}
    for v in range(5):
        for w in range(v + 1, 5):
            cells = []
            pr = res["pairs"].setdefault(f"t{v}<-t{w}", {})
            for m in mods:
                r = reads[v][m]
                tot = sum(r.values())
                wu = upds[w][m]
                bleed = sum(c for s, c in r.items() if s in wu) / max(tot, 1)
                ev_core = sum(e for s, e in wu.items() if s in cores[v][m])
                pr[short_mod(m)] = {"bleed": round(bleed, 4), "ev_into_core": ev_core}
                cells.append(f"{short_mod(m)}:{bleed*100:.1f}%|{ev_core/1000:.0f}k")
            print(f" t{v}/e{ENV[v]} <- t{w}/e{ENV[w]}: " + "  ".join(cells), file=fh)

    # cumulative RTO per victim (union of all later writers)
    print(f"\n-- cumulative RTO (read mass on ANY later writer's updates) --", file=fh)
    for v in range(5):
        if v == 4:
            continue
        cells = []
        for m in mods:
            r = reads[v][m]
            tot = sum(r.values())
            later = set()
            for w in range(v + 1, 5):
                later |= set(upds[w][m])
            rto = sum(c for s, c in r.items() if s in later) / max(tot, 1)
            # core-restricted: victim core mass on later-updated slots
            cmass = sum(r[s] for s in cores[v][m])
            core_rto = sum(r[s] for s in cores[v][m] if s in later) / max(cmass, 1)
            cells.append(f"{short_mod(m)}:{rto*100:.0f}%/core{core_rto*100:.0f}%")
            res["tasks"][v].setdefault("rto", {})[short_mod(m)] = round(rto, 4)
            res["tasks"][v].setdefault("core_rto", {})[short_mod(m)] = round(core_rto, 4)
        print(f" t{v}/e{ENV[v]}: " + "  ".join(cells), file=fh)
    return res


all_res = {}
with open(os.path.join(OUT_DIR, "slots_e60.out"), "w") as fh:
    for name, d in RUNS.items():
        all_res[name] = analyze(name, d, fh)
json.dump(all_res, open(os.path.join(OUT_DIR, "slots_e60.json"), "w"))
print("done; wrote", OUT_DIR + "/slots_e60.{out,json}")
