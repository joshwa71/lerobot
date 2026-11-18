#!/usr/bin/env python
"""Physical test of every object-destination pairing the planner may use (projects/dataset_gen.md).

For each (object, destination, phrase variant): a two-item kitchen scene with the destination item and the object,
the goal false after a settled reset, then the goal forced (joints set, object teleported into or onto the
destination), settled, and the success check read. Fridge and bookcase levels are tried in turn; the first
level that works for an object is kept. Writes scripts/libero_tasks/pairing_ok.json:
  {dest: {object: {"variants": [ok phrase indices], "level": level or null}}}

  python scripts/libero_tasks/pair_test.py [--workers 8]
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import plan_tasks as pt  # noqa: E402
import scene_lib as sl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "pairing_ok.json")
LEVELS = {
    "fridge": ["middle_region", "lower_region", "upper_region"],
    "bookcase": ["middle_region", "bottom_region", "top_region"],
}


def one(job):
    obj, dest, v, level = job
    phrase = pt.DESTS[dest][0][v][0]
    item = sl.DEST_ITEM.get(dest) or (pt.DESTS[dest][0][v][1] if dest == "beside" else None)
    kinds = {obj} | ({item} if item else set())
    sc = sl.Scene(0, "kitchen", kinds, random.Random(0))
    # fixed layout: the destination item on the left side (front-facing fixtures face the centre), the object right
    if item:
        if sl.ITEMS[item][3]:
            sc.pos[item] = (0.0, 0.30, 0.0)
        else:
            sc.pos[item] = (-0.05, 0.12, math.pi if item == "caddy" else 0.0)
    sc.pos[obj] = (0.10, -0.18, 0.0)
    if dest == "fridge":
        sl.DEST_GOAL["fridge"] = ("In", "site", level)
    if dest == "bookcase":
        sl.DEST_GOAL["bookcase"] = ("In", "site", level)
    task = {"steps": [(obj, dest, phrase)], "artic": None}
    res = {"obj": obj, "dest": dest, "v": v, "level": level, "ok": False, "why": ""}
    fd, path = tempfile.mkstemp(suffix=".bddl", prefix="pair_")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(sl.write_bddl(sc, task, f"put the {obj} {phrase}"))
        env = sl.make_env(path)
        drift, low, _ = sl.stable_reset(env)
        if env.check_success():
            res["why"] = "goal true at start"
        else:
            ok = sl.force_goal(env)
            res["ok"] = bool(ok)
            res["why"] = "" if ok else "forced goal not reached"
        env.close()
    except Exception as ex:  # noqa: BLE001
        res["why"] = f"error: {type(ex).__name__}: {str(ex)[:120]}"
        if os.environ.get("PAIR_DEBUG"):
            traceback.print_exc()
    finally:
        os.remove(path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated destination kinds")
    a = ap.parse_args()
    jobs = []
    dests = a.only.split(",") if a.only else list(pt.DESTS)
    for dest in dests:
        objs = pt.DESTS[dest][1]
        nv = len(pt.DESTS[dest][0])
        for obj in objs:
            if (obj, dest) in pt.SEEN:
                continue
            # beside is a table region: one reference and relation per object is enough
            vs = [0] if dest == "beside" else range(nv)
            for v in vs:
                for level in LEVELS.get(dest, [None]):
                    jobs.append((obj, dest, v, level))
    print(f"{len(jobs)} pairing tests")
    with Pool(a.workers) as pool:
        results = pool.map(one, jobs, chunksize=1)
    ok = json.loads(Path(OUT).read_text()) if (a.only and os.path.exists(OUT)) else {}
    for d in dests:
        ok.pop(d, None)
    for r in results:
        d = ok.setdefault(r["dest"], {})
        e = d.setdefault(r["obj"], {"variants": [], "level": None, "fail": []})
        if r["ok"]:
            if r["level"] is None or e["level"] in (None, r["level"]):
                if r["v"] not in e["variants"]:
                    e["variants"].append(r["v"])
                e["level"] = r["level"]
        else:
            e["fail"].append(f"v{r['v']}{'/' + r['level'] if r['level'] else ''}: {r['why']}")
    Path(OUT).write_text(json.dumps(ok, indent=1))
    n_ok = sum(1 for d in ok.values() for e in d.values() if e["variants"])
    n_all = sum(len(d) for d in ok.values())
    print(f"pairs with at least one working variant: {n_ok} of {n_all}")
    for dest, d in ok.items():
        bad = [o for o, e in d.items() if not e["variants"]]
        if bad:
            print(f"  {dest}: fails for " + ", ".join(f"{o} ({d[o]['fail'][0]})" for o in bad))


if __name__ == "__main__":
    main()
