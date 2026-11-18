#!/usr/bin/env python
"""Expert demonstrability test of every pairing the planner may use (projects/dataset_gen.md, §3).

pair_test.py checks that a pairing's goal can hold (the object teleported into place); this checks that the
scripted expert (expert.py) can achieve it. For each (object, destination, phrase variant) that passed
pair_test.py, and each open/close/turn template with each object its placing step may use: small scenes holding
only the items the task needs, laid out at random as the suite builder lays them out (sample_layout, in the room
the planner would give those items), and the expert run from fresh resets (one episode in three with a widened
start and DART noise, as in the dataset). Writes scripts/libero_tasks/pairing_expert.json, a list of
  {"obj", "dest", "v", "artic", "artic_variant", "room", "ok", "n", "why"}

  python scripts/libero_tasks/pair_expert.py [--workers 6] [--layouts 3] [--episodes 3] [--only drawer,stove]
"""

import argparse
import itertools
import json
import os
import random
import sys
import tempfile
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert as ex  # noqa: E402
import plan_tasks as pt  # noqa: E402
import scene_lib as sl  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairing_expert.json")
ARTIC_ITEMS = {"faucet": ["faucet"], "microwave": ["microwave"], "drawer": ["cabinet"], "stove": ["stove", "frying pan"]}
ARTIC_DEST = {"faucet": None, "microwave": "microwave", "drawer": "drawer", "stove": "pan"}
ARTIC_LANG = {
    "faucet": "turn on the faucet",
    "microwave": "put the {o} in the microwave and close it",
    "drawer": "open the {p} drawer and put the {o} in it",
    "stove": "turn on the stove and put the {o} in the pan",
}
LIVING = ["living room", "coffee table", "tabletop", "floor"]


def one(job):
    i, obj, dest, v, artic, artic_variant, n_layouts, n_eps = job
    res = {"obj": obj, "dest": dest, "v": v, "artic": artic, "artic_variant": artic_variant, "ok": 0, "n": 0,
           "why": Counter(), "lens": []}
    if artic:
        phrase = pt.DESTS[dest][0][0][0] if dest else None
        if dest == "drawer":
            phrase = f"in the {artic_variant} drawer"
        steps = [(obj, dest, phrase)] if dest else []
        language = ARTIC_LANG[artic].format(o=obj, p=artic_variant)
        kinds = set(ARTIC_ITEMS[artic]) | ({obj} if obj else set())
    else:
        phrase, item = pt.DESTS[dest][0][v]
        steps = [(obj, dest, phrase)]
        language = f"lift the {obj}" if dest == "lift" else f"put the {obj} {phrase}"
        kinds = {obj} | ({item} if item else set()) | ({sl.DEST_ITEM[dest]} if dest in sl.DEST_ITEM else set())
    room = pt.room_for(kinds, itertools.cycle(LIVING[i % 4:] + LIVING[: i % 4]),
                       table_goal=any(d in ("beside", "placemat") for _, d, _ in steps))
    room = "kitchen" if room == "mixed" else room
    res["room"] = room
    task = {"steps": steps, "artic": artic, "artic_variant": artic_variant, "language": language}
    opens = set()
    for _, d, _ in steps:
        if d in ("drawer", "microwave", "fridge"):
            opens.add("cabinet" if d == "drawer" else d)
    if artic in ("drawer", "microwave"):
        opens.add("cabinet" if artic == "drawer" else artic)
    reserve = []
    for o, d, ph in steps:
        if d == "beside":
            rel = next(r for r in sl.REL_DIR if ph.startswith(r))
            ref = ph[len(rel):].replace(" the ", "", 1).strip()
            reserve.append((ref, sl.REL_DIR[rel], o))
    suite_task = {"language": language, "steps": [list(s) for s in steps], "artic": artic}
    fd, path = tempfile.mkstemp(suffix=".bddl", prefix="pairx_")
    os.close(fd)
    layouts = 0
    try:
        for k in range(n_layouts * 4):
            if layouts >= n_layouts:
                break
            rng = random.Random(7919 * i + k)
            scene = sl.Scene(k, room, kinds, rng)
            work = {"cabinet" if artic == "drawer" else artic} if artic in ("drawer", "microwave") else set()
            if not sl.sample_layout(scene, kinds, reserve, rng, opens=opens, work=work):
                continue
            Path(path).write_text(sl.write_bddl(scene, task, language))
            try:
                env = sl.make_env(path)
            except Exception as e:  # noqa: BLE001
                res["why"][f"env: {type(e).__name__}"] += 1
                continue
            drift, low, _ = sl.stable_reset(env)
            if env.check_success():
                env.close()
                continue
            layouts += 1
            nrng = np.random.default_rng(1000 * i + k)
            for ep in range(n_eps):
                wide = ep % 3 == 2
                sl.stable_reset(env)
                if wide and not ex.widen_start(env, nrng):
                    sl.stable_reset(env)
                    wide = False
                exp = ex.Expert(env, suite_task, scene.cls, nrng, dart_sigma=0.15 if wide else 0.0)
                try:
                    ok, n = exp.run()
                except Exception as e:  # noqa: BLE001
                    ok, n = False, len(exp.labels)
                    exp.why = f"error {type(e).__name__}: {str(e)[:60]}"
                res["n"] += 1
                res["ok"] += int(ok)
                if ok:
                    res["lens"].append(n)
                else:
                    res["why"][exp.why.split(":")[0][:60]] += 1
            env.close()
    except Exception as e:  # noqa: BLE001
        res["why"][f"error: {type(e).__name__}: {str(e)[:80]}"] += 1
        if os.environ.get("PAIR_DEBUG"):
            traceback.print_exc()
    finally:
        os.remove(path)
    res["why"] = dict(res["why"])
    return res


def jobs_for(only=None):
    jobs = []
    for d, (variants, objs) in pt.DESTS.items():
        if only and d not in only:
            continue
        for o in objs:
            if (o, d) in pt.SEEN:
                continue
            vs = pt.ok_variants(o, d) if pt.OK_VARIANTS is not None else range(len(variants))
            if pt.OK_VARIANTS is not None and (o, d) not in pt.OK_VARIANTS:
                continue
            if d == "beside":
                vs = vs[:1]
            for v in vs:
                jobs.append((o, d, v, None, None))
    for art, dest in ARTIC_DEST.items():
        if only and art not in only:
            continue
        if dest is None:
            jobs.append((None, None, None, art, None))
            continue
        for o in sorted({o for o, d in pt.CANDS if d == dest} | set(pt.DESTS[dest][1])):
            if (o, dest) not in pt.OK_VARIANTS:
                continue
            jobs.append((o, dest, None, art, "top" if art == "drawer" else None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--layouts", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated destination kinds or articulations")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    jobs = [(i, *j, a.layouts, a.episodes) for i, j in enumerate(jobs_for(only))]
    print(f"{len(jobs)} pairings, {a.layouts} layouts x {a.episodes} episodes each", flush=True)
    with Pool(a.workers) as pool:
        results = []
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            results.append(r)
            tag = f"{r['artic'] + ': ' if r['artic'] else ''}{r['obj']} -> {r['dest']} v{r['v']}"
            print(f"  {r['ok']}/{r['n']}  {tag}  [{r['room']}]  {r['why'] or ''}", flush=True)
    old = json.loads(Path(OUT).read_text()) if (only and os.path.exists(OUT)) else []
    keep = [r for r in old if not ((r["artic"] or r["dest"]) in only)] if only else []
    Path(OUT).write_text(json.dumps(keep + results, indent=1))
    rate = [r["ok"] / max(1, r["n"]) for r in results]
    print(f"done: {len(results)} pairings; >= 2/3 success: {sum(x >= 0.66 for x in rate)}; "
          f"0 success: {sum(x == 0 for x in rate)}")


if __name__ == "__main__":
    main()
