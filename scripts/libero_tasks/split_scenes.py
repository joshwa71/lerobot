#!/usr/bin/env python
"""Split crowded scenes of plan.json in two (projects/dataset_gen.md): the tasks are unchanged, a scene that
cannot be laid out in the reachable area gives part of its tasks to a new scene in the same room.

The split chosen keeps every scene's tasks together with as few items as possible: it minimises the larger
half's (2 x large items + small items), preferring halves of 2 tasks.

  python scripts/libero_tasks/split_scenes.py 3,10,14,20
"""

import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_lib as sl  # noqa: E402

PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")


def task_items(t):
    items = set()
    for o, d, _ in t["steps"]:
        items.add(o)
        if d in sl.DEST_ITEM:
            items.add(sl.DEST_ITEM[d])
        if d == "beside":
            rel = next(r for r in sl.REL_DIR if _.startswith(r))
            items.add(_[len(rel) :].replace(" the ", "", 1).strip())
    art = {
        "faucet": ["faucet"],
        "fridge": ["fridge"],
        "microwave": ["microwave"],
        "drawer": ["cabinet"],
        "stove": ["stove", "frying pan"],
    }.get(t.get("artic"), [])
    return items | set(art)


def load(items):
    large = sum(1 for k in items if sl.ITEMS[k][1] or k in ("basket", "tray"))
    return 2 * large + (len(items) - large)


def main():
    ids = [int(x) for x in sys.argv[1].split(",")]
    plan = json.loads(Path(PLAN).read_text())
    nxt = max(s["id"] for s in plan["scenes"]) + 1
    for sid in ids:
        sc = next(s for s in plan["scenes"] if s["id"] == sid)
        ts = sc["tasks"]
        best = None
        for k in range(1, len(ts) // 2 + 1):
            for a in itertools.combinations(range(len(ts)), k):
                part_a = [ts[i] for i in a]
                part_b = [ts[i] for i in range(len(ts)) if i not in a]
                ia, ib = set().union(*map(task_items, part_a)), set().union(*map(task_items, part_b))
                score = (max(load(ia), load(ib)), abs(len(part_a) - len(part_b)))
                if best is None or score < best[0]:
                    best = (score, part_a, part_b, ia, ib)
        _, part_a, part_b, ia, ib = best
        sc["tasks"], sc["items"] = part_a, sorted(ia)
        plan["scenes"].append(
            {"id": nxt, "room": sc["room"], "items": sorted(ib), "tasks": part_b, "split_from": sid}
        )
        print(
            f"S{sid:02d} -> S{sid:02d} ({len(part_a)} tasks, {sorted(ia)}) + S{nxt:02d} ({len(part_b)} tasks, {sorted(ib)})"
        )
        nxt += 1
    Path(PLAN).write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
