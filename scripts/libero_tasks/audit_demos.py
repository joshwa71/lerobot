#!/usr/bin/env python
"""Mode audit of libero_ext expert demos (projects/dataset_gen.md §3d): for each task's npz, how many of its demos
share the most common hand-orientation signature (generate_demos.yaw_signature: the wrist yaw at each change of
hand orientation between gripper events), and the signatures present. A task below ~0.9 has demos that differ in a
discrete choice, which the policy cannot learn to make (task 11 of the v4 panel: 36 / 14, 12% success).

  python scripts/libero_tasks/audit_demos.py outputs/libero_ext_demos [--min-share 0.9]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_demos import dominant, same_mode, yaw_signature  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demos")
    ap.add_argument("--min-share", type=float, default=0.9)
    a = ap.parse_args()
    flagged = []
    for f in sorted(Path(a.demos).glob("*.npz")):
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            sigs = [yaw_signature(z[f"state_{k}"], z[f"action_{k}"]) for k in range(meta["n_demos"])]
        best = dominant(sigs)
        share = len(best) / max(1, len(sigs))
        # the other modes: group the rest greedily
        rest, modes = [j for j in range(len(sigs)) if j not in set(best)], []
        while rest:
            grp = [j for j in rest if same_mode(sigs[rest[0]], sigs[j])]
            modes.append((len(grp), [round(y) for y in sigs[rest[0]]]))
            rest = [j for j in rest if j not in set(grp)]
        flag = share < a.min_share
        if flag:
            flagged.append(meta["task_index"])
        others = "  others: " + ", ".join(f"{n}x{s}" for n, s in modes) if modes else ""
        print(
            f"{'!!' if flag else '  '} {meta['task_index']:2d} {share:4.2f} {len(best)}x{[round(y) for y in sigs[best[0]]]}"
            f"  {meta['language'][:60]}{others}"
        )
    print(f"flagged (share < {a.min_share}): {flagged}")


if __name__ == "__main__":
    main()
