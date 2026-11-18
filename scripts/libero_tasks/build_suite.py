#!/usr/bin/env python
"""Build the libero_ext suite from scripts/libero_tasks/plan.json (projects/dataset_gen.md).

Per scene: sample a layout (scene_lib.sample_layout), write each task's BDDL, and validate every task:
  - stable resets: 3 resets, each settled, with every movable object drifting < 1 cm over the last 30 steps,
    nothing below the work surface, and the goal false;
  - forced goal: joints set and objects teleported into the goal configuration, settled, success true
    (and no object interpenetrating);
  - in view: the moved objects and destination items project inside the agentview image;
  - in reach: a proportional end-effector controller gets within REACH_TOL of a point above each moved object
    and of each destination's approach point.
A failed task rejects the layout; up to --tries layouts per scene. Accepted scenes get --init start states per
task (settled resets passing the same checks), saved in LIBERO's .pruned_init format.

Outputs: src/lerobot/envs/libero_ext/files/*.bddl|*.pruned_init, src/lerobot/envs/libero_ext/suite.json,
projects/dataset_gen_build.md (validation report), and a contact sheet of first start states (--sheet).

  python scripts/libero_tasks/build_suite.py [--scenes 1,5] [--workers 10] [--tries 40] [--init 50]
"""

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_lib as sl  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")
EXT = os.path.join(ROOT, "src", "lerobot", "envs", "libero_ext")
FILES = os.path.join(EXT, "files")
REPORT = os.path.join(ROOT, "projects", "dataset_gen_build.md")
DRIFT_TOL, REACH_TOL = 0.01, 0.05
# Expert check of an accepted layout (dataset_gen.md §3): the scripted expert must succeed in EXPERT_MIN_OK of
# EXPERT_EPISODES episodes of every task, from fresh resets, or the layout is rejected.
EXPERT_EPISODES, EXPERT_MIN_OK = 3, 2


def expert_check(path, scene, task, seed):
    """Successes of the scripted expert on one task of a laid-out scene."""
    import expert as ex

    env = sl.make_env(path)
    rng = np.random.default_rng(seed)
    task_x = {"steps": [list(s) for s in task["steps"]], "artic": task["artic"]}
    ok = 0
    for _ in range(EXPERT_EPISODES):
        sl.stable_reset(env)
        try:
            ok += int(ex.Expert(env, task_x, scene.cls, rng).run()[0])
        except Exception:  # noqa: BLE001
            pass
    env.close()
    return ok


def task_name(sid, language):
    return f"S{sid:02d}_" + re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")


def approach_points(e, scene, task):
    """Points the end effector must reach: above each moved object, and each destination's approach point."""
    pts = []
    for obj, dest, phrase in task["steps"]:
        inst = scene.inst[obj]
        p = sl.body_pos(e, inst)
        pts.append((f"above {obj}", np.array([p[0], p[1], sl._top_of(e, inst) + 0.05])))
        if dest == "lift":
            continue
        if dest == "beside":
            g = [x for x in e.parsed_problem["goal_state"] if x[1] == inst][0]
            c, _, hi = sl._site_box(e, g[2])
            pts.append((f"{dest}", np.array([c[0], c[1], sl.surface_z_of(e) + 0.10])))
            continue
        item = sl.DEST_ITEM[dest]
        t = scene.inst[item]
        if sl.DEST_GOAL[dest][1] == "object" or sl.DEST_GOAL[dest][1] == "table_at_item":
            q = sl.body_pos(e, t)
            pts.append((f"{dest}", np.array([q[0], q[1], sl._top_of(e, t) + 0.06])))
            continue
        g = [x for x in e.parsed_problem["goal_state"] if x[1] == inst and len(x) == 3][0]
        c, lo, hi = sl._site_box(e, g[2])
        if dest in ("fridge", "microwave", "bookcase") or (dest == "shelf" and "lower" in phrase):
            # front opening: 6 cm in front of the fixture's face, approached from further out
            s = 1.0 if scene.pos[item][1] > 0 else -1.0
            face = scene.pos[item][1] - s * sl.footprint(item, scene.pos[item][2])[1]
            tgt = np.array([c[0], face - s * 0.06, c[2]])
            pts.append((f"{dest} opening", tgt, tgt + np.array([0.0, -s * 0.10, 0.0])))
        elif dest == "drawer":
            s = 1.0 if scene.pos[item][1] > 0 else -1.0
            face = scene.pos[item][1] - s * sl.footprint(item, scene.pos[item][2])[1]
            if (
                task.get("artic") == "drawer"
            ):  # closed at the start: its front (handle), approached from outside
                tgt = np.array(
                    [c[0], face - s * 0.07, c[2]]
                )  # just clear of the handle, which stands ~4 cm proud
                pts.append(("drawer front", tgt, tgt + np.array([0.0, -s * 0.10, 0.0])))
            else:  # open at the start: just above its pulled-out part
                pts.append((f"{dest}", np.array([c[0], face - s * 0.08, hi[2] + 0.04])))
        elif dest in ("cabinet top", "shelf") and sl.ITEMS[item][3]:
            # top of a side fixture: its inner third (towards the table centre) must be in view and in reach
            s = 1.0 if scene.pos[item][1] > 0 else -1.0
            half = (hi[1] - lo[1]) / 2
            pts.append((f"{dest}", np.array([c[0], c[1] - s * half * 0.6, hi[2] + 0.08])))
        else:
            pts.append((f"{dest}", np.array([c[0], c[1], hi[2] + 0.08])))
    if task.get("artic") == "faucet":
        q = sl.body_pos(e, scene.inst["faucet"])
        pts.append(("faucet", np.array([q[0], q[1], sl._top_of(e, scene.inst["faucet"]) + 0.03])))
    return pts


def validate_task(path, scene, task):
    """Return (ok, details dict)."""
    det = {"drift": 0.0, "low": 1.0, "start_success": False, "forced": False, "view": True, "reach": {}}
    env = sl.make_env(path)
    try:
        for _ in range(3):
            try:
                drift, low, _ = sl.stable_reset(env)
            except Exception as ex:  # placement sampling can fail in a crowded layout
                det["error"] = f"reset: {type(ex).__name__}"
                return False, det
            det["drift"] = max(det["drift"], drift)
            det["low"] = min(det["low"], low)
            if env.check_success():
                det["start_success"] = True
        if det["drift"] > DRIFT_TOL or det["low"] < -0.05 or det["start_success"]:
            return False, det
        e = env.env
        # in view: the moved objects and every approach point (above objects, destination openings / tops)
        aps = approach_points(e, scene, task)
        pts = [sl.body_pos(e, scene.inst[o]) for o, _, _ in task["steps"]] + [ap[1] for ap in aps]
        pix = sl.project(e, pts)
        det["view"] = bool(np.all((pix > 4) & (pix < 252)))
        if not det["view"]:
            return False, det
        # in reach (each from a fresh reset)
        for ap in aps:
            label, pt, via = ap[0], ap[1], (ap[2] if len(ap) > 2 else None)
            env.reset()
            d = sl.reach_test(env, pt, via=via)
            det["reach"][label] = round(d, 3)
        if max(det["reach"].values(), default=0) > REACH_TOL:
            return False, det
        # forced goal
        env.reset()
        sl.settle(env, 10)
        det["forced"] = sl.force_goal(env)
        return det["forced"], det
    finally:
        env.close()


def init_states(path, n, max_tries=None):
    import torch  # noqa: F401  (LIBERO's .pruned_init files are torch-saved numpy arrays)

    env = sl.make_env(path)
    states, tries = [], 0
    max_tries = max_tries or 3 * n
    while len(states) < n and tries < max_tries:
        tries += 1
        try:
            drift, low, _ = sl.stable_reset(env)
        except Exception:
            continue
        if drift > DRIFT_TOL or low < -0.05 or env.check_success():
            continue
        states.append(env.get_sim_state())
    env.close()
    return np.array(states), tries


def build_scene(args):
    scene_spec, tries, n_init, out_dir = args
    sid, room = scene_spec["id"], scene_spec["room"]
    kinds = set(scene_spec["items"])
    tasks = []
    for t in scene_spec["tasks"]:
        tasks.append(
            {
                "language": t["language"],
                "kind": t["kind"],
                "artic": t["artic"],
                "artic_variant": t.get("artic_variant"),
                "steps": [tuple(s) for s in t["steps"]],
            }
        )
    opens = set()
    for t in tasks:
        for _, dest, _ in t["steps"]:
            if dest in ("drawer", "microwave", "fridge"):
                opens.add("cabinet" if dest == "drawer" else dest)
        if t["artic"] in ("drawer", "microwave", "fridge"):
            opens.add("cabinet" if t["artic"] == "drawer" else t["artic"])
    work = {("cabinet" if t["artic"] == "drawer" else t["artic"]) for t in tasks if t["artic"] in ("drawer", "microwave")}
    reserve = []
    for t in tasks:
        for obj, dest, phrase in t["steps"]:
            if dest == "beside":
                rel = next(r for r in sl.REL_DIR if phrase.startswith(r))
                ref = phrase[len(rel) :].replace(" the ", "", 1).strip()
                reserve.append((ref, sl.REL_DIR[rel], obj))
    log = []
    tmpdir = tempfile.mkdtemp(prefix=f"scene{sid}_")
    t0 = time.time()
    for attempt in range(tries):
        rng = random.Random(1000 * sid + attempt)
        scene = sl.Scene(sid, room, kinds, rng, variant=attempt // 10)
        if not sl.sample_layout(scene, kinds, reserve, rng, opens=opens, work=work):
            log.append((attempt, "layout", "no non-overlapping layout"))
            continue
        ok_all, dets = True, []
        for t in tasks:
            path = os.path.join(tmpdir, task_name(sid, t["language"]) + ".bddl")
            Path(path).write_text(sl.write_bddl(scene, t, t["language"]))
            try:
                ok, det = validate_task(path, scene, t)
            except Exception as ex:  # noqa: BLE001
                ok, det = (
                    False,
                    {
                        "error": f"{type(ex).__name__}: {str(ex)[:160]}",
                        "trace": traceback.format_exc()[-600:],
                    },
                )
            dets.append(det)
            if not ok:
                ok_all = False
                log.append((attempt, t["language"], {k: v for k, v in det.items() if k != "trace"}))
                break
        if not ok_all:
            continue
        for k, t in enumerate(tasks):
            path = os.path.join(tmpdir, task_name(sid, t["language"]) + ".bddl")
            n_ok = expert_check(path, scene, t, 1000 * sid + 10 * attempt + k)
            if n_ok < EXPERT_MIN_OK:
                ok_all = False
                log.append((attempt, t["language"], f"expert {n_ok}/{EXPERT_EPISODES}"))
                break
            dets[k]["expert"] = n_ok
        if not ok_all:
            continue
        # accepted: start states, final files
        out = []
        for t, det in zip(tasks, dets, strict=True):
            name = task_name(sid, t["language"])
            src = os.path.join(tmpdir, name + ".bddl")
            states, n_tries = init_states(src, n_init)
            if len(states) < n_init:
                ok_all = False
                log.append(
                    (attempt, t["language"], f"only {len(states)} valid start states in {n_tries} resets")
                )
                break
            text = Path(src).read_text()
            Path(os.path.join(out_dir, name + ".bddl")).write_text(text)
            import torch

            torch.save(states, os.path.join(out_dir, name + ".pruned_init"))
            out.append(
                {
                    "name": name,
                    "language": t["language"],
                    "bddl_file": name + ".bddl",
                    "init_file": name + ".pruned_init",
                    "scene": sid,
                    "room": room,
                    "kind": t["kind"],
                    "steps": [list(s) for s in t["steps"]],
                    "artic": t["artic"],
                    "validation": {
                        "drift_m": round(det["drift"], 4),
                        "min_height_m": round(det["low"], 3),
                        "reach_m": det["reach"],
                        "init_tries": n_tries,
                        "expert_ok": f"{det.get('expert')}/{EXPERT_EPISODES}",
                    },
                }
            )
        if not ok_all:
            continue
        return {
            "scene": sid,
            "room": room,
            "ok": True,
            "attempts": attempt + 1,
            "tasks": out,
            "layout": {k: [round(v, 3) for v in p] for k, p in scene.pos.items()},
            "classes": scene.cls,
            "log": log[-5:],
            "secs": round(time.time() - t0),
        }
    return {
        "scene": sid,
        "room": room,
        "ok": False,
        "attempts": tries,
        "tasks": [],
        "log": log[-8:],
        "secs": round(time.time() - t0),
    }


def contact_sheet(suite, out_png, size=224, cols=6):
    import torch
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from PIL import Image, ImageDraw

    tiles = []
    for t in suite["tasks"]:
        env = sl.make_env(os.path.join(FILES, t["bddl_file"]), render=True, size=size)
        env.reset()
        st = torch.load(os.path.join(FILES, t["init_file"]), weights_only=False)
        obs = env.set_init_state(st[0])
        for _ in range(5):
            obs, *_ = env.step(sl.NOOP)
        img = Image.fromarray(obs["agentview_image"][::-1, ::-1].copy())
        env.close()
        canvas = Image.new("RGB", (size, size + 34), "white")
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        words, line, lines = t["language"].split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 40:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        d.text((3, size + 1), f"S{t['scene']:02d} " + "\n".join(lines[:2]), fill="black")
        tiles.append(canvas)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * size, rows * (size + 34)), "white")
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % cols) * size, (i // cols) * (size + 34)))
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--init", type=int, default=50)
    ap.add_argument("--sheet", default=None, help="write a contact sheet PNG of first start states here")
    a = ap.parse_args()
    plan = json.loads(Path(PLAN).read_text())
    scenes = plan["scenes"]
    if a.scenes:
        keep = {int(x) for x in a.scenes.split(",")}
        scenes = [s for s in scenes if s["id"] in keep]
    os.makedirs(FILES, exist_ok=True)
    with Pool(a.workers) as pool:
        results = pool.map(build_scene, [(s, a.tries, a.init, FILES) for s in scenes], chunksize=1)
    # merge with earlier results for scenes not rebuilt now
    prev_path = os.path.join(EXT, "build_results.json")
    prev = (
        {r["scene"]: r for r in json.loads(Path(prev_path).read_text())} if os.path.exists(prev_path) else {}
    )
    for r in results:
        prev[r["scene"]] = r
    allres = [prev[k] for k in sorted(prev)]
    Path(prev_path).write_text(json.dumps(allres, indent=1))
    suite = {
        "name": "libero_ext",
        "plan_seed": plan["seed"],
        "tasks": [t for r in allres if r["ok"] for t in r["tasks"]],
    }
    Path(os.path.join(EXT, "suite.json")).write_text(json.dumps(suite, indent=1))
    # report
    lines = [
        "# libero_ext build report",
        "",
        f"Built by `scripts/libero_tasks/build_suite.py` from `scripts/libero_tasks/plan.json` (planner seed "
        f"{plan['seed']}). Checks per task: stable settled resets (drift < {DRIFT_TOL * 100:.0f} cm), goal false "
        f"at start, forced goal reached without interpenetration, objects in the agentview image, end effector "
        f"within {REACH_TOL * 100:.0f} cm of each approach point, and the scripted expert succeeding in at least "
        f"{EXPERT_MIN_OK} of {EXPERT_EPISODES} episodes. {a.init} start states per task.",
        "",
        f"**{len(suite['tasks'])} tasks in {sum(r['ok'] for r in allres)} of {len(allres)} scenes built.**",
        "",
    ]
    for r in allres:
        lines.append(
            f"## S{r['scene']:02d} ({r['room']}): {'built' if r['ok'] else 'FAILED'} after {r['attempts']} "
            f"layout(s), {r['secs']} s"
        )
        lines.append("")
        if r["ok"]:
            lines.append("Classes: " + ", ".join(f"{k} = {v}" for k, v in sorted(r["classes"].items())) + ".")
            lines.append("")
            for t in r["tasks"]:
                v = t["validation"]
                reach = ", ".join(f"{k} {x * 100:.1f}" for k, x in v["reach_m"].items())
                lines.append(
                    f"- {t['language']}: drift {v['drift_m'] * 100:.2f} cm; reach (cm) {reach}; "
                    f"{v['init_tries']} resets for the start states; expert {v.get('expert_ok', '-')}"
                )
        if r["log"]:
            lines.append("")
            lines.append(
                "Rejected layouts (last few): " + "; ".join(f"#{x[0]} {x[1]}: {x[2]}" for x in r["log"])
            )
        lines.append("")
    Path(REPORT).write_text("\n".join(lines))
    print(
        f"{len(suite['tasks'])} tasks; scenes built: {[r['scene'] for r in allres if r['ok']]}; "
        f"failed: {[r['scene'] for r in allres if not r['ok']]}"
    )
    if a.sheet:
        contact_sheet(suite, a.sheet)
        print(f"contact sheet: {a.sheet}")


if __name__ == "__main__":
    main()
