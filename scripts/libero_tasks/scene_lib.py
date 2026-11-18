"""Authoring library for the libero_ext suite (projects/dataset_gen.md): item catalogue, scene layout, BDDL
writing, goal forcing and the validation checks. Used by pair_test.py and build_suite.py.

Coordinates are LIBERO's: relative to the arena's workspace offset; +x is towards the agentview camera (away from
the robot), +y is the camera image's left. LIBERO's "left of X" is -y and "right of X" is +y (living_room_table
plate_left/right regions), "front" is +x.
"""

from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# ---------------------------------------------------------------- arenas
# room -> (BDDL problem name, workspace name, workspace fixture category, x bounds, y bounds) of the layout area
ARENAS = {  # layout bounds for item centres, from reach maps (end effector within 3 cm at surface + 6 cm)
    "kitchen": (
        "LIBERO_Kitchen_Tabletop_Manipulation",
        "kitchen_table",
        "kitchen_table",
        (-0.32, 0.08),
        (-0.32, 0.32),
    ),
    "study": (
        "LIBERO_Study_Tabletop_Manipulation",
        "study_table",
        "study_table",
        (-0.26, 0.16),
        (-0.32, 0.32),
    ),
    "tabletop": ("LIBERO_Tabletop_Manipulation", "main_table", "table", (-0.32, 0.08), (-0.32, 0.32)),
    "living room": (
        "LIBERO_Living_Room_Tabletop_Manipulation",
        "living_room_table",
        "living_room_table",
        (-0.20, 0.20),
        (-0.32, 0.32),
    ),
    "coffee table": (
        "LIBERO_Coffee_Table_Manipulation",
        "coffee_table",
        "coffee_table",
        (-0.20, 0.20),
        (-0.32, 0.32),
    ),
    "floor": ("LIBERO_Floor_Manipulation", "floor", "floor", (-0.30, 0.10), (-0.32, 0.32)),
}

# ---------------------------------------------------------------- items
# kind -> (LIBERO classes, fixture?, (x, y, z) extent at yaw 0 in m, front-facing?)
# Front-facing fixtures open towards local -y; LIBERO puts them at the table's sides: yaw 0 at +y, pi at -y.
ITEMS = {
    "cabinet": (["wooden_cabinet", "white_cabinet"], True, (0.255, 0.221, 0.218), True),
    "microwave": (["microwave"], True, (0.345, 0.263, 0.189), True),
    "fridge": (["short_fridge"], True, (0.202, 0.233, 0.261), True),
    "shelf": (["wooden_two_layer_shelf"], True, (0.352, 0.191, 0.210), True),
    "bookcase": (["wooden_shelf"], True, (0.256, 0.191, 0.217), True),
    "caddy": (["desk_caddy"], True, (0.151, 0.434, 0.167), False),
    "stove": (["flat_stove"], True, (0.297, 0.190, 0.070), False),
    "faucet": (["faucet"], True, (0.092, 0.238, 0.145), False),
    "drainer": (["bowl_drainer"], True, (0.205, 0.232, 0.144), False),
    "storage box": (["white_storage_box"], True, (0.142, 0.117, 0.139), False),
    "placemat": (["dining_set_group"], True, (0.280, 0.160, 0.034), False),
    "basket": (["basket"], False, (0.157, 0.170, 0.141), False),
    "tray": (["wooden_tray"], False, (0.303, 0.167, 0.114), False),
    "mug": (["porcelain_mug", "red_coffee_mug", "white_yellow_mug"], False, (0.10, 0.14, 0.13), False),
    "bowl": (["akita_black_bowl", "white_bowl"], False, (0.107, 0.107, 0.051), False),
    "ramekin": (["glazed_rim_porcelain_ramekin"], False, (0.089, 0.089, 0.042), False),
    "plate": (["plate"], False, (0.138, 0.138, 0.019), False),
    "book": (["black_book", "yellow_book"], False, (0.109, 0.082, 0.134), False),
    "frying pan": (["chefmate_8_frypan"], False, (0.204, 0.356, 0.058), False),
    "moka pot": (["moka_pot"], False, (0.081, 0.148, 0.152), False),
    "wine bottle": (["wine_bottle"], False, (0.044, 0.044, 0.157), False),
    "alphabet soup": (["alphabet_soup"], False, (0.063, 0.063, 0.076), False),
    "tomato sauce": (["tomato_sauce"], False, (0.063, 0.062, 0.076), False),
    "ketchup": (["ketchup"], False, (0.037, 0.056, 0.146), False),
    "bbq sauce": (["bbq_sauce"], False, (0.029, 0.047, 0.107), False),
    "salad dressing": (["salad_dressing", "new_salad_dressing"], False, (0.036, 0.053, 0.146), False),
    "milk": (["milk"], False, (0.053, 0.053, 0.131), False),
    "orange juice": (["orange_juice"], False, (0.053, 0.053, 0.131), False),
    "butter": (["butter"], False, (0.076, 0.040, 0.017), False),
    "cream cheese": (["cream_cheese"], False, (0.081, 0.043, 0.018), False),
    "chocolate pudding": (["chocolate_pudding"], False, (0.080, 0.046, 0.027), False),
    "cookies": (["cookies"], False, (0.083, 0.062, 0.019), False),
    "macaroni": (["macaroni_and_cheese"], False, (0.025, 0.084, 0.116), False),
    "popcorn": (["popcorn"], False, (0.062, 0.042, 0.020), False),
}

# Destination kind -> how the goal reads. ("site", region) = a site of the destination item; ("object",) = on the
# item itself; ("table",) = a table region beside a reference item; ("lifted",). Prep: In or On.
FRIDGE_LEVEL = "middle_region"
BOOKCASE_LEVEL = "middle_region"
DEST_GOAL = {
    "basket": ("In", "site", "contain_region"),
    "tray": ("In", "site", "contain_region"),
    "caddy": ("In", "site", "{p}_contain_region"),
    "drainer": ("In", "site", "{p}_region"),
    "drawer": ("In", "site", "{p}_region"),
    "microwave": ("In", "site", "heating_region"),
    "fridge": ("In", "site", FRIDGE_LEVEL),
    "shelf": (None, "site", None),  # variant-dependent, see goal_for
    "bookcase": ("In", "site", BOOKCASE_LEVEL),
    "pan": ("On", "object", None),
    "bowl": ("On", "object", None),
    "plate": ("On", "object", None),
    "cabinet top": ("On", "site", "top_side"),
    "microwave top": ("On", "site", "top_side"),
    "storage box": ("On", "site", "top_side"),
    "placemat": ("On", "table_at_item", None),  # the mat's cloth does not collide: a table region over it
    "beside": ("On", "table", None),
    "lift": ("Lifted", None, None),
}
DEST_ITEM = {
    "basket": "basket",
    "tray": "tray",
    "caddy": "caddy",
    "drainer": "drainer",
    "drawer": "cabinet",
    "microwave": "microwave",
    "fridge": "fridge",
    "shelf": "shelf",
    "bookcase": "bookcase",
    "pan": "frying pan",
    "bowl": "bowl",
    "plate": "plate",
    "cabinet top": "cabinet",
    "microwave top": "microwave",
    "storage box": "storage box",
    "placemat": "placemat",
}
# Doors and drawers that must be open for a placing step into them (unless the task itself opens them).
NEEDS_OPEN = {"drawer", "microwave", "fridge"}
REL_DIR = {
    "to the left of": (0.0, -1.0, "left"),
    "to the right of": (0.0, 1.0, "right"),
    "in front of": (1.0, 0.0, "front"),
    "behind": (-1.0, 0.0, "behind"),
}


def footprint(kind, yaw=0.0):
    """Axis-aligned half extents (hx, hy) at a yaw of 0, pi/2, pi or -pi/2."""
    w, d, _ = ITEMS[kind][2]
    if abs(math.sin(yaw)) > 0.7:
        w, d = d, w
    return w / 2, d / 2


# ---------------------------------------------------------------- scene instances
class Scene:
    """A laid-out scene: room, instances (kind -> LIBERO class and instance name) and positions."""

    def __init__(self, sid, room, kinds, rng, variant=0):
        self.sid, self.room = sid, room
        self.problem, self.ws, self.ws_cat, self.xb, self.yb = ARENAS[room]
        self.cls = {}
        for k in sorted(kinds):
            opts = ITEMS[k][0]
            self.cls[k] = opts[(sid + variant) % len(opts)]
        self.inst = {k: f"{c}_1" for k, c in self.cls.items()}
        self.pos = {}  # kind -> (x, y, yaw)
        self.rng = rng

    def is_fixture(self, k):
        return ITEMS[k][1]


# Depth kept free in front of a fixture whose door or drawers some task of the scene uses (open at the start or
# opened during the task): drawers pull out ~15 cm; the fridge and microwave doors swing through ~20 cm.
CLEARANCE = {"cabinet": 0.17, "fridge": 0.22, "microwave": 0.18}
# Larger strips where the robot itself opens or closes the fixture: the expert pulls a drawer with its hand turned
# horizontal (fingers above and below the handle), and the hand reaches ~10 cm below the handle and ~25 cm out
WORK_CLEARANCE = {"cabinet": 0.28, "microwave": 0.22}
FACE_Y = 0.28  # |y| of a side fixture's front face: its opening and the inner part of its top stay in view


def sample_layout(scene, kinds, reserve_beside, rng, opens=(), work=()):
    """Place every item without overlap inside the arena's layout bounds (item centres; extents may overhang by
    a few cm). Front-facing fixtures go to the table's sides and face the centre, as in LIBERO, with a clearance
    strip kept free in front for doors and drawers. Other items, largest first, take the first free spot of a
    shuffled 2 cm grid. A frying pan in a scene with a stove sits on the stove (scene.pan_on_stove). reserve_beside
    lists (ref kind, direction) whose beside-region must stay free; opens lists the fixtures whose doors or
    drawers are used (they get CLEARANCE; work lists those the robot opens or closes itself: WORK_CLEARANCE). Side fixtures sit outside the reach bounds, front face at |y| = FACE_Y.
    Returns True on success."""
    (x0, x1), (y0, y1) = scene.xb, scene.yb
    boxes = []  # (cx, cy, hx, hy)
    scene.pan_on_stove = "stove" in kinds and "frying pan" in kinds

    def free(cx, cy, hx, hy, margin=0.015):
        return all(
            abs(cx - bx) >= hx + bhx + margin or abs(cy - by) >= hy + bhy + margin
            for bx, by, bhx, bhy in boxes
        )

    front = [k for k in kinds if ITEMS[k][3]]
    rng.shuffle(front)
    sides = [1, -1]
    rng.shuffle(sides)
    for i, k in enumerate(front):
        s = sides[i % 2]
        yaw = 0.0 if s > 0 else math.pi
        hx, hy = footprint(k, yaw)
        cy = s * (FACE_Y + hy)
        xs = list(np.arange(x0 + hx - 0.12, x1 - hx + 0.08 + 1e-9, 0.02))
        rng.shuffle(xs)
        for cx in xs:
            if free(cx, cy, hx, hy):
                boxes.append((cx, cy, hx, hy))
                c = CLEARANCE.get(k, 0.0) if k in opens else 0.0
                if k in work:
                    c = max(c, WORK_CLEARANCE.get(k, 0.0))
                if c:
                    boxes.append((cx, cy - s * (hy + c / 2), hx, c / 2))
                scene.pos[k] = (float(cx), float(cy), yaw)
                break
        else:
            return False
    others = sorted(
        [k for k in kinds if not ITEMS[k][3] and not (k == "frying pan" and scene.pan_on_stove)],
        key=lambda k: -max(ITEMS[k][2][:2]),
    )
    for k in others:
        # low fixtures may turn 90 degrees to pack better (movable objects keep LIBERO's fixed orientation)
        yaws = [math.pi, math.pi / 2] if k == "caddy" else ([0.0, math.pi / 2] if ITEMS[k][1] else [0.0])
        rng.shuffle(yaws)
        cands = []
        for yaw in yaws:
            hx, hy = footprint(k, yaw)
            if k == "book":  # LIBERO's books spawn at a random lean: use the enclosing square
                hx = hy = max(hx, hy)
            cands += [
                (x, y, yaw, hx, hy)
                for x in np.arange(x0 + min(hx, 0.08) - 0.02, x1 - min(hx, 0.08) + 0.02 + 1e-9, 0.02)
                for y in np.arange(y0 + min(hy, 0.10) - 0.03, y1 - min(hy, 0.10) + 0.03 + 1e-9, 0.02)
            ]
        rng.shuffle(cands)
        for cx, cy, yaw, hx, hy in cands:
            if free(cx, cy, hx, hy):
                boxes.append((cx, cy, hx, hy))
                scene.pos[k] = (float(cx), float(cy), yaw)
                break
        else:
            return False
    if scene.pan_on_stove:
        sx, sy, _ = scene.pos["stove"]
        scene.pos["frying pan"] = (sx, sy, 0.0)  # notional: its BDDL init puts it on the stove's cook region
    # beside regions must be inside the bounds and free of other items
    for entry in reserve_beside:
        ref, (dx, dy, _) = entry[0], entry[1]
        obj = entry[2] if len(entry) > 2 else None
        tx, ty = beside_point(scene, ref, obj, dx, dy)
        if not (x0 - 0.02 <= tx <= x1 + 0.02 and y0 - 0.02 <= ty <= y1 + 0.02):
            return False
        ohx, ohy = footprint(obj) if obj else (0.05, 0.05)
        for k, (bx, by, yaw) in scene.pos.items():
            if k in (ref, obj) or (k == "frying pan" and scene.pan_on_stove):
                continue
            bhx, bhy = footprint(k, yaw)
            if abs(tx - bx) < max(0.05, ohx) + bhx and abs(ty - by) < max(0.05, ohy) + bhy:
                return False
    return True


def beside_point(scene, ref, obj, dx, dy):
    """Centre of the table region beside ref: clear of ref by the moved object's own half extent plus 2 cm (the
    first build used a fixed 7 cm, which put a 36 cm frying pan on top of the basket it was to sit beside)."""
    cx, cy, _ = scene.pos[ref]
    hx, hy = footprint(ref, scene.pos[ref][2])
    ohx, ohy = footprint(obj) if obj else (0.05, 0.05)
    if dx:
        return cx + dx * (hx + max(ohx, 0.05) + 0.02), cy
    return cx, cy + dy * (hy + max(ohy, 0.05) + 0.02)


# ---------------------------------------------------------------- BDDL
def _region(name, target, cx=None, cy=None, half=None, yaw=None):
    if cx is None:
        return f"      ({name}\n          (:target {target})\n      )"
    return (
        f"      ({name}\n          (:target {target})\n          (:ranges (\n              ({cx - half:.4f} {cy - half:.4f} "
        f"{cx + half:.4f} {cy + half:.4f})\n            )\n          )\n          (:yaw_rotation (\n              "
        f"({yaw:.4f} {yaw:.4f})\n            )\n          )\n      )"
    )


def goal_for(scene, step, variant_phrase):
    """BDDL goal predicate(s), extra regions and the required open state for one placing step.

    step = (object kind, destination kind). Returns (goal strings, region decls, open-needed item instance or
    None, beside region spec or None)."""
    obj, dest = step
    o = scene.inst[obj]
    if dest == "lift":
        return [f"(Lifted {o})"], [], None, None
    if dest == "beside":
        rel = next(r for r in REL_DIR if variant_phrase.startswith(r))
        ref = variant_phrase[len(rel) :].replace(" the ", "", 1).strip()
        dx, dy, tag = REL_DIR[rel]
        cx, cy, _ = scene.pos[ref]
        tx, ty = beside_point(scene, ref, obj, dx, dy)
        rname = f"{scene.cls[ref]}_{tag}_region"
        return (
            [f"(On {o} {scene.ws}_{rname})"],
            [_region(rname, scene.ws, tx, ty, 0.05, 0.0)],
            None,
            (ref, tag),
        )
    item = DEST_ITEM[dest]
    t = scene.inst[item]
    pred, kind, region = DEST_GOAL[dest]
    if dest == "shelf":
        pred, region = ("On", "top_side") if "lower" not in variant_phrase else ("In", "bottom_region")
    p = None
    for word in ("front", "back", "left", "right", "top", "middle", "bottom"):
        if f" {word} " in f" {variant_phrase} ":
            p = word
            break
    if kind == "object":
        return [f"({pred} {o} {t})"], [], None, None
    if kind == "table_at_item":
        cx, cy, _ = scene.pos[item]
        rname = f"{scene.cls[item]}_mat_region"
        return [f"(On {o} {scene.ws}_{rname})"], [_region(rname, scene.ws, cx, cy, 0.06, 0.0)], None, None
    region = region.format(p=p)
    need_open = None
    if dest in NEEDS_OPEN:
        need_open = f"{t}_{region}" if dest == "drawer" else t
    return [f"({pred} {o} {t}_{region})"], [_region(region, t)], need_open, None


def write_bddl(scene, task, language):
    """BDDL text for one task of a laid-out scene.

    task: dict with "steps" [(obj, dest, phrase)], and "artic" (fixture kind or None) with "artic_variant"."""
    regions, goals, inits, extra_regions = [], [], [], []
    open_needed, closed_needed = set(), set()
    pan_on_stove = getattr(scene, "pan_on_stove", False)
    for k, (cx, cy, yaw) in sorted(scene.pos.items()):
        if k == "frying pan" and pan_on_stove:
            extra_regions.append(_region("cook_region", scene.inst["stove"]))
            inits.append(f"(On {scene.inst[k]} {scene.inst['stove']}_cook_region)")
            continue
        half = 0.01 if ITEMS[k][1] else 0.015
        regions.append(_region(f"{scene.cls[k]}_init_region", scene.ws, cx, cy, half, yaw))
        inits.append(f"(On {scene.inst[k]} {scene.ws}_{scene.cls[k]}_init_region)")
    for obj, dest, phrase in task["steps"]:
        g, r, need_open, _ = goal_for(scene, (obj, dest), phrase)
        goals += g
        extra_regions += r
        if need_open:
            open_needed.add(need_open)
    art = task.get("artic")
    if art == "faucet":
        goals.append(f"(Turnon {scene.inst['faucet']})")
        inits.append(f"(Turnoff {scene.inst['faucet']})")
    elif art == "fridge":  # open the fridge and put X in it
        goals.append(f"(Open {scene.inst['fridge']})")
        open_needed.discard(scene.inst["fridge"])
        closed_needed.add(scene.inst["fridge"])
    elif art == "microwave":  # put X in the microwave and close it: starts open
        goals.append(f"(Close {scene.inst['microwave']})")
    elif art == "drawer":  # open the p drawer and put X in it
        p = task["artic_variant"]
        site = f"{scene.inst['cabinet']}_{p}_region"
        goals.append(f"(Open {site})")
        open_needed.discard(site)
        closed_needed.add(site)
        extra_regions.append(_region(f"{p}_region", scene.inst["cabinet"]))
    elif art == "stove":  # turn on the stove and put X in the pan (the pan starts on the stove: pan_on_stove)
        goals.append(f"(Turnon {scene.inst['stove']})")
        inits.append(f"(Turnoff {scene.inst['stove']})")
    for s in sorted(open_needed):
        inits.append(f"(Open {s})")
    for s in sorted(closed_needed):
        inits.append(f"(Close {s})")
    # de-duplicate region declarations (same site used twice)
    seen, regs = set(), []
    for r in regions + extra_regions:
        key = r.split("\n")[0].strip() + r.split("\n")[1].strip()
        if key not in seen:
            seen.add(key)
            regs.append(r)
    fixtures = [f"    {scene.ws} - {scene.ws_cat}"] + [
        f"    {scene.inst[k]} - {scene.cls[k]}" for k in sorted(scene.pos) if ITEMS[k][1]
    ]
    objects = [f"    {scene.inst[k]} - {scene.cls[k]}" for k in sorted(scene.pos) if not ITEMS[k][1]]
    interest = sorted(
        {scene.inst[o] for o, _, _ in task["steps"]}
        | {scene.inst[DEST_ITEM[d]] for _, d, _ in task["steps"] if d in DEST_ITEM}
        | (
            {scene.inst[task["artic"] if task["artic"] != "drawer" else "cabinet"]}
            if task.get("artic")
            else set()
        )
    )
    goal = goals[0] if len(goals) == 1 else "(And " + " ".join(goals) + ")"
    if len(goals) == 1:
        goal = f"(And {goals[0]})"
    nl = "\n"
    return f"""(define (problem {scene.problem})
  (:domain robosuite)
  (:language {language})
    (:regions
{nl.join(regs)}
    )

  (:fixtures
{nl.join(fixtures)}
  )

  (:objects
{nl.join(objects)}
  )

  (:obj_of_interest
{nl.join("    " + i for i in interest)}
  )

  (:init
{nl.join("    " + i for i in inits)}
  )

  (:goal
    {goal}
  )

)
"""


# ---------------------------------------------------------------- simulation helpers
def make_env(bddl_path, render=False, size=256):
    import libero.libero as libero_pkg

    if os.path.isdir(
        "/home/josh/phddev/LIBERO/libero/libero/assets"
    ):  # desk PC: local assets, no hub download
        libero_pkg._assets_path_cache = "/home/josh/phddev/LIBERO/libero/libero/assets"
    from lerobot.envs.libero_ext import register_runtime

    register_runtime()
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    if render:
        return OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=size, camera_widths=size)
    return ControlEnv(
        bddl_file_name=bddl_path, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False
    )


NOOP = [0, 0, 0, 0, 0, 0, -1]


def movable_names(e):
    return list(e.objects_dict.keys())


def body_pos(e, name):
    return e.sim.data.body_xpos[e.obj_body_id[name]].copy()


def settle(env, n):
    for _ in range(n):
        env.step(NOOP)


SETTLE_STEPS, DRIFT_WINDOW, MAX_TILT_DEG = 80, 20, 25.0


def stable_reset(env, n_settle=SETTLE_STEPS):
    """Reset and settle; report (max displacement over the last DRIFT_WINDOW steps, min height of any movable
    object above the work surface, positions). An object that tipped over (tilt > MAX_TILT_DEG from its spawn
    orientation) counts as a drift of 1 m."""
    from lerobot.envs.libero_ext import lowest_point, surface_z

    env.reset()
    e = env.env
    q0 = {n: e.sim.data.body_xquat[e.obj_body_id[n]].copy() for n in movable_names(e)}
    settle(env, n_settle - DRIFT_WINDOW)
    p0 = {n: body_pos(e, n) for n in movable_names(e)}
    settle(env, DRIFT_WINDOW)
    p1 = {n: body_pos(e, n) for n in movable_names(e)}
    drift = max((np.linalg.norm(p1[n] - p0[n]) for n in p0), default=0.0)
    for n in p0:
        q = e.sim.data.body_xquat[e.obj_body_id[n]]
        if np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q, q0[n])))))) > MAX_TILT_DEG:
            drift = max(drift, 1.0)
    sz = surface_z(e)
    low = min((lowest_point(e, n) - sz for n in p0), default=0.0)
    return drift, low, p1


def _set_free_pose(e, name, pos):
    obj = e.objects_dict[name]
    jn = obj.joints[-1]
    q = e.sim.data.get_joint_qpos(jn).copy()
    q[:3] = pos
    e.sim.data.set_joint_qpos(jn, q)
    e.sim.data.set_joint_qvel(jn, np.zeros(6))


def _site_box(e, site):
    sid = e.sim.model.site_name2id(site)
    c = e.sim.data.site_xpos[sid].copy()
    rot = e.sim.data.site_xmat[sid].reshape(3, 3)
    h = e.sim.model.site_size[sid].copy()
    if e.sim.model.site_type[sid] != 6:  # not a box: treat as a sphere/point
        h = np.array([h[0]] * 3)
    ext = np.abs(rot) @ h
    return c, c - ext, c + ext


def _top_of(e, name):
    """Highest point of an object's geoms."""
    from lerobot.envs.libero_ext import _object_geoms

    model, data = e.sim.model._model, e.sim.data._data
    geoms, _ = _object_geoms(e, name)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    top = -np.inf
    for g in geoms:
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        corners = data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (c + signs * h).T).T
        top = max(top, corners[:, 2].max())
    return float(top)


def place_lowest_at(e, name, xy, z_low):
    from lerobot.envs.libero_ext import lowest_point

    p = body_pos(e, name)
    lp = lowest_point(e, name)
    _set_free_pose(e, name, np.array([xy[0], xy[1], z_low + (p[2] - lp)]))
    e.sim.forward()


def _set_articulation(e, target, state):
    st = e.object_states_dict[target]
    obj = e.get_object(st.parent_name) if st.object_state_type == "site" else e.get_object(target)
    art = obj.object_properties["articulation"]
    key = {
        "open": "default_open_ranges",
        "close": "default_close_ranges",
        "turnon": "default_turnon_ranges",
        "turnoff": "default_turnoff_ranges",
    }[state]
    rng_ = art.get(key) or []
    if not rng_:
        raise ValueError(f"no {key} for {target}")
    st.set_joint(float(np.mean(rng_)))
    e.sim.forward()


def max_penetration(e, names):
    """Deepest contact penetration (m, positive) involving the named objects' geoms."""
    from lerobot.envs.libero_ext import _object_geoms

    data = e.sim.data._data
    gs = set()
    for n in names:
        gs |= set(_object_geoms(e, n)[0].tolist())
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in gs or c.geom2 in gs:
            worst = max(worst, -float(c.dist))
    return worst


PENETRATION_TOL = 0.006


def force_goal(env, n_settle=60):
    """Put the scene into its goal state by setting joints and teleporting objects; return success.

    Order: open/turn-on first, then placements, then close/turn-off, then lifts (checked at once, before they fall)."""
    from lerobot.envs.libero_ext import surface_z

    e = env.env
    goal = e.parsed_problem["goal_state"]
    for g in goal:
        if g[0] in ("open", "turnon"):
            _set_articulation(e, g[1], g[0])
    for g in goal:
        if g[0] not in ("in", "on"):
            continue
        obj, tgt = g[1], g[2]
        if tgt in e.object_sites_dict and tgt not in e.objects_dict:
            c, lo, hi = _site_box(e, tgt)
            z = (lo[2] + 0.004) if g[0] == "in" else (hi[2] + 0.004)
            if tgt.startswith(e.workspace_name) or tgt.startswith("floor"):
                z = surface_z(e) + 0.004
            place_lowest_at(e, obj, c[:2], z)
        else:  # on an object: drop onto its top
            c = body_pos(e, tgt)
            place_lowest_at(e, obj, c[:2], _top_of(e, tgt) + 0.004)
    for g in goal:
        if g[0] in ("close", "turnoff"):
            _set_articulation(e, g[1], g[0])
    lifts = [g for g in goal if g[0] == "lifted"]
    moved = [g[1] for g in goal if g[0] in ("in", "on", "lifted")]
    if not lifts:
        settle(env, n_settle)
        if moved and max_penetration(e, moved) > PENETRATION_TOL:
            return False
        return bool(env.check_success())
    settle(env, n_settle)
    for g in lifts:
        p = body_pos(e, g[1])
        place_lowest_at(e, g[1], p[:2], surface_z(e) + 0.12)
    return bool(env.check_success())


def project(e, points, cam="agentview", h=256, w=256):
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    world_to_pix = get_camera_transform_matrix(e.sim, cam, h, w)
    pix = project_points_from_world_to_camera(np.array(points), world_to_pix, h, w)
    return pix  # (row, col)


def reach_test(env, target, steps=120, tol=0.03, via=None):
    """Drive the end effector towards a point (through an optional waypoint) with a proportional controller;
    return the closest distance to the target."""
    obs = env.env._get_observations()
    for wp in [via] if via is not None else []:
        for _ in range(steps):
            d = wp - obs["robot0_eef_pos"]
            if np.linalg.norm(d) < 0.02:
                break
            a = np.zeros(7)
            a[:3] = np.clip(d * 15.0, -1, 1)
            a[6] = -1
            obs, *_ = env.step(a)
    best = np.inf
    for _ in range(steps):
        d = target - obs["robot0_eef_pos"]
        best = min(best, float(np.linalg.norm(d)))
        if best < tol:
            break
        a = np.zeros(7)
        a[:3] = np.clip(d * 15.0, -1, 1)
        a[6] = -1
        obs, *_ = env.step(a)
    return best


def surface_z_of(e):
    from lerobot.envs.libero_ext import surface_z

    return surface_z(e)
