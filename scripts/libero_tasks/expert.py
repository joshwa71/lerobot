"""Scripted expert for the libero_ext suite (projects/dataset_gen.md, §3).

A closed-loop expert on privileged simulator state: object poses, destination sites, joint states. Each task's
plan comes from its suite.json steps; there is no per-task code. A step is a sequence of stages (move to a pose,
open or close the gripper, wait); each stage's target is computed from the current state when the stage starts.
Actions are LIBERO's: OSC_POSE deltas (action 1 commands 5 cm or 0.5 rad) plus the gripper (-1 open, +1 close).

Variety, so that 50 demos of a task are not near-copies (dataset_gen.md, §3):
  - expert parameters drawn per episode: speed, approach and transport heights, grasp offsets, release height and
    pauses (continuous only: discrete choices, such as step order or which of several grasps, are made the same
    way in every episode, or the policy gets several modes it cannot choose between; dataset_gen.md §3c);
  - DART-style noise: the executed action is perturbed, the recorded label is the expert's action from the state
    actually reached (so the data contains corrections);
  - wider starts: after a reset, objects are nudged and round ones turned, and the arm's joints jittered.
"""

from __future__ import annotations

import math

import numpy as np
from robosuite.utils import transform_utils as tu

import scene_lib as sl

ROUND = {"akita_black_bowl", "white_bowl", "glazed_rim_porcelain_ramekin", "plate", "alphabet_soup", "tomato_sauce"}
RIM = {"porcelain_mug", "red_coffee_mug", "white_yellow_mug", "akita_black_bowl", "white_bowl",
       "glazed_rim_porcelain_ramekin", "plate"}
MAX_OPEN = 0.075  # widest object the Panda gripper can close on, m
# Recovery data (dataset_gen.md §3c): the scripted expert's grasps almost never fail, so its demos held no
# recoveries, and a policy that closed 1-3 cm off a grasp carried an empty hand on or looped. This share of first
# attempts is executed wrong on purpose (labels: the expert's action from each state reached).
P_MISS = 0.25


# ---------------------------------------------------------------- geometry
def obj_box(e, name):
    """World AABB (lo, hi) of an object's geoms."""
    from lerobot.envs.libero_ext import _object_geoms

    model, data = e.sim.model._model, e.sim.data._data
    geoms, _ = _object_geoms(e, name)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    pts = []
    for g in geoms:
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual-only geom
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        pts.append(data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (c + signs * h).T).T)
    pts = np.concatenate(pts)
    return pts.min(0), pts.max(0)


def down_rot(close_dir):
    """Gripper pointing down with its fingers closing along the horizontal unit vector close_dir."""
    y = np.array([close_dir[0], close_dir[1], 0.0])
    y /= np.linalg.norm(y)
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def nearest_equiv(rot_target, rot_now):
    """The target, or the target turned 180 degrees about its z axis (same grasp), whichever is closer."""
    flip = rot_target @ np.diag([-1.0, -1.0, 1.0])
    err = [np.linalg.norm(tu.quat2axisangle(tu.mat2quat(r @ rot_now.T))) for r in (rot_target, flip)]
    return rot_target if err[0] <= err[1] else flip


def obstacles(e, exclude):
    """World AABBs (N x 2 x 3) of every collision geom of every object and fixture except those named in exclude
    (and the table). Per geom, so a cabinet's open drawer is not swallowed by the cabinet's box."""
    from lerobot.envs.libero_ext import _object_geoms

    model, data = e.sim.model._model, e.sim.data._data
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    boxes = []
    for name in list(e.objects_dict) + list(e.fixtures_dict):
        if name in exclude or name == e.workspace_name:
            continue
        try:
            geoms, _ = _object_geoms(e, name)
        except KeyError:
            continue
        for g in geoms:
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            pts = data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (c + signs * h).T).T
            boxes.append([pts.min(0), pts.max(0)])
    return np.array(boxes).reshape(-1, 2, 3)


def hand_boxes(pt, d, z_tip, open_w=0.08, fingers=True):
    """AABBs approximating the Panda gripper pointing down with fingertips at (pt, z_tip), closing along d:
    two fingers (5 cm tall) either side, and the hand body above them (about 21 x 6 cm)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])

    def box(c_off, half_d, half_n, z0, z1):
        c = np.asarray(pt[:2]) + c_off
        ex = np.abs(d) * half_d + np.abs(n) * half_n
        return np.array([[c[0] - ex[0], c[1] - ex[1], z0], [c[0] + ex[0], c[1] + ex[1], z1]])

    f = open_w / 2 + 0.008
    out = [box(np.zeros(2), 0.105, 0.032, z_tip + 0.052, z_tip + 0.25)]
    if fingers:
        out += [box(d * f, 0.009, 0.011, z_tip - 0.002, z_tip + 0.052),
                box(-d * f, 0.009, 0.011, z_tip - 0.002, z_tip + 0.052)]
    return out


def clearance(pt, d, z_tip, obs, fingers=True, open_w=0.08):
    """Smallest separation between the gripper's boxes and the obstacle boxes (negative: overlap)."""
    if len(obs) == 0:
        return 1.0
    best = 1.0
    for h in hand_boxes(pt, d, z_tip, open_w, fingers):
        gap = np.maximum(obs[:, 0, :] - h[1][None, :], h[0][None, :] - obs[:, 1, :])  # N x 3
        best = min(best, float(gap.max(axis=1).min()))
    return best


def mug_cup_and_handle(e, inst, lo, hi):
    """Cup centre (xy) and unit handle direction of a mug, from its collision boxes: the cup is the end of the
    mug's long horizontal axis where its wall pieces cluster; the handle sticks out of the other end."""
    from lerobot.envs.libero_ext import _object_geoms

    model, data = e.sim.model._model, e.sim.data._data
    geoms, _ = _object_geoms(e, inst)
    pts = np.array([data.geom_xpos[g][:2] for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]])
    ext = hi[:2] - lo[:2]
    ax = 0 if ext[0] >= ext[1] else 1
    r = min(ext) / 2
    best = None
    for end in (lo[ax] + r, hi[ax] - r):
        n = int(np.sum(np.abs(pts[:, ax] - end) <= r))
        if best is None or n > best[0]:
            best = (n, end)
    cup = (lo[:2] + hi[:2]) / 2
    cup[ax] = best[1]
    handle = np.zeros(2)
    handle[ax] = 1.0 if (lo[ax] + hi[ax]) / 2 > best[1] else -1.0
    return cup, handle


def grasp_candidates(e, inst, cls, rng, p):
    """Candidate top-down grasps (point, closing direction, top z) for an object, in random order."""
    lo, hi = obj_box(e, inst)
    body = sl.body_pos(e, inst)
    ext = hi - lo
    centre = (lo + hi) / 2
    cands = []
    if cls in RIM:  # rim grasp: one finger inside the wall, one outside, closing radially; mugs avoid the handle
        mug = cls.endswith("mug")
        cup, handle = body[:2].copy(), None
        if mug:
            cup, handle = mug_cup_and_handle(e, inst, lo, hi)
            r = min(ext[0], ext[1]) / 2
        else:
            cup = centre[:2]
            r = max(ext[0], ext[1]) / 2
        for k in range(12):
            a = k * math.pi / 6 + rng.uniform(-0.1, 0.1)
            dvec = np.array([math.cos(a), math.sin(a)])
            if handle is not None and np.dot(dvec, handle) > 0.0:  # stay on the side away from the handle
                continue
            inset = 0.012 if cls == "plate" else 0.008
            z = hi[2] - (0.004 if cls == "plate" else min(0.02, ext[2] * 0.5))
            pt = cup + dvec * (r - inset)
            cands.append((np.array([pt[0], pt[1], z]), dvec))
    else:  # body grasp: a direction and height at which the part of the object between the finger pads fits
        parts = self_parts(e, inst)
        top_z = hi[2] - p["grasp_depth"] if ext[2] >= 0.035 else lo[2] + ext[2] * 0.55
        levels = [max(top_z, lo[2] + 0.012)] + list(np.arange(top_z - 0.015, lo[2] + 0.011, -0.015))
        levels.append(lo[2] + 0.009)  # lowest: under a handle or spout
        for rank, z in enumerate(levels):
            found = []
            for k in range(8):
                a = k * math.pi / 8
                dvec = np.array([math.cos(a), math.sin(a)])
                along = np.array([-dvec[1], dvec[0]])
                proj_a = parts[:, :, :2] @ along  # N x 2 (lo, hi corners)
                a_lo, a_hi = proj_a.min(), proj_a.max()
                room = max(0.0, (a_hi - a_lo) / 2 - 0.02)
                mid_a = (a_lo + a_hi) / 2
                for s_a in {mid_a, mid_a + along_slide(rng, p) * room}:
                    g = body_grasp(parts, dvec, along, s_a, z)
                    if g is not None:
                        found.append((np.array([g[0], g[1], z]), dvec))
            if found:
                cands += found
                if rank >= 1 or len(cands) >= 4:
                    break  # the highest heights that work (a lower one only when nothing above does)
    order = rng.permutation(len(cands))
    return [(cands[i][0], cands[i][1], hi[2]) for i in order]


def self_parts(e, inst):
    """World AABBs (N x 2 x 3) of an object's own collision geoms."""
    from lerobot.envs.libero_ext import _object_geoms

    model, data = e.sim.model._model, e.sim.data._data
    geoms, _ = _object_geoms(e, inst)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    out = []
    for g in geoms:
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        pts = data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (c + signs * h).T).T
        out.append([pts.min(0), pts.max(0)])
    return np.array(out).reshape(-1, 2, 3)


def along_slide(rng, p):
    return rng.uniform(-1, 1) * p["grasp_slide"]


PAD_HALF_W, PAD_H, FINGER_H, FINGER_T = 0.012, 0.03, 0.045, 0.012


def body_grasp(parts, d, along, s_a, z_tip):
    """Grasp centre (xy) for fingers closing along d on the line at offset s_a (along `along`), fingertips at
    z_tip: the parts the pads would close on (those meeting the pads' strip and height) must fit the opening, and
    the open fingers, coming down, must miss the object's other parts (a handle, a spout). None if impossible."""
    c3 = (parts[:, 0, :] + parts[:, 1, :]) / 2
    h3 = (parts[:, 1, :] - parts[:, 0, :]) / 2
    pa = c3[:, :2] @ along
    ha = np.abs(h3[:, 0] * along[0]) + np.abs(h3[:, 1] * along[1])
    pd = c3[:, :2] @ d
    hd = np.abs(h3[:, 0] * d[0]) + np.abs(h3[:, 1] * d[1])
    in_strip = np.abs(pa - s_a) <= ha + PAD_HALF_W
    in_pads = in_strip & (parts[:, 1, 2] >= z_tip) & (parts[:, 0, 2] <= z_tip + PAD_H)
    if not in_pads.any():
        return None
    lo_d, hi_d = (pd - hd)[in_pads].min(), (pd + hd)[in_pads].max()
    width = hi_d - lo_d
    if width > MAX_OPEN - 0.005:
        return None
    in_fingers = in_strip & (parts[:, 1, 2] >= z_tip) & (parts[:, 0, 2] <= z_tip + FINGER_H)
    slack = (0.08 - width) / 2 - 0.003
    for shift in sorted(np.arange(-slack, slack + 1e-9, 0.002), key=abs):
        cd = (lo_d + hi_d) / 2 + shift
        clear = True
        for side in (-1, 1):  # each open finger: inner face 4 cm from the centre, FINGER_T thick
            f_lo, f_hi = (cd + 0.04, cd + 0.04 + FINGER_T) if side > 0 else (cd - 0.04 - FINGER_T, cd - 0.04)
            hit = in_fingers & (pd + hd > f_lo) & (pd - hd < f_hi)
            if hit.any():
                clear = False
                break
        if clear:
            return cd * d + s_a * along
    return None


def grasp_spec(e, inst, cls, rng, p, exclude_extra=(), want_dir=None, rot_now=None):
    """A top-down grasp: (point, closing direction, top z, clearance). Among grasps clear by 5 mm or more, the one
    needing the least wrist rotation from rot_now (consistent demos: a random choice among clear grasps gave the
    data several grasp orientations per task, which the policy could not commit to; dataset_gen.md §3c);
    otherwise the one with the most clearance."""
    obs = obstacles(e, {inst, *exclude_extra})
    cands = grasp_candidates(e, inst, cls, rng, p)
    if want_dir is not None:  # e.g. fingers parallel to a fixture's face, for a sideways insertion
        cands = [c for c in cands if abs(np.dot(c[1], want_dir)) > 0.9] or cands
    scored = [(clearance(pt, d, pt[2], obs), pt, d, top) for pt, d, top in cands]
    if not scored:
        return None
    good = [x for x in scored if x[0] >= 0.005]
    if good and rot_now is not None:
        def turn(x):
            r = nearest_equiv(down_rot(x[2]), rot_now)
            return float(np.linalg.norm(tu.quat2axisangle(tu.mat2quat(r @ rot_now.T))))

        good.sort(key=turn)
    c, pt, d, top = good[0] if good else max(scored, key=lambda x: x[0])
    return pt, d, top, c


# ---------------------------------------------------------------- expert
class OutOfTime(Exception):
    """The episode used up its step budget (it must fit the 520-step evaluation limit)."""


class Success(Exception):
    """The task's goal holds: the episode ends here, as an evaluation episode does (LIBERO stops at success)."""


class Expert:
    """Generates one episode's actions for a libero_ext task on a (ControlEnv) env."""

    def __init__(self, env, task, scene_classes, rng, dart_sigma=0.0, max_len=500):
        self.env, self.e, self.task, self.rng = env, env.env, task, rng
        self.max_len = max_len
        self.cls = scene_classes  # object kind -> LIBERO class (from build_results.json)
        self.dart = dart_sigma
        self.p = {
            "speed": rng.uniform(0.8, 1.0),
            "approach": rng.uniform(0.07, 0.13),
            "transport": rng.uniform(0.12, 0.22),
            "release": rng.uniform(0.008, 0.025),
            "retreat": rng.uniform(0.06, 0.12),
            "grasp_slide": rng.uniform(0.0, 0.25),  # was 0.6: a spread grasp point blurred where the policy grasped
            "grasp_depth": rng.uniform(0.025, 0.04),
            "pause": int(rng.integers(0, 4)),
        }
        # the instruction's order (a random order made two modes the policy could not choose between; §3c)
        self.steps = [tuple(s) for s in task["steps"]]
        self.labels, self.executed, self.eef = [], [], []
        self.why = ""  # failure reason, for diagnostics
        self.stop_at_success = True
        self.p_miss = P_MISS  # chance that a pick's (or a drawer pull's) first attempt is a deliberate miss

    # ---- low level
    def obs(self):
        return self.e._get_observations()

    def _act(self, target_pos, target_rot, grip, max_norm):
        o = self.obs()
        pos, rot = o["robot0_eef_pos"], tu.quat2mat(o["robot0_eef_quat"])
        a = np.zeros(7)
        dp = (target_pos - pos) / 0.05
        n = np.linalg.norm(dp)
        if n > max_norm:
            dp *= max_norm / n
        elif 1e-6 < n < 0.25:  # do not creep over the last centimetre
            dp *= 0.25 / n
        a[:3] = dp
        if target_rot is not None:
            aa = tu.quat2axisangle(tu.mat2quat(target_rot @ rot.T))
            a[3:6] = np.clip(aa / 0.5, -0.6, 0.6)
        a[6] = grip
        return np.clip(a, -1, 1), np.linalg.norm(target_pos - pos), (
            np.linalg.norm(tu.quat2axisangle(tu.mat2quat(target_rot @ rot.T))) if target_rot is not None else 0.0)

    def _step(self, label, executed=None):
        """Record label (the expert's action from this state) and send executed: the label plus DART noise by
        default, or a given action (a deliberate miss, whose labels still say what the expert would do)."""
        if len(self.labels) >= self.max_len:
            raise OutOfTime
        if executed is not None:
            a = np.asarray(executed, float).copy()
        else:
            a = label.copy()
            if self.dart > 0:
                a[:6] = np.clip(a[:6] + self.rng.normal(0, self.dart, 6) * np.array([1, 1, 1, 0.5, 0.5, 0.5]), -1, 1)
        self.labels.append(label)
        self.executed.append(a)
        self.eef.append(self.obs()["robot0_eef_pos"].copy())
        self.env.step(a)
        if self.stop_at_success and self.env.check_success():
            raise Success

    def move(self, pos, rot=None, grip=-1.0, tol=0.01, rot_tol=0.08, max_steps=120, speed=None):
        """Servo to a pose; False if not reached within max_steps, or if blocked (no progress for 15 steps)."""
        speed = self.p["speed"] if speed is None else speed
        errs = []
        for _ in range(max_steps):
            a, dpos, drot = self._act(pos, rot, grip, speed)
            if dpos < tol and drot < rot_tol:
                return True
            errs.append(dpos + 0.05 * drot)
            if len(errs) > 15 and errs[-16] - errs[-1] < 0.002:
                return False  # blocked: something is in the way
            self._step(a)
        return False

    def move_dual(self, true_pos, exec_pos, rot, grip_label, grip_exec, tol=0.008, max_steps=80, speed=None):
        """Send the arm to exec_pos while recording, at every step, the expert's action towards true_pos (with
        grip_label): DART-style recovery data, where the executed motion goes wrong and the labels say what the
        expert would do from each state actually reached. Ends when exec_pos is reached or the arm is blocked."""
        speed = min(self.p["speed"], 0.6) if speed is None else speed
        errs = []
        for _ in range(max_steps):
            a_exec, dpos, _ = self._act(exec_pos, rot, grip_exec, speed)
            if dpos < tol:
                return True
            errs.append(dpos)
            if len(errs) > 15 and errs[-16] - errs[-1] < 0.002:
                return False
            a_label, _, _ = self._act(true_pos, rot, grip_label, speed)
            self._step(a_label, executed=a_exec)
        return False

    def missed_grasp(self, true_pt, miss_pt, rot, n_close=7):
        """A deliberate miss: go to miss_pt instead of true_pt and close there, then open again. Every label is the
        expert's action from the state reached: on to true_pt with the gripper open."""
        self.move_dual(true_pt, miss_pt, rot, -1.0, -1.0)
        for _ in range(n_close):
            a_label, _, _ = self._act(true_pt, rot, -1.0, min(self.p["speed"], 0.6))
            a_exec, _, _ = self._act(miss_pt, rot, 1.0, min(self.p["speed"], 0.6))
            self._step(a_label, executed=a_exec)
        self.hold(6, -1.0)  # the expert's own recovery from here on: open (labels = executed)

    def hold(self, n, grip):
        o = self.obs()
        pos, rot = o["robot0_eef_pos"].copy(), tu.quat2mat(o["robot0_eef_quat"])
        for _ in range(n):
            a, _, _ = self._act(pos, rot, grip, self.p["speed"])
            self._step(a)

    def eef_rot(self):
        return tu.quat2mat(self.obs()["robot0_eef_quat"])

    def grasped(self, inst):
        obj = self.e.get_object(inst)
        return bool(self.e._check_grasp(gripper=self.e.robots[0].gripper, object_geoms=obj))

    # ---- primitives
    def pick(self, kind, tries=3, want_dir=None):
        inst = f"{self.cls[kind]}_1"
        for try_ in range(tries):
            low0 = obj_box(self.e, inst)[0][2]
            spec = grasp_spec(self.e, inst, self.cls[kind], self.rng, self.p, want_dir=want_dir,
                              rot_now=self.eef_rot())
            if spec is None:
                return None
            pt, d, top, _ = spec
            rot = nearest_equiv(down_rot(d), self.eef_rot())
            above = np.array([pt[0], pt[1], top + self.p["approach"]])
            self.move(above, rot, -1.0, tol=0.015)
            self.hold(self.p["pause"], -1.0)
            if try_ == 0 and self.rng.random() < self.p_miss:
                # recovery data (dataset_gen.md §3c): close too high, or beside the object along the fingers'
                # width, then open, rise 4 cm and grasp properly
                lo, hi = obj_box(self.e, inst)
                along = np.array([-d[1], d[0]])
                if self.rng.random() < 0.5:
                    miss = pt + np.array([0.0, 0.0, self.rng.uniform(0.015, 0.025)])
                else:
                    half = (abs(along[0]) * (hi[0] - lo[0]) + abs(along[1]) * (hi[1] - lo[1])) / 2
                    off = min(half + self.rng.uniform(0.01, 0.02), 0.04) * self.rng.choice([-1.0, 1.0])
                    miss = pt + np.append(along * off, 0.0)
                self.missed_grasp(pt, miss, rot)
                here = self.obs()["robot0_eef_pos"]
                self.move(here + np.array([0, 0, 0.04]), rot, -1.0, tol=0.012)
                self.move(np.array([pt[0], pt[1], pt[2] + 0.04]), rot, -1.0, tol=0.012)
            self.move(pt, rot, -1.0, tol=0.006, speed=min(self.p["speed"], 0.6))
            self.hold(7, 1.0)
            lift = np.array([pt[0], pt[1], top + self.p["approach"]])
            self.move(lift, rot, 1.0, tol=0.015, speed=min(self.p["speed"], 0.6))
            fingers = self.obs()["robot0_gripper_qpos"]
            # held: lifted with the hand (its lowest point up 2 cm, so not just tipped), fingers not shut on air
            # (a rim grasp closes on a wall only a few mm thick)
            if obj_box(self.e, inst)[0][2] - low0 > 0.02 and abs(fingers[0] - fingers[1]) > 0.002:
                return inst
            self.hold(6, -1.0)  # missed: open and try again
        return None

    def carry_and_release(self, inst, drop_xy, surface_z, release_extra=0.0, drop=False):
        """Carry the held object above drop_xy and lower it until its bottom is just above surface_z."""
        o = self.obs()
        eef = o["robot0_eef_pos"]
        rot = self.eef_rot()
        hang = eef[2] - obj_box(self.e, inst)[0][2]  # eef height above the object's lowest point
        sz = sl.surface_z_of(self.e)
        # clear everything along the way: obstacles whose box meets the corridor from here to the drop point
        start, end = eef[:2], np.asarray(drop_xy, float)
        tallest = sz
        for lo, hi in obstacles(self.e, {inst}):  # noqa: B007
            for t in np.linspace(0, 1, 12):
                q = start + t * (end - start)
                if lo[0] - 0.06 <= q[0] <= hi[0] + 0.06 and lo[1] - 0.06 <= q[1] <= hi[1] + 0.06:
                    tallest = max(tallest, hi[2])
                    break
        z_tr = max(eef[2], surface_z + hang + 0.06, sz + hang + self.p["transport"], tallest + hang + 0.06)
        self.move(np.array([eef[0], eef[1], z_tr]), rot, 1.0, tol=0.02)
        # at the drop point: a hand orientation and the lowest release height with the hand body clear
        release_z = surface_z + hang + self.p["release"] + release_extra
        obs = obstacles(self.e, {inst})
        fwd = rot[:, 1][:2]  # current closing direction
        best = None
        for k in range(8):
            a = math.atan2(fwd[1], fwd[0]) + k * math.pi / 8
            dvec = np.array([math.cos(a), math.sin(a)])
            turn = min(k, 8 - k)
            for dz in np.arange(0.0, 0.12, 0.01):
                if clearance(drop_xy, dvec, release_z + dz, obs, fingers=False) >= 0.004:
                    key = (dz, turn + self.rng.uniform(0, 0.5))
                    if best is None or key < best[0]:
                        best = (key, dvec, release_z + dz)
                    break
        rot_drop = rot if best is None else nearest_equiv(down_rot(best[1]), rot)
        release_z = release_z if best is None else best[2]
        # aim the object (its body origin, which the goal checks use), not the hand: the held object sits off
        # the hand's axis (a rim grasp by ~5 cm), and that offset turns with any change of wrist yaw
        here = self.obs()["robot0_eef_pos"]
        off = here[:2] - sl.body_pos(self.e, inst)[:2]
        turn = math.atan2(rot_drop[1, 0], rot_drop[0, 0]) - math.atan2(rot[1, 0], rot[0, 0])
        c, s_ = math.cos(turn), math.sin(turn)
        drop_xy = np.asarray(drop_xy, float) + np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
        z_hi = max(z_tr, release_z)
        for _ in range(2):  # if the drop point is not reached (caught on something), lift and try again
            if self.move(np.array([drop_xy[0], drop_xy[1], z_hi]), rot_drop, 1.0, tol=0.015):
                break
            z_hi += 0.05
            here = self.obs()["robot0_eef_pos"]
            self.move(np.array([here[0], here[1], z_hi]), rot_drop, 1.0, tol=0.02)
        self.move(np.array([drop_xy[0], drop_xy[1], release_z]), rot_drop, 1.0, tol=0.008,
                  speed=min(self.p["speed"], 0.6))
        if np.linalg.norm(self.obs()["robot0_eef_pos"][:2] - np.asarray(drop_xy)) > 0.03:
            self.why = "could not reach the drop point"
            raise OutOfTime  # abandon: releasing here would drop the object anywhere
        self.hold(6, -1.0)
        up = self.obs()["robot0_eef_pos"] + np.array([0, 0, self.p["retreat"]])
        self.move(up, rot_drop, -1.0, tol=0.02)

    def destination(self, inst, dest, phrase):
        """(drop xy, surface z, extra release height) for a placing step's destination."""
        goal = [g for g in self.e.parsed_problem["goal_state"] if len(g) == 3 and g[1] == inst][0]
        tgt = goal[2]
        if tgt in self.e.objects_dict:  # on an object: plate, bowl, pan
            lo, hi = obj_box(self.e, tgt)
            c = sl.body_pos(self.e, tgt)
            return c[:2], hi[2], (0.02 if "bowl" in tgt else 0.0)
        c, lo, hi = sl._site_box(self.e, tgt)
        if tgt.startswith(self.e.workspace_name) or tgt.startswith("floor"):
            jitter = self.rng.uniform(-0.015, 0.015, 2)
            return c[:2] + jitter, sl.surface_z_of(self.e), 0.0
        if goal[0] == "in" and "_level" not in tgt and dest == "drawer":
            # an open drawer: drop over its pulled-out part (the end of the region nearer the table centre)
            y = hi[1] - 0.045 if c[1] < 0 else lo[1] + 0.045
            return np.array([c[0], y]), lo[2], 0.0
        if goal[0] == "in":
            return c[:2], lo[2], 0.0
        # on a fixture's top: towards the table centre so it stays in view
        xy = c[:2].copy()
        half_y = (hi[1] - lo[1]) / 2
        if abs(xy[1]) > 0.2:
            xy[1] -= np.sign(xy[1]) * half_y * 0.4
        return xy, hi[2], 0.0

    def insert(self, inst, dest, phrase):
        """Sideways insertion through a fixture's front opening (microwave, fridge, bookcase, lower shelf): turn
        the hand to point horizontally into the opening, line the object up with the region, slide it in, release,
        back out."""
        goal = [g for g in self.e.parsed_problem["goal_state"] if len(g) == 3 and g[1] == inst][0]
        c, lo, hi = sl._site_box(self.e, goal[2])
        sgn = 1.0 if c[1] > 0 else -1.0  # the fixture's side of the table; its opening faces -sgn*y
        edge = lo[1] if sgn > 0 else hi[1]  # the region's face towards the table centre
        rot0 = self.eef_rot()
        # 1. lift clear and turn the hand: approach axis into the fixture, fingers still closing along x
        here = self.obs()["robot0_eef_pos"]
        mid = np.array([here[0], here[1] - sgn * 0.05, max(here[2], c[2] + 0.12)])
        self.move(mid, rot0, 1.0, tol=0.02)
        # first turn the wrist (pointing down) so the fingers close along x, then pitch it towards the opening:
        # done in one go from a y-closing grasp, the twist shakes the object loose
        rot_x = nearest_equiv(down_rot(np.array([1.0, 0.0])), rot0)
        self.move(mid, rot_x, 1.0, tol=0.02, rot_tol=0.08, max_steps=120)
        y_g = np.array([np.sign(rot_x[0, 1]) or 1.0, 0.0, 0.0])
        z_g = np.array([0.0, sgn, 0.0])
        rot_in = np.stack([np.cross(y_g, z_g), y_g, z_g], axis=1)
        self.move(mid, rot_in, 1.0, tol=0.02, rot_tol=0.1, max_steps=160)
        # 2. the object's centre in front of the opening at the region's height, then slide in
        lo_o, hi_o = obj_box(self.e, inst)
        off = self.obs()["robot0_eef_pos"] - (lo_o + hi_o) / 2
        half_depth = abs(hi_o[1] - lo_o[1]) / 2
        pre = np.array([c[0], edge - sgn * (half_depth + 0.08), c[2]]) + off
        self.move(pre, rot_in, 1.0, tol=0.012, rot_tol=0.1)
        final = np.array([c[0], c[1], c[2]]) + off
        self.move(final, rot_in, 1.0, tol=0.01, rot_tol=0.12, speed=min(self.p["speed"], 0.5))
        self.hold(6, -1.0)
        # 3. back out, turn back to pointing down
        back = self.obs()["robot0_eef_pos"] - np.array([0.0, sgn * 0.16, 0.0])
        self.move(back, rot_in, -1.0, tol=0.02, rot_tol=0.15)
        self.move(back + np.array([0, 0, 0.06]), rot0, -1.0, tol=0.03, rot_tol=0.15)

    # ---- open / close / turn
    def _joint(self, fixture):
        """(joint name, world anchor, world axis, moving body id) of a fixture's articulation."""
        e = self.e
        m, d = e.sim.model._model, e.sim.data._data
        if fixture == "drawer":
            site = [g[1] for g in e.parsed_problem["goal_state"] if g[0] in ("open", "close")][0]
            jn = e.object_sites_dict[site].joints[0]
        else:
            jn = e.get_object(f"{self.cls[fixture]}_1").joints[0]
        jid = e.sim.model.joint_name2id(jn)
        b = m.jnt_bodyid[jid]
        axis = d.xmat[b].reshape(3, 3) @ m.jnt_axis[jid]
        return jn, d.xanchor[jid].copy(), axis, b

    def _qpos(self, jn):
        m = self.e.sim.model._model
        return float(self.e.sim.data._data.qpos[m.jnt_qposadr[self.e.sim.model.joint_name2id(jn)]])

    def _part(self, body, pick="largest"):
        """Oriented box of one collision geom of a moving body (the largest, or the one reaching highest):
        (centre, unit axes as columns, half sizes, world AABB lo, world AABB hi)."""
        m, d = self.e.sim.model._model, self.e.sim.data._data
        best = None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
            h = m.geom_aabb[g][3:]
            ext = np.abs(R) @ h
            key = float(np.prod(h)) if pick == "largest" else float(c[2] + ext[2])
            if best is None or key > best[0]:
                best = (key, c, R, h, c - ext, c + ext)
        return best[1:]

    def _all_boxes(self, body):
        """World AABB (lo, hi) of all of a body's collision geoms."""
        m, d = self.e.sim.model._model, self.e.sim.data._data
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                R = d.geom_xmat[g].reshape(3, 3)
                c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
                ext = np.abs(R) @ m.geom_aabb[g][3:]
                pts += [c - ext, c + ext]
        return np.min(pts, axis=0), np.max(pts, axis=0)

    def swing(self, fixture, done, increase, frac, mode="push", lead=0.03, max_steps=260, z_off=0.0):
        """Turn a part hinged about a vertical axis (a door, the faucet lever) by moving the closed gripper with it,
        closed loop: every step the gripper aims `lead` ahead, along the part's motion, of its contact point on the
        part, wherever the part has actually got to. mode "push": the fingertips press on the face that trails the
        motion; "drag": they press down on the part's top and pull it round by friction. frac: the contact
        point's distance from the hinge, as a fraction of the part's length. Returns done(q) at the end."""
        jn, anchor, axis, body = self._joint(fixture)
        s_ax = float(np.sign(axis[2]))
        w = s_ax * (1.0 if increase else -1.0)  # +1: the part turns anticlockwise seen from above
        if mode == "drag":  # a closed door: the box of all its pieces (aligned with the fixture)
            lo, hi = self._all_boxes(body)
            c, R, h = (lo + hi) / 2, np.eye(3), (hi - lo) / 2
        else:
            c, R, h, lo, hi = self._part(body)
        # the part's long horizontal axis (pointing away from the hinge) and its thin horizontal axis
        horiz = [i for i in range(3) if abs(R[2, i]) < 0.7]
        i_long = max(horiz, key=lambda i: h[i])
        i_thin = min(horiz, key=lambda i: h[i])
        a = R[:, i_long][:2] / np.linalg.norm(R[:, i_long][:2])
        if np.dot(c[:2] - anchor[:2], a) < 0:
            a = -a
        s_c = float(np.dot(c[:2] - anchor[:2], a))
        s = frac * (s_c + h[i_long])
        p_line = c[:2] + a * (s - s_c)  # on the part's centre line, distance s from the hinge
        r = p_line - anchor[:2]
        t = w * np.array([-r[1], r[0]]) / np.linalg.norm(r)  # direction of motion there
        if mode == "push":
            contact = p_line - t * (h[i_thin] + 0.016)
            # fingertips at mid-height, but no lower than 4 cm under the part's top: the hand body (5 cm above the
            # fingertips) must pass over a door's top edge
            z = max((lo[2] + hi[2]) / 2, hi[2] - 0.04) + z_off
        else:  # on the top, towards its outer face (the fixture's own top is behind the door)
            contact = p_line + t * h[i_thin] * 0.4
            z = hi[2] - 0.012 + z_off
        r0 = contact - anchor[:2]
        q_ref = self._qpos(jn)
        rot = self.eef_rot()
        self.move(np.array([contact[0], contact[1], hi[2] + 0.07]), rot, 1.0, tol=0.015)
        self.move(np.array([contact[0], contact[1], z]), rot, 1.0, tol=0.01, speed=0.5, max_steps=60)
        hist, held = [], 0
        for _ in range(max_steps):
            q = self._qpos(jn)
            held = held + 1 if done(q) else 0
            if held >= 6:  # keep pressing until it stays there: a door that slams shut bounces back open
                break
            hist.append(q)
            if len(hist) > 50 and abs(hist[-1] - hist[-50]) < 0.02:
                break  # stalled
            ang = s_ax * (q - q_ref)
            cs, sn = math.cos(ang), math.sin(ang)
            rr = np.array([cs * r0[0] - sn * r0[1], sn * r0[0] + cs * r0[1]])
            tt = w * np.array([-rr[1], rr[0]]) / np.linalg.norm(rr)
            tgt = np.array([*(anchor[:2] + rr + tt * lead), z])
            act, _, _ = self._act(tgt, rot, 1.0, min(self.p["speed"], 0.6))
            self._step(act)
        ok = done(self._qpos(jn))
        here = self.obs()["robot0_eef_pos"]
        if mode == "push":  # step back off the face before lifting
            ang = s_ax * (self._qpos(jn) - q_ref)
            cs, sn = math.cos(ang), math.sin(ang)
            rr = np.array([cs * r0[0] - sn * r0[1], sn * r0[0] + cs * r0[1]])
            tt = w * np.array([-rr[1], rr[0]]) / np.linalg.norm(rr)
            here = here - np.append(tt * 0.03, 0.0)
            self.move(here, rot, 1.0, tol=0.015, max_steps=30)
        self.move(here + np.array([0, 0, 0.08]), rot, 1.0, tol=0.02, max_steps=60)
        return ok

    def twist_knob(self, fixture, done, increase=True):
        """Stove knob: grip its raised ridge from above and turn the wrist."""
        jn, anchor, axis, body = self._joint(fixture)
        c, R, h, lo, hi = self._part(body, pick="highest")
        horiz = [i for i in range(3) if abs(R[2, i]) < 0.7]
        i_long = max(horiz, key=lambda i: h[i])
        a = R[:, i_long][:2] / np.linalg.norm(R[:, i_long][:2])
        w = float(np.sign(axis[2])) * (1.0 if increase else -1.0)
        turn = 1.15 * w
        now = self.eef_rot()
        rz = lambda t: np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1.0]])  # noqa: E731
        base = down_rot(np.array([-a[1], a[0]]))
        # of the two equivalent grips, the one whose turn passes nearest the current wrist angle (joint limits)
        g = min((base, base @ np.diag([-1.0, -1.0, 1.0])),
                key=lambda r: np.linalg.norm(tu.quat2axisangle(tu.mat2quat(rz(turn / 2) @ r @ now.T))))
        centre = np.array([anchor[0], anchor[1], hi[2] - 0.016])
        self.move(centre + np.array([0, 0, 0.08]), g, -1.0, tol=0.012, rot_tol=0.08)
        self.move(centre, g, -1.0, tol=0.006, speed=0.4, max_steps=60)
        self.hold(8, 1.0)
        for _ in range(80):
            if done(self._qpos(jn)):
                break
            act, _, _ = self._act(centre, rz(turn) @ g, 1.0, 0.5)
            self._step(act)
        ok = done(self._qpos(jn))
        self.hold(5, -1.0)
        here = self.obs()["robot0_eef_pos"]
        self.move(here + np.array([0, 0, 0.08]), self.eef_rot(), -1.0, tol=0.02, max_steps=60)
        return ok

    def _handle(self, body, out):
        """Centre of a drawer's handle: the part of the drawer standing out furthest along the opening direction."""
        m, d = self.e.sim.model._model, self.e.sim.data._data
        boxes = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                R = d.geom_xmat[g].reshape(3, 3)
                c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
                ext = np.abs(R) @ m.geom_aabb[g][3:]
                boxes.append((c - ext, c + ext))
        hb = max(boxes, key=lambda b: float(np.dot(out, ((b[0] + b[1]) / 2)[:2])))
        return (hb[0] + hb[1]) / 2

    def open_drawer(self):
        """Grip the drawer handle with the hand horizontal (fingers above and below the bar) and pull it out; if it
        stops short of open (the grip slipped), grip it again where it now is and pull again."""
        jn, anchor, axis, body = self._joint("drawer")
        out = -axis[:2] / np.linalg.norm(axis[:2])  # opening = negative joint travel along the axis
        site = [g[1] for g in self.e.parsed_problem["goal_state"] if g[0] == "open"][0]
        is_open = lambda: bool(self.e._eval_predicate(["open", site]))  # noqa: E731
        z_g = np.array([-out[0], -out[1], 0.0])  # approach axis: into the cabinet
        y_g = np.array([0.0, 0.0, 1.0])  # fingers close vertically
        rot_h = nearest_equiv(np.stack([np.cross(y_g, z_g), y_g, z_g], axis=1), self.eef_rot())
        h = self._handle(body, out)
        # turn the hand horizontal first, clear of the cabinet (turning while moving in can jam the arm)
        clear = h + np.append(out * 0.16, 0.06)
        self.move(clear, self.eef_rot(), -1.0, tol=0.02)
        self.move(clear, rot_h, -1.0, tol=0.02, rot_tol=0.06, max_steps=120)
        for attempt in range(2):
            h = self._handle(body, out)
            pre = h + np.append(out * 0.12, 0.0)
            grasp = h + np.append(out * 0.005, 0.0)
            for k in range(2):
                self.move(pre, rot_h, -1.0, tol=0.01, rot_tol=0.05, max_steps=160)
                if attempt == 0 and k == 0 and self.rng.random() < self.p_miss:
                    # recovery data: close 3-5 cm short of the handle, open, then grasp it (labels: on to the handle)
                    short = grasp + np.append(out * self.rng.uniform(0.03, 0.05), 0.0)
                    self.missed_grasp(grasp, short, rot_h)
                self.move(grasp, rot_h, -1.0, tol=0.006, rot_tol=0.06, speed=0.4)
                self.hold(8, 1.0)
                f = self.obs()["robot0_gripper_qpos"]
                if abs(f[0] - f[1]) > 0.01:  # something (the handle) between the fingers
                    break
                self.hold(6, -1.0)
                self.move(pre, rot_h, -1.0, tol=0.015)
            self.move(grasp + np.append(out * 0.20, 0.0), rot_h, 1.0, tol=0.012, speed=0.4, max_steps=100)
            self.hold(5, -1.0)
            if is_open():
                break
            here = self.obs()["robot0_eef_pos"]
            self.move(here + np.append(out * 0.04, 0.0), rot_h, -1.0, tol=0.02)
        here = self.obs()["robot0_eef_pos"]
        self.move(here + np.append(out * 0.06, 0.0), rot_h, -1.0, tol=0.02)
        self.move(here + np.append(out * 0.06, 0.10), down_rot(np.array([1.0, 0.0])), -1.0, tol=0.03, rot_tol=0.2)
        return is_open()

    def _fixture_state(self, fixture, pred):
        """done(q) for a goal predicate (open / close / turnon) of a fixture, from the fixture class's own ranges."""
        obj = self.e.get_object(f"{self.cls[fixture]}_1")
        fn = getattr(obj, {"open": "is_open", "close": "is_close", "turnon": "turn_on"}[pred])
        return lambda q: bool(fn(q))

    def run_artic(self):
        """The open/close/turn step of an articulation task, in the order its goal needs."""
        art = self.task["artic"]
        place = self.steps[0] if self.steps else None
        if art == "faucet":
            ok = self.swing("faucet", lambda q: abs(q) > 0.8, increase=True,  # one way, not either (§3c)
                            frac=self.rng.uniform(0.5, 0.65), mode="push")
            if not ok:
                self.why = "faucet not turned"
            return ok
        if art == "stove":
            on = self._fixture_state("stove", "turnon")
            first = True  # the instruction's order: turn on the stove, then the placing step (§3c)
            if first and not self.twist_knob("stove", on):
                self.why = "stove knob not turned"
                return False
            if not self.run_step(*place):
                return False
            if not first and not self.twist_knob("stove", on):
                self.why = "stove knob not turned"
                return False
            return True
        if art == "drawer":
            if not self.open_drawer():
                self.why = "drawer not opened"
                return False
            return self.run_step(*place)
        if art == "fridge":
            if not self.swing("fridge", lambda q: q > 2.2, increase=True, frac=self.rng.uniform(0.7, 0.85), mode="drag"):
                self.why = "fridge not opened"
                return False
            return self.run_step(*place)
        if art == "microwave":
            if not self.run_step(*place):
                return False
            shut = self._fixture_state("microwave", "close")
            ok = self.swing("microwave", shut, increase=True, frac=self.rng.uniform(0.5, 0.65), mode="push")
            if not ok:
                self.why = "microwave not closed"
            return ok
        return False

    def run_step(self, kind, dest, phrase):
        if dest in ("microwave", "fridge", "bookcase") or (dest == "shelf" and "lower" in phrase):
            inst = self.pick(kind, want_dir=np.array([1.0, 0.0]))
            if inst is None:
                self.why = f"pick {kind}"
                return False
            self.insert(inst, dest, phrase)
            goal = [g for g in self.e.parsed_problem["goal_state"] if len(g) == 3 and g[1] == inst]
            if goal and not self.e._eval_predicate(goal[0]):
                lo = obj_box(self.e, inst)[0][2] - sl.surface_z_of(self.e)
                self.why = f"inserted {kind} {dest}: not in the region (bottom {lo * 100:.0f} cm above the table)"
            return True
        inst = self.pick(kind)
        if inst is None:
            self.why = f"pick {kind}"
            return False
        if dest == "lift":
            sz = sl.surface_z_of(self.e)
            eef = self.obs()["robot0_eef_pos"]
            hang = eef[2] - obj_box(self.e, inst)[0][2]
            self.move(np.array([eef[0], eef[1], sz + hang + 0.13]), self.eef_rot(), 1.0, tol=0.015)
            self.hold(12, 1.0)
            return True
        xy, z, extra = self.destination(inst, dest, phrase)
        self.carry_and_release(inst, xy, z, extra)
        goal = [g for g in self.e.parsed_problem["goal_state"] if len(g) == 3 and g[1] == inst]
        if goal and not self.e._eval_predicate(goal[0]):
            p = sl.body_pos(self.e, inst)
            off = p[:2] - np.asarray(xy)
            low = obj_box(self.e, inst)[0][2] - sl.surface_z_of(self.e)
            self.why = (f"placed {kind} {dest}: {np.linalg.norm(off) * 100:.0f} cm from the drop point, "
                        f"bottom {low * 100:.0f} cm above the table")
        return True

    def run(self):
        """Run the episode until the goal first holds (success) or the plan ends; returns (success, n_steps)."""
        try:
            if self.task.get("artic"):
                self.run_artic()
            else:
                for kind, dest, phrase in self.steps:
                    if not self.run_step(kind, dest, phrase):
                        break
            ok = bool(self.env.check_success())
        except Success:
            ok = True
        except OutOfTime:
            ok = False
            self.why = self.why or "out of time"
        if not ok and not self.why:
            self.why = "goal false at the end"
        return ok, len(self.labels)


def widen_start(env, rng, shift=0.02, joint_noise=0.03):
    """Nudge movable objects (round ones also turned) and jitter the arm, then settle; True if still valid."""
    e = env.env
    for name, obj in e.objects_dict.items():
        cls = name[:-2]
        jn = obj.joints[-1]
        q = e.sim.data.get_joint_qpos(jn).copy()
        q[:2] += rng.uniform(-shift, shift, 2)
        q[2] += 0.003
        if cls in ROUND:
            turn = tu.axisangle2quat(np.array([0, 0, rng.uniform(-math.pi, math.pi)]))
            qw = np.array([q[4], q[5], q[6], q[3]])  # mujoco wxyz -> xyzw
            qn = tu.quat_multiply(turn, qw)
            q[3:7] = [qn[3], qn[0], qn[1], qn[2]]
        e.sim.data.set_joint_qpos(jn, q)
        e.sim.data.set_joint_qvel(jn, np.zeros(6))
    rob = e.robots[0]
    qpos = e.sim.data.qpos[rob._ref_joint_pos_indexes] + rng.uniform(-joint_noise, joint_noise, len(rob._ref_joint_pos_indexes))
    e.sim.data.qpos[rob._ref_joint_pos_indexes] = qpos
    e.sim.forward()
    rob.controller.update_initial_joints(qpos)
    rob.controller.reset_goal()
    from lerobot.envs.libero_ext import lowest_point, surface_z

    p0 = {n: sl.body_pos(e, n) for n in e.objects_dict}
    sl.settle(env, 30)
    drift = max(np.linalg.norm(sl.body_pos(e, n) - p0[n]) for n in p0)
    low = min(lowest_point(e, n) - surface_z(e) for n in p0)
    return drift < 0.03 and low > -0.05 and not env.check_success()
