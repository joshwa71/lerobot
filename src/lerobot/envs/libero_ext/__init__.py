"""The ``libero_ext`` LIBERO suite: new tasks built from LIBERO's own assets (projects/dataset_gen.md).

``register()`` (called when ``lerobot.envs.libero`` is imported, so in every process that builds a LIBERO env)
makes three additions to hf-libero at runtime, without changing the package:

- a ``lifted`` goal predicate: the object's lowest point is at least ``LIFT_CLEARANCE`` above the work surface
  and it touches nothing but the robot (LIBERO's own ``up`` predicate uses an absolute height of 1.0 m, which
  only suits the 0.9 m tables and is already true for a tall bottle standing on one);
- on/off checks for LIBERO's faucet fixture, whose class has none (on: lever turned past ~35 degrees either way);
- the ``libero_ext`` benchmark suite, read from ``suite.json`` beside this file. BDDL and init-state files live
  in ``files/``; each task's ``problem_folder`` is that absolute directory, so LIBERO's and lerobot's
  ``<root>/<problem_folder>/<file>`` path joins resolve to it unchanged.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"
SUITE_JSON = HERE / "suite.json"
SUITE_NAME = "libero_ext"
LIFT_CLEARANCE = 0.06  # m between the object's lowest point and the work surface
FAUCET_ON, FAUCET_OFF = 0.6, 0.1  # |lever angle| in rad

_registered = False
_geom_cache: OrderedDict = OrderedDict()  # id(model) -> (model, {object name: (geoms, bodies)})
_GEOM_CACHE_MODELS = 4


def _object_geoms(env, name: str) -> tuple[np.ndarray, set]:
    """Geoms of an object's body subtree (cached per model), and the set of its body ids.

    The cache holds a reference to each model it has entries for, so a model's id cannot be reused by another
    model while its entries exist (keyed by id alone, a closed env's entries were served to the next env's model,
    which then read another scene's geom ids). Only the last few models are kept."""
    model = env.sim.model._model
    entry = _geom_cache.get(id(model))
    if entry is None or entry[0] is not model:
        entry = (model, {})
        _geom_cache[id(model)] = entry
        while len(_geom_cache) > _GEOM_CACHE_MODELS:
            _geom_cache.popitem(last=False)
    _geom_cache.move_to_end(id(model))
    per_model = entry[1]
    if name not in per_model:
        root = env.obj_body_id[name]
        bodies = set()
        for b in range(model.nbody):
            p = b
            while p > 0 and p != root:
                p = model.body_parentid[p]
            if p == root:
                bodies.add(b)
        geoms = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies], dtype=int)
        per_model[name] = (geoms, bodies)
    return per_model[name]


def lowest_point(env, name: str) -> float:
    """World z of the lowest corner of the object's geom bounding boxes."""
    model, data = env.sim.model._model, env.sim.data._data
    geoms, _ = _object_geoms(env, name)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    low = np.inf
    for g in geoms:
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        corners = data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (c + signs * h).T).T
        low = min(low, corners[:, 2].min())
    return float(low)


# Measured resting height of objects above each arena's nominal workspace z (its table top is not at the offset).
_SURFACE_ABOVE_OFFSET = {
    "libero_study_tabletop_manipulation": 0.016,
    "libero_living_room_tabletop_manipulation": 0.024,
    "libero_coffee_table_manipulation": 0.012,
}


def surface_z(env) -> float:
    """Height of the work surface: the table top, or the floor (z = 0) in the floor arena."""
    z = float(env.workspace_offset[2]) + _SURFACE_ABOVE_OFFSET.get(type(env).__name__.lower(), 0.0)
    return max(z, 0.0)


def touches_only_robot(env, name: str) -> bool:
    model, data = env.sim.model._model, env.sim.data._data
    geoms, _ = _object_geoms(env, name)
    gset = set(geoms.tolist())
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in gset or c.geom2 in gset:
            other = c.geom2 if c.geom1 in gset else c.geom1
            body = model.body(model.geom_bodyid[other]).name
            if not (body.startswith("robot") or body.startswith("gripper")):
                return False
    return True


def _make_lifted():
    from libero.libero.envs.predicates.base_predicates import UnaryAtomic

    class Lifted(UnaryAtomic):
        def __call__(self, arg):
            env = arg.env
            name = arg.object_name
            return (lowest_point(env, name) >= surface_z(env) + LIFT_CLEARANCE) and touches_only_robot(
                env, name
            )

    return Lifted()


def _patch_faucet():
    from libero.libero.envs.objects.articulated_objects import Faucet

    if getattr(Faucet, "_libero_ext_patched", False):
        return
    original_init = Faucet.__init__

    def init_with_ranges(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        art = self.object_properties["articulation"]
        art["default_turnon_ranges"] = [0.8, 1.2]
        art["default_turnoff_ranges"] = [-0.02, 0.02]

    Faucet.__init__ = init_with_ranges
    Faucet.turn_on = lambda self, qpos: abs(qpos) > FAUCET_ON
    Faucet.turn_off = lambda self, qpos: abs(qpos) < FAUCET_OFF
    Faucet.is_open = lambda self, qpos: abs(qpos) > FAUCET_ON
    Faucet.is_close = lambda self, qpos: abs(qpos) < FAUCET_OFF
    Faucet._libero_ext_patched = True


def register_runtime() -> None:
    """Predicate and faucet additions only (enough to build and check libero_ext BDDL files)."""
    from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT

    VALIDATE_PREDICATE_FN_DICT.setdefault("lifted", _make_lifted())
    _patch_faucet()


def register() -> None:
    """Runtime additions plus the libero_ext suite (if suite.json exists). Idempotent."""
    global _registered
    if _registered:
        return
    register_runtime()
    if SUITE_JSON.exists():
        from libero.libero import benchmark

        spec = json.loads(SUITE_JSON.read_text())
        benchmark.task_maps[SUITE_NAME] = {
            t["name"]: benchmark.Task(
                name=t["name"],
                language=t["language"],
                problem="Libero",
                problem_folder=str(FILES),
                bddl_file=t["bddl_file"],
                init_states_file=t["init_file"],
            )
            for t in spec["tasks"]
        }

        class LIBERO_EXT(benchmark.Benchmark):  # noqa: N801 (LIBERO's naming; registered as "libero_ext")
            def __init__(self, task_order_index: int = 0):
                super().__init__(task_order_index=task_order_index)
                self.name = SUITE_NAME
                # Suite order as in suite.json; LIBERO's 10-task permutations do not apply.
                self.tasks = list(benchmark.task_maps[SUITE_NAME].values())
                self.n_tasks = len(self.tasks)

        benchmark.register_benchmark(LIBERO_EXT)
    _registered = True
