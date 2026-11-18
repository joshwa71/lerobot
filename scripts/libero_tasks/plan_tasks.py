#!/usr/bin/env python
"""Draft the 100-task LIBERO extension list (projects/dataset_gen.md, §2) and check it against the rules.

A task is one or two steps; a step is (object, destination). Open/close/turn steps use the fixture as both.
Rules (dataset_gen.md, "Agreed rule"):
  - no two tasks share both an object and a destination (hard). For a two-step task this covers every pairing
    of its objects with its destinations, not just its own steps;
  - instructions sharing more than one content token (a multi-word name such as "cream cheese" is one token)
    are reported and kept few, but not forbidden: forbidding them caps the set near 55 tasks;
  - destinations and objects are reused as evenly and as little as possible;
  - no step already used by LIBERO-90 or LIBERO-10 (except the fixed open/close/turn steps, which are);
  - a scene's tasks never repeat a destination or a moved object, and a scene holds at most 3 large items
    (fixtures, containers) and 7 small ones.

Method: randomised greedy construction (a task is added only if it shares no object-destination pairing with
the tasks so far; least-used objects and destinations are sampled first), restarted many times, keeping the
build with the most even reuse and fewest token overlaps; then tasks are grouped into scenes by annealing over
swaps. Size and mix: 60 tasks, 30 single-step and 30 two-step (dataset_gen.md, decision of 18 Nov 26).

  python scripts/libero_tasks/plan_tasks.py [--restarts N] [--seed S] [--out projects/dataset_gen_tasks.md]
"""

import argparse
import itertools
import math
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- vocabulary
OBJECTS = {  # kind used in instructions -> LIBERO object classes (each scene picks one)
    "mug": ["porcelain_mug", "red_coffee_mug", "white_yellow_mug"],
    "bowl": ["akita_black_bowl", "white_bowl"],
    "ramekin": ["glazed_rim_porcelain_ramekin"],
    "plate": ["plate"],
    "book": ["black_book", "yellow_book"],
    "frying pan": ["chefmate_8_frypan"],
    "moka pot": ["moka_pot"],
    "wine bottle": ["wine_bottle"],
    "alphabet soup": ["alphabet_soup"],
    "tomato sauce": ["tomato_sauce"],
    "ketchup": ["ketchup"],
    "bbq sauce": ["bbq_sauce"],
    "salad dressing": ["salad_dressing", "new_salad_dressing"],
    "milk": ["milk"],
    "orange juice": ["orange_juice"],
    "butter": ["butter"],
    "cream cheese": ["cream_cheese"],
    "chocolate pudding": ["chocolate_pudding"],
    "cookies": ["cookies"],
    "macaroni": ["macaroni_and_cheese"],
    "popcorn": ["popcorn"],
}
GROC = [
    "alphabet soup",
    "tomato sauce",
    "ketchup",
    "bbq sauce",
    "salad dressing",
    "milk",
    "orange juice",
    "butter",
    "cream cheese",
    "chocolate pudding",
    "cookies",
    "macaroni",
    "popcorn",
]
ALL = list(OBJECTS)

# Destination kinds: phrase variants as (phrase, scene item providing the destination), and the objects that
# sensibly go there. "beside" is one kind (a spot on the table next to something), whatever the reference.
_D = {
    "basket": (["in the basket"], "basket", GROC),
    "tray": (["in the tray"], "tray", GROC + ["mug", "bowl", "ramekin", "moka pot"]),
    "caddy": (
        [f"in the {p} of the caddy" for p in ("front", "back", "left", "right")],
        "caddy",
        ["book", "butter", "cream cheese", "chocolate pudding", "ramekin"],
    ),
    "drainer": (
        [f"in the {p} side of the drainer" for p in ("left", "right")],
        "drainer",
        ["mug", "bowl", "plate", "ramekin", "frying pan"],
    ),
    "drawer": (
        [f"in the {p} drawer" for p in ("top", "middle", "bottom")],
        "cabinet",
        [
            "butter",
            "cream cheese",
            "chocolate pudding",
            "cookies",
            "macaroni",
            "popcorn",
            "alphabet soup",
            "tomato sauce",
            "ketchup",
            "bbq sauce",
            "book",
            "bowl",
            "ramekin",
            "mug",
        ],
    ),
    "microwave": (["in the microwave"], "microwave", ["mug", "bowl", "ramekin", "popcorn", "macaroni"]),
    "fridge": (
        ["in the fridge"],
        "fridge",
        [
            "milk",
            "orange juice",
            "butter",
            "cream cheese",
            "ketchup",
            "bbq sauce",
            "salad dressing",
            "chocolate pudding",
            "tomato sauce",
            "wine bottle",
        ],
    ),
    "shelf": (
        ["on the shelf", "on the lower shelf"],
        "shelf",
        [
            "book",
            "mug",
            "bowl",
            "ramekin",
            "moka pot",
            "wine bottle",
            "alphabet soup",
            "tomato sauce",
            "cookies",
            "macaroni",
            "popcorn",
            "milk",
            "orange juice",
            "ketchup",
            "bbq sauce",
            "butter",
            "cream cheese",
            "chocolate pudding",
        ],
    ),
    "bookcase": (
        ["in the bookcase"],
        "bookcase",
        [
            "book",
            "mug",
            "ramekin",
            "cookies",
            "macaroni",
            "butter",
            "cream cheese",
            "chocolate pudding",
            "popcorn",
        ],
    ),
    "pan": (["in the pan"], "frying pan", ["butter", "cream cheese"]),
    "bowl": (
        ["in the bowl"],
        "bowl",
        ["butter", "cream cheese", "chocolate pudding", "ramekin", "cookies", "popcorn"],
    ),
    "plate": (
        ["on the plate"],
        "plate",
        ["mug", "bowl", "ramekin", "butter", "cream cheese", "chocolate pudding", "cookies", "popcorn"],
    ),
    "cabinet top": (
        ["on the cabinet"],
        "cabinet",
        [
            "mug",
            "book",
            "ramekin",
            "moka pot",
            "alphabet soup",
            "tomato sauce",
            "ketchup",
            "cookies",
            "popcorn",
            "butter",
            "cream cheese",
            "chocolate pudding",
            "milk",
            "orange juice",
            "bbq sauce",
            "salad dressing",
            "wine bottle",
            "macaroni",
        ],
    ),
    # "on the microwave" was dropped (18 Nov 26): LIBERO's microwave top_side site sits above the real top, so its
    # On check never succeeds for an object resting there.
    "storage box": (
        ["on the storage box"],
        "storage box",
        [
            "mug",
            "bowl",
            "ramekin",
            "book",
            "moka pot",
            "alphabet soup",
            "tomato sauce",
            "butter",
            "cream cheese",
            "chocolate pudding",
            "cookies",
            "popcorn",
            "ketchup",
            "milk",
            "orange juice",
        ],
    ),
    # "on the book" was dropped (18 Nov 26): LIBERO's books stand upright.
    "placemat": (["on the placemat"], "placemat", ["plate", "bowl", "mug", "ramekin"]),
    "lift": (
        ["lift"],
        None,
        [
            "mug",
            "bowl",
            "book",
            "milk",
            "orange juice",
            "ketchup",
            "salad dressing",
            "alphabet soup",
            "moka pot",
            "wine bottle",
            "cookies",
            "macaroni",
        ],
    ),
}
DESTS = {d: ([(ph, item) for ph in phrases], objs) for d, (phrases, item, objs) in _D.items()}
DESTS["beside"] = (
    [
        (f"{rel} the {ref}", ref)
        for ref in ("plate", "bowl", "basket", "tray")
        for rel in ("to the left of", "to the right of", "in front of", "behind")
    ],
    [o for o in ALL if o != "plate"],
)
LARGE_ITEMS = {
    "basket",
    "tray",
    "caddy",
    "drainer",
    "cabinet",
    "microwave",
    "fridge",
    "shelf",
    "bookcase",
    "storage box",
    "placemat",
    "stove",
    "faucet",
}

# Room themes, so that a task's destinations and a scene's fixtures belong to one kind of room.
K, S, LV = "kitchen", "study", "living"
ANY = {K, S, LV}
DEST_THEMES = {
    "basket": {LV},
    "tray": {LV, K},
    "caddy": {S},
    "drainer": {K},
    "drawer": {K, S},
    "microwave": {K},
    "fridge": {K},
    "shelf": {S, LV},
    "bookcase": {S, LV},
    "pan": {K},
    "bowl": ANY,
    "plate": ANY,
    "cabinet top": {K, S},
    "storage box": {LV, S},
    "placemat": {LV, K},
    "lift": ANY,
    "beside": ANY,
    "faucet": {K},
    "stove": {K},
}
ITEM_THEMES = {
    "fridge": {K},
    "microwave": {K},
    "stove": {K},
    "faucet": {K},
    "drainer": {K},
    "cabinet": {K, S},
    "caddy": {S},
    "bookcase": {S, LV},
    "shelf": {S, LV},
    "basket": {LV},
    "tray": {LV, K},
    "placemat": {LV, K},
    "storage box": {LV, S},
}

# Steps LIBERO-90 / LIBERO-10 already use (object kind, destination kind): avoided.
SEEN = {
    ("moka pot", "stove"),
    ("bowl", "plate"),
    ("mug", "plate"),
    ("bowl", "bowl"),
    ("alphabet soup", "basket"),
    ("frying pan", "stove"),
    ("bowl", "cabinet top"),
    ("tomato sauce", "basket"),
    ("cream cheese", "basket"),
    ("book", "caddy"),
    ("bowl", "tray"),
    ("chocolate pudding", "beside"),
    ("butter", "drawer"),
    ("butter", "basket"),
    ("bowl", "drawer"),
    ("book", "shelf"),
    ("bowl", "shelf"),
    ("bowl", "beside"),
    ("frying pan", "shelf"),
    ("mug", "microwave"),
    ("tomato sauce", "tray"),
    ("salad dressing", "tray"),
    ("milk", "basket"),
    ("ketchup", "tray"),
    ("ketchup", "drawer"),
    ("ketchup", "basket"),
    ("cream cheese", "tray"),
    ("chocolate pudding", "tray"),
    ("chocolate pudding", "drawer"),
    ("butter", "tray"),
    ("alphabet soup", "tray"),
    ("orange juice", "basket"),
}

# Tasks with an open / close / turn step: one per fixture (the fixture counts as object and destination).
# (template, fixture, scene items, destination of the paired placing step or None). The drawer, microwave and
# stove steps are themselves in LIBERO-90, so they only appear paired with a new placing step.
ARTIC = [
    ("turn on the faucet", "faucet", ["faucet"], None),
    ("open the fridge and put the {o} in it", "fridge", ["fridge"], "fridge"),
    ("put the {o} in the microwave and close it", "microwave", ["microwave"], "microwave"),
    ("open the {p} drawer and put the {o} in it", "drawer", ["cabinet"], "drawer"),
    ("turn on the stove and put the {o} in the pan", "stove", ["stove", "frying pan"], "pan"),
]

STOP = {
    "the",
    "a",
    "an",
    "and",
    "it",
    "in",
    "on",
    "of",
    "to",
    "put",
    "place",
    "both",
    "then",
    "into",
    "onto",
    "its",
    "at",
    "pick",
    "up",
    "them",
    "from",
    "side",
    "is",
}
MULTI = sorted([k for k in OBJECTS if " " in k] + ["storage box"], key=len, reverse=True)
N_SINGLE, N_DOUBLE, N_SCENES, MIN_SCENE, MAX_SCENE = 30, 30, 20, 2, 4
MAX_LARGE, MAX_SMALL = 3, 7
DEST_CAP = {
    "lift": 4,
    "beside": 6,
    "fridge": 5,
}  # lift, beside: weak motions, kept rare; fridge: kitchen-only
DEFAULT_DEST_CAP, OBJ_CAP = 7, 7


def tokens(text):
    """Content tokens of an instruction; a multi-word name ("cream cheese") is one token."""
    for m in MULTI:
        text = text.replace(m, m.replace(" ", "_"))
    return frozenset(w for w in text.replace(",", "").split() if w not in STOP)


def cells(t):
    """(object, destination) pairings a task covers; two tasks break the hard rule iff they share one."""
    return {(o, d) for o in t.objs for d in t.dests}


class Task:
    def __init__(self, kind, steps, variants, artic=None):
        self.kind, self.steps, self.variants, self.artic = kind, list(steps), list(variants), artic
        self.phrase = self._phrase()
        self.tokens = tokens(self.phrase)
        self.objs = {o for o, _ in self.steps} | ({ARTIC[artic][1]} if artic is not None else set())
        self.dests = {d for _, d in self.steps} | ({ARTIC[artic][1]} if artic is not None else set())
        self.items = self._items()
        self.pairs = set(itertools.combinations(sorted(self.tokens), 2))

    def _phrase(self):
        if self.artic is not None:
            tmpl, _, _, dest = ARTIC[self.artic]
            return tmpl if dest is None else tmpl.format(o=self.steps[0][0], p=self.variants[0] or "")
        parts = []
        for (o, d), v in zip(self.steps, self.variants, strict=True):
            parts.append(f"lift the {o}" if d == "lift" else f"put the {o} {DESTS[d][0][v][0]}")
        return " and ".join(parts)

    def _items(self):
        it = set(ARTIC[self.artic][2]) if self.artic is not None else set()
        for (o, d), v in zip(self.steps, self.variants, strict=True):
            it.add(o)
            vi = v if isinstance(v, int) else 0
            item = DESTS[d][0][vi][1]
            if item:
                it.add(item)
        return it


CANDS = [(o, d) for d, (_, objs) in DESTS.items() for o in objs if (o, d) not in SEEN]

# Physical pairing test results (pair_test.py): keep only pairings, and phrase variants, that worked in simulation.
import json as _json  # noqa: E402
import os as _os  # noqa: E402

_PAIRING = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pairing_ok.json")
OK_VARIANTS = None
if _os.path.exists(_PAIRING):
    _ok = _json.loads(Path(_PAIRING).read_text())
    OK_VARIANTS = {}
    for _d, _objs in _ok.items():
        for _o, _e in _objs.items():
            if _e["variants"]:
                # beside was tested on one variant: a table region works for every reference and relation
                OK_VARIANTS[(_o, _d)] = list(range(len(DESTS[_d][0]))) if _d == "beside" else _e["variants"]
    CANDS = [c for c in CANDS if c in OK_VARIANTS]


def ok_variants(o, d):
    return (
        OK_VARIANTS[(o, d)]
        if OK_VARIANTS is not None and (o, d) in OK_VARIANTS
        else list(range(len(DESTS[d][0])))
    )


# ---------------------------------------------------------------- construction
def weighted_choice(rng, items, weight):
    w = [weight(x) for x in items]
    r = rng.random() * sum(w)
    for x, wx in zip(items, w, strict=True):
        r -= wx
        if r <= 0:
            return x
    return items[-1]


def build(rng):
    tasks, covered = [], set()
    dc, oc = Counter(), Counter()

    def fits(t):
        if cells(t) & covered or any(t.phrase == u.phrase for u in tasks):
            return False
        if any(dc[d] + 1 > DEST_CAP.get(d, DEFAULT_DEST_CAP) for d in t.dests):
            return False
        return all(oc[o] + 1 <= OBJ_CAP for o in t.objs)

    def add(t):
        tasks.append(t)
        covered.update(cells(t))
        dc.update(t.dests)
        oc.update(t.objs)

    def step_weight(c):  # favour least-used objects and destinations
        o, d = c
        return 1.0 / (1 + dc[d]) ** 2 / (1 + oc[o]) ** 2

    def variant(o, d, excl):
        ok = [v for v, (_, item) in enumerate(DESTS[d][0]) if item not in excl and v in ok_variants(o, d)]
        return rng.choice(ok) if ok else None

    for i, (_, _, _, dest) in enumerate(ARTIC):  # open / close / turn tasks first
        if dest is None:
            add(Task(1, [], [], artic=i))
            continue
        for _ in range(200):
            o = rng.choice([o for o, d in CANDS if d == dest] or DESTS[dest][1])
            p = (
                rng.choice([["top", "middle", "bottom"][v] for v in ok_variants(o, dest)])
                if dest == "drawer"
                else None
            )
            t = Task(2, [(o, dest)], [p], artic=i)
            if fits(t):
                add(t)
                break
        else:
            return None
    n_single_art = sum(1 for a in ARTIC if a[3] is None)
    for _ in range(N_DOUBLE - (len(ARTIC) - n_single_art)):  # two-step tasks are harder to place: first
        for _ in range(400):
            c1 = weighted_choice(rng, [c for c in CANDS if c[1] != "lift"], step_weight)
            v1 = variant(c1[0], c1[1], {c1[0]})
            if v1 is None:
                continue
            item1 = DESTS[c1[1]][0][v1][1]
            pool = [
                c
                for c in CANDS
                if c[1] not in ("lift", c1[1])
                and c[0] not in (c1[0], item1)
                and DEST_THEMES[c[1]] & DEST_THEMES[c1[1]]
            ]
            if not pool:
                continue
            c2 = weighted_choice(rng, pool, step_weight)
            v2 = variant(c2[0], c2[1], {c2[0], c1[0], item1})
            if v2 is None or DESTS[c2[1]][0][v2][1] == c1[0]:  # (variant() already keeps item2 != item1)
                continue
            t = Task(2, [c1, c2], [v1, v2])
            if fits(t):
                add(t)
                break
        else:
            return None
    for _ in range(N_SINGLE - n_single_art):
        for _ in range(400):
            c = weighted_choice(rng, CANDS, step_weight)
            v = variant(c[0], c[1], {c[0]})
            if v is None:
                continue
            t = Task(1, [c], [v])
            if fits(t):
                add(t)
                break
        else:
            return None
    return tasks


def alternatives(t):
    """Tasks differing from t in one step or one phrase variant (open/close/turn tasks: their placing step)."""
    out = []
    if t.artic is not None:
        dest = ARTIC[t.artic][3]
        if dest is not None:
            for o in {o for o, d in CANDS if d == dest}:
                for p in (
                    [["top", "middle", "bottom"][v] for v in ok_variants(o, dest)]
                    if dest == "drawer"
                    else [None]
                ):
                    out.append(Task(2, [(o, dest)], [p], artic=t.artic))
        return out
    for k in range(len(t.steps)):
        other = t.steps[1 - k] if t.kind == 2 else None
        oitem = DESTS[other[1]][0][t.variants[1 - k]][1] if other else None
        for o, d in CANDS:
            if other and (
                d == "lift"
                or o in (other[0], oitem)
                or d == other[1]
                or not DEST_THEMES[d] & DEST_THEMES[other[1]]
            ):
                continue
            for v, (_, item) in enumerate(DESTS[d][0]):
                if v not in ok_variants(o, d):
                    continue
                if item == o or (other and item in (other[0], oitem)):  # two steps never share an item
                    continue
                steps, variants = list(t.steps), list(t.variants)
                steps[k], variants[k] = (o, d), v
                out.append(Task(t.kind, steps, variants))
    return out


def overlap(tasks):
    return sum(max(0, len(a.tokens & b.tokens) - 1) for a, b in itertools.combinations(tasks, 2))


def repair(tasks, passes=10):
    """Swap tasks for alternatives that keep the hard rule and caps, lowering token overlap (then spread)."""
    for _ in range(passes):
        improved = False
        for i in range(len(tasks)):
            rest = tasks[:i] + tasks[i + 1 :]
            covered = set().union(*(cells(u) for u in rest))
            dc, oc = Counter(), Counter()
            for u in rest:
                dc.update(u.dests)
                oc.update(u.objs)
            phrases = {u.phrase for u in rest}
            own = sum(max(0, len(tasks[i].tokens & u.tokens) - 1) for u in rest)
            best = (own, spread(tasks), None)
            for n in alternatives(tasks[i]):
                if (
                    cells(n) & covered
                    or n.phrase in phrases
                    or any(n.steps[k] in SEEN for k in range(len(n.steps)) if n.artic is None)
                ):
                    continue
                if any(dc[d] + 1 > DEST_CAP.get(d, DEFAULT_DEST_CAP) for d in n.dests) or any(
                    oc[o] + 1 > OBJ_CAP for o in n.objs
                ):
                    continue
                ov = sum(max(0, len(n.tokens & u.tokens) - 1) for u in rest)
                if ov > best[0]:
                    continue
                sp = spread(rest + [n])
                if (ov, sp) < best[:2]:
                    best = (ov, sp, n)
            if best[2] is not None:
                tasks[i] = best[2]
                improved = True
        if not improved:
            break
    return tasks


def spread(tasks):
    """Lower is better: pairs sharing a destination or an object, plus (weighted) pairs sharing >1 token."""
    dc, oc = Counter(), Counter()
    for t in tasks:
        dc.update(t.dests)
        oc.update(t.objs)
    c = sum(n * (n - 1) / 2 for n in dc.values()) + sum(n * (n - 1) / 2 for n in oc.values())
    return c + 5 * sum(1 for a, b in itertools.combinations(tasks, 2) if len(a.tokens & b.tokens) > 1)


# ---------------------------------------------------------------- scenes
W_SCENE_DEST, W_SCENE_OBJ, W_ITEMS, W_THEME = 300, 60, 1000, 500


def scene_themes(items):
    """Themes that cover every large item of a scene (empty: the scene mixes rooms)."""
    th = set(ANY)
    for it in items & LARGE_ITEMS:
        th &= ITEM_THEMES[it]
    return th


def scene_violations(members, tasks):
    dc, oc, items = Counter(), Counter(), set()
    for i in members:
        dc.update(tasks[i].dests)
        oc.update({o for o, _ in tasks[i].steps})
        items |= tasks[i].items
    out = []
    if any(n > 1 for n in dc.values()):
        out.append("repeats a destination")
    if any(n > 1 for n in oc.values()):
        out.append("repeats a moved object")
    if len(items & LARGE_ITEMS) > MAX_LARGE or len(items - LARGE_ITEMS) > MAX_SMALL:
        out.append("over the item budget")
    if not scene_themes(items):
        out.append("mixes rooms")
    return out


def scene_cost(members, tasks):
    dc, oc, items = Counter(), Counter(), set()
    for i in members:
        t = tasks[i]
        dc.update(t.dests)
        oc.update({o for o, _ in t.steps})
        items |= t.items
    c = W_SCENE_DEST * sum(n - 1 for n in dc.values() if n > 1)
    c += W_SCENE_OBJ * sum(n - 1 for n in oc.values() if n > 1)
    large, small = len(items & LARGE_ITEMS), len(items - LARGE_ITEMS)
    c += W_ITEMS * (max(0, large - MAX_LARGE) + max(0, small - MAX_SMALL))
    c += W_THEME * (not scene_themes(items))
    c += 10 * (large + small)  # prefer compact scenes: tasks that share items sit together
    return c


def group(tasks, rng, iters=200000):
    """Anneal tasks into N_SCENES scenes of MIN_SCENE..MAX_SCENE tasks: swap two tasks, or move one."""
    order = list(range(len(tasks)))
    rng.shuffle(order)
    scenes = [order[k::N_SCENES] for k in range(N_SCENES)]
    scene_of = {i: si for si, m in enumerate(scenes) for i in m}

    def large(m):
        return frozenset(set().union(*(tasks[i].items for i in m)) & LARGE_ITEMS)

    def same_sets(sc):
        c = Counter(large(m) for m in sc)
        return 400 * sum(n - 1 for n in c.values() if n > 1)

    costs = [scene_cost(m, tasks) for m in scenes]
    dup = same_sets(scenes)
    t_start, t_end = 200.0, 0.2
    for it in range(iters):
        temp = t_start * (t_end / t_start) ** (it / iters)
        i = rng.randrange(len(tasks))
        si = scene_of[i]
        move = (
            rng.random() < 0.3
        )  # move task i to another scene; otherwise swap it with task j of another scene
        if move:
            j, sj = None, rng.randrange(N_SCENES)
            if sj == si or len(scenes[si]) <= MIN_SCENE or len(scenes[sj]) >= MAX_SCENE:
                continue
            scenes[si].remove(i)
            scenes[sj].append(i)
        else:
            j = rng.randrange(len(tasks))
            sj = scene_of[j]
            if si == sj:
                continue
            scenes[si][scenes[si].index(i)] = j
            scenes[sj][scenes[sj].index(j)] = i
        ci, cj = scene_cost(scenes[si], tasks), scene_cost(scenes[sj], tasks)
        nd = same_sets(scenes)
        delta = ci + cj - costs[si] - costs[sj] + nd - dup
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            scene_of[i] = sj
            if not move:
                scene_of[j] = si
            costs[si], costs[sj], dup = ci, cj, nd
        elif move:
            scenes[sj].remove(i)
            scenes[si].append(i)
        else:
            scenes[si][scenes[si].index(j)] = i
            scenes[sj][scenes[sj].index(i)] = j
    return scenes


# ---------------------------------------------------------------- report
def room_for(items, living_rooms):
    th = scene_themes(items)
    if K in th and (th == {K} or items & {"fridge", "microwave", "stove", "faucet", "drainer", "frying pan"}):
        return "kitchen"
    if S in th and (th == {S} or items & {"caddy", "book"}):
        return "study"
    if LV in th:
        return next(living_rooms)
    return "kitchen" if K in th else ("study" if S in th else "mixed")


def report(tasks, scenes, out_path, args, rooms=None, ids=None, header=None):
    pairs = list(itertools.combinations(range(len(tasks)), 2))
    hard = [(a, b) for a, b in pairs if (tasks[a].objs & tasks[b].objs) and (tasks[a].dests & tasks[b].dests)]
    tshare = Counter(len(tasks[a].tokens & tasks[b].tokens) for a, b in pairs)
    share_d = sum(1 for a, b in pairs if tasks[a].dests & tasks[b].dests)
    share_o = sum(1 for a, b in pairs if tasks[a].objs & tasks[b].objs)
    seen = sum(1 for t in tasks if t.artic is None for s in t.steps if s in SEEN)
    dc, oc = Counter(), Counter()
    for t in tasks:
        dc.update(t.dests)
        oc.update(t.objs)
    if rooms is None:
        living_rooms = itertools.cycle(["living room", "coffee table", "tabletop", "floor"])
        rooms = [room_for(set().union(*(tasks[i].items for i in m)), living_rooms) for m in scenes]
    ids = ids or list(range(1, len(scenes) + 1))
    rank = {
        r: k
        for k, r in enumerate(
            ["kitchen", "study", "living room", "coffee table", "tabletop", "floor", "mixed"]
        )
    }
    order = sorted(range(len(scenes)), key=lambda s: (rank[rooms[s]], ids[s]))
    bad_scenes = {
        s: [x for x in v if not (x == "mixes rooms" and header)]
        for s, m in enumerate(scenes)
        if (v := scene_violations(m, tasks))
    }
    bad_scenes = {s: v for s, v in bad_scenes.items() if v}
    same = Counter(frozenset(set().union(*(tasks[i].items for i in m)) & LARGE_ITEMS) for m in scenes)
    n_same = sum(n - 1 for n in same.values() if n > 1)
    lines = [
        "# Task list: 60 new LIBERO tasks (libero_ext)",
        "",
        header
        or (
            f"Generated by `scripts/libero_tasks/plan_tasks.py --seed {args.seed} --restarts {args.restarts}` "
            "(18 Nov 26). Rules and reasoning: `dataset_gen.md` §2. Objects are named by kind; each scene "
            "picks one LIBERO model per kind (for example, which of the three mugs). Rooms are a first "
            "assignment, from the scene's fixtures."
        ),
        "",
        "## Checks",
        "",
        f"- {sum(t.kind == 1 for t in tasks)} single-step and {sum(t.kind == 2 for t in tasks)} two-step tasks "
        f"in {len(scenes)} scenes ("
        + ", ".join(f"{n} of {k} tasks" for k, n in sorted(Counter(map(len, scenes)).items(), reverse=True))
        + "). Rooms: "
        + ", ".join(f"{r} {n}" for r, n in Counter(rooms).most_common())
        + ".",
        f"- Pairs sharing both an object and a destination (hard rule): {len(hard)}.",
        "- Instruction pairs by number of shared content tokens: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(tshare.items()))
        + " (target: none above 1).",
        f"- Placing steps already used by LIBERO-90/10: {seen}.",
        f"- Pairs sharing a destination: {share_d} of {len(pairs)} ({100 * share_d / len(pairs):.1f}%); "
        f"sharing an object: {share_o} ({100 * share_o / len(pairs):.1f}%).",
        f"- Scenes that repeat a destination or moved object, exceed the item budget or mix rooms: "
        f"{len(bad_scenes)}"
        + (" (" + "; ".join(f"{', '.join(v)}" for v in bad_scenes.values()) + ")" if bad_scenes else "")
        + ".",
        f"- Scenes with the same large items as another scene: {n_same}.",
        "- Destination uses (tasks): " + ", ".join(f"{d} {n}" for d, n in dc.most_common()) + ".",
        "- Object uses (tasks): " + ", ".join(f"{o} {n}" for o, n in oc.most_common()) + ".",
        "",
        "## Scenes and tasks",
        "",
        "Each scene lists its large items (fixtures and containers), then its movable objects. (1) and (2) mark "
        "single-step and two-step tasks. † = the faucet's on/off check has to be written.",
        "",
    ]
    for s in order:
        m = scenes[s]
        items = set().union(*(tasks[i].items for i in m))
        lines.append(
            f"**S{ids[s]:02d} · {rooms[s]}** · "
            + ", ".join(sorted(items & LARGE_ITEMS))
            + " · "
            + ", ".join(sorted(items - LARGE_ITEMS))
        )
        lines.append("")
        for i in sorted(m, key=lambda i: (tasks[i].kind, tasks[i].phrase)):
            t = tasks[i]
            mark = " †" if t.artic is not None and ARTIC[t.artic][1] == "faucet" else ""
            lines.append(f"- ({t.kind}) {t.phrase}{mark}")
        lines.append("")
    Path(out_path).write_text("\n".join(lines))
    if getattr(args, "json", None):
        art_phrase = {"fridge": "in the fridge", "microwave": "in the microwave", "stove": "in the pan"}
        plan = {"seed": args.seed, "restarts": args.restarts, "scenes": []}
        for n, s in enumerate(order, 1):
            m = scenes[s]
            items = sorted(set().union(*(tasks[i].items for i in m)))
            ts = []
            for i in sorted(m, key=lambda i: (tasks[i].kind, tasks[i].phrase)):
                t = tasks[i]
                art = ARTIC[t.artic][1] if t.artic is not None else None
                steps = []
                for (o, d), v in zip(t.steps, t.variants, strict=True):
                    if art == "drawer":
                        ph = f"in the {v} drawer"
                    elif art in art_phrase:
                        ph = art_phrase[art]
                    else:
                        ph = DESTS[d][0][v][0]
                    steps.append([o, d, ph])
                ts.append(
                    {
                        "language": t.phrase,
                        "kind": t.kind,
                        "steps": steps,
                        "artic": art,
                        "artic_variant": t.variants[0] if art == "drawer" else None,
                    }
                )
            plan["scenes"].append({"id": n, "room": rooms[s], "items": items, "tasks": ts})
        Path(args.json).write_text(_json.dumps(plan, indent=1))
    print("\n".join(lines[6:15]))


def from_plan(path):
    """Tasks, scenes (index lists), rooms and ids from a plan.json (after any scene splits or room moves)."""
    plan = _json.loads(Path(path).read_text())
    art_index = {a[1]: i for i, a in enumerate(ARTIC)}
    tasks, scenes, rooms, ids = [], [], [], []
    for sc in plan["scenes"]:
        members = []
        for t in sc["tasks"]:
            art = t.get("artic")
            if art is not None:
                i = art_index[art]
                if ARTIC[i][3] is None:
                    task = Task(1, [], [], artic=i)
                else:
                    o, d, _ = t["steps"][0]
                    task = Task(2, [(o, d)], [t.get("artic_variant")], artic=i)
            else:
                steps, variants = [], []
                for o, d, ph in t["steps"]:
                    steps.append((o, d))
                    variants.append([v for v, (p, _) in enumerate(DESTS[d][0]) if p == ph][0])
                task = Task(t["kind"], steps, variants)
            assert task.phrase == t["language"], (task.phrase, t["language"])
            members.append(len(tasks))
            tasks.append(task)
        scenes.append(members)
        rooms.append(sc["room"])
        ids.append(sc["id"])
    return tasks, scenes, rooms, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", default=None, help="regenerate the markdown from an (edited) plan.json")
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="projects/dataset_gen_tasks.md")
    ap.add_argument(
        "--json", default="scripts/libero_tasks/plan.json", help="machine-readable plan for build_suite.py"
    )
    args = ap.parse_args()
    if args.from_json:
        tasks, scenes, rooms, ids = from_plan(args.from_json)
        args.json = None
        n_split = sum(
            1 for sc in _json.loads(Path(args.from_json).read_text())["scenes"] if sc.get("split_from")
        )
        header = (
            "Generated by `scripts/libero_tasks/plan_tasks.py --seed 41 --restarts 400`, then adjusted while "
            f"building the suite (18 Nov 26): {n_split} crowded scenes were split in two "
            "(`split_scenes.py`; S21-S25) and two scenes changed room (S20 floor -> tabletop, S22 study -> "
            "tabletop). Scene numbers match the task names in `src/lerobot/envs/libero_ext/suite.json`. "
            "Rules and reasoning: `dataset_gen.md` §2; validation: `dataset_gen_build.md`. Objects are named "
            "by kind; the build picks one LIBERO model per kind."
        )
        report(tasks, scenes, args.out, args, rooms=rooms, ids=ids, header=header)
        return
    rng = random.Random(args.seed)
    best, ok = None, 0
    for _ in range(args.restarts):
        tasks = build(rng)
        if tasks is None:
            continue
        ok += 1
        if best is None or spread(tasks) < spread(best):
            best = tasks
    print(f"{ok} of {args.restarts} constructions completed")
    if best is None:
        raise SystemExit("no complete construction; loosen the caps or add destinations")
    before = overlap(best)
    best = repair(best)
    print(f"token overlap (extra shared tokens over all pairs): {before} -> {overlap(best)}")
    # group into scenes: several restarts, keeping the grouping with the fewest scene violations
    tries = []
    for _ in range(8):
        sc = group(best, rng, iters=300000)
        bad = sum(1 for m in sc if scene_violations(m, best))
        tries.append((bad, sum(scene_cost(m, best) for m in sc), sc))
    tries.sort(key=lambda x: x[:2])
    print(f"scene groupings with violations: {[t[0] for t in tries]}")
    report(best, tries[0][2], args.out, args)


if __name__ == "__main__":
    main()
