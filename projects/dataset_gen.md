# Dataset generation: new LIBERO tasks for Paper 1

Working notes, not a research log. The aim is to provide 30–60 new tasks for Paper 1 item 1 (research_log2, executive summary). The tasks must be disjoint from libero_90 (the pretraining corpus) and from libero_10. Each task needs:
- expert demonstrations, built into a LeRobot dataset in the same format as libero_10;
- init-state files, so the existing evaluation instrument runs on it unchanged.

**Plan.**
1. Investigate the LIBERO backend (done 18 Nov 26, §1).
2. Propose how to create the new environments (§2), then discuss.
3. Decide how to get expert data for them (§3). To be discussed once the environments exist.

---
## 1. How LIBERO defines a task

This describes hf-libero 0.1.3, the version in the cluster venv. Its core is identical to the original LIBERO code (the local clone `~/phddev/LIBERO`); only asset loading differs.

**A task is three things:**
- **A BDDL file.** It names:
  - the problem class (the arena);
  - the fixtures and movable objects;
  - a placement region for each object;
  - the initial predicates;
  - the goal, a conjunction of predicates;
  - the language instruction.
- **An init-state file** (`.pruned_init`): 50 flattened MuJoCo states, an array of shape (50, D). Evaluation starts episodes from these.
- **An entry in a suite** (`libero_task_map` plus a registered `Benchmark` class). This gives the task its index in the suite, and its language, which is parsed from the filename.

**Runtime.**
- `OffScreenRenderEnv(bddl_file_name=...)` parses the BDDL, builds the scene through the problem class, and evaluates the goal predicates at every step (`check_success`).
- The wrapper fixes the robot (mounted Panda), the controller (OSC_POSE, 20 Hz) and the cameras (agentview and wrist). A new task therefore has the same action and observation spaces as libero_10 automatically.

**Authoring tools.** These are needed only to write BDDL files; nothing at runtime uses them.
- *Scene templates* are `InitialSceneTemplates` subclasses registered with `@register_mu`. Each lists the fixtures, objects, placement regions (centre, half-width, yaw) and initial predicates. LIBERO's 20 scenes are in `benchmark/mu_creation.py`.
- `register_task_info(language, scene, goal_states)`, then `generate_bddl_from_task_info()`, writes one BDDL file per task.

**What there is to build from.**
- **Arenas (problem classes):**
  - kitchen, living-room and study tabletops (libero_90 and libero_10);
  - a plain tabletop (libero_spatial and libero_goal);
  - a floor (libero_object);
  - a coffee table, which no suite uses.
- **Goal predicates.** No existing task uses Stack, Up or InContact; the others are used this often across all suites:

  | Predicate | Uses |
  |---|---|
  | On | 61 |
  | In | 63 |
  | Close | 11 |
  | TurnOn | 8 |
  | Open | 7 |
  | TurnOff | 1 |

  Goals can only be conjunctions.
- **Objects.** 49 object classes are registered and 31 are used by some suite.
  - Unused, with assets present:
    - fixtures: short fridge, short cabinet, slide cabinet, window, faucet, basin faucet;
    - containers and furniture: white storage box, bowl drainer, wooden shelf, dining set, rack;
    - groceries: popcorn, macaroni and cheese.
  - Registered but with no assets in LIBERO's asset pack: cherries, corn, mayo.
  - Containers and fixtures expose named regions that In and Open goals refer to. Examples:
    - short fridge: upper, middle and lower;
    - bowl drainer: left and right;
    - desk caddy: four compartments;
    - cabinets: top, middle and bottom.
- **Start-state variation is small.** Each object is sampled within ±1–2.5 cm of its region centre, and evaluation always starts from the 50 stored states.

**libero_10 against libero_90.** Every libero_10 task is set in a libero_90 scene: its 9 scenes all appear among libero_90's 20. So libero_10 tasks are new goals, mostly two-step, in scenes the base model has already seen. Even so, the SmolVLA libero_90 base scores only 7.2 on them zero-shot, almost all of it on e5 (66; every other task scores 0–4).

**Integration with our stack.**
- lerobot's `LiberoEnv` looks suites up in hf-libero's benchmark registry. It finds BDDL and init files through the LIBERO config (`get_libero_path`).
- On the cluster that config (`$CLUSTER_ROOT/env/libero`) points into the venv's hf-libero package.
- Episode limits per suite are in `TASK_SUITE_MAX_STEPS` (`src/lerobot/envs/libero.py`).
- A new suite therefore needs three things:
  - its BDDL and init files somewhere the cluster can read (the lerobot repo: they are small text and numpy files);
  - a registry entry;
  - a step limit.
- hf-libero is a pip package inside the venv, so our code should add the registry entry at import time (a few lines), rather than editing or forking the package.
- **Gotcha:** the registry parses the language from the filename and assumes single-digit scene numbers, with SCENE10 as a special case. New scenes need names that parse, or the language should be taken from the BDDL's `:language` field.

**Probes** (18 Nov 26, desk PC, hf-libero 0.1.3, MuJoCo 3.8 as on the cluster):
- **New goal in an existing scene.** In LIBERO's kitchen scene 1: "open the bottom drawer of the cabinet and put the black bowl on the plate", which is in neither libero_90 nor libero_10.
  - The BDDL was generated, and the environment was built and reset in 1.8 s, with 256×256 agentview and wrist images.
  - Success was False at reset and stayed False with only the bowl on the plate. It became True once the drawer joint was also set open.
- **New scene from unused objects.** A kitchen table with the short fridge, bowl drainer, popcorn, macaroni and cheese and a white bowl, with two tasks: "put the white bowl on the bowl drainer" and "open the fridge". Both built, settled and rendered, with success False at the start.

The authoring path works end to end, with no change to LIBERO.

---
## 2. Proposal: creating the new environments

**Three levels of novelty, in order of cost:**
- **A. New goals in LIBERO's 20 existing scenes.** This is how libero_10 was made.
  - Cost: BDDL files only.
  - Limit: every scene, object and layout was in pretraining, and many of the obvious goals are already libero_90 tasks.
- **B. New scenes built from LIBERO's assets.** New combinations of fixtures and objects in new layouts, including the 13 unused objects and the coffee-table arena, with several goals per scene.
  - Cost: per scene, placing the regions so that resets are valid and everything is reachable and visible. After that, goals are cheap.
- **C. New object assets** (meshes from outside LIBERO).
  - Cost: mesh import, plus collision and grasp tuning for each object.
  - Not needed for 30–60 tasks.

**Recommendation: B as the core, A as a top-up.**
- ~~Build about 12 new scenes with 4–6 goals each, giving 50–70 candidate tasks, filtered to 30–60.~~ Superseded by the decisions below: 100 tasks over about 30 scenes. Author some spare candidates, since validation and the expert-data step will rule some tasks out.
- Scenes give natural families (tasks that share a scene). Item 1 needs these for its two task orders: grouped (a family together) and interleaved.
- Goals: single steps (pick-and-place with On or In, Open or Close, TurnOn or TurnOff) mixed with two-step compositions in libero_10's style, so that difficulty is spread.
- Keep every goal within a fixed set of primitives:
  - grasp and place on a surface;
  - place in a container region;
  - open or close a drawer or door;
  - turn a knob.

  This bounds the work in step 3, whatever the expert turns out to be.

**Pipeline.** The code is written and committed locally, then pulled to the cluster.
1. **Author.** One Python module of scene templates and task registrations writes the BDDL files into a suite folder in the lerobot repo (e.g. `scripts/libero_tasks/`).
2. **Validate every task automatically:**
   - it builds and resets for 50 or more samples (placement can fail in crowded layouts);
   - objects settle, with no drift, nothing off the table and nothing interpenetrating;
   - the goal is False at every start state and stays False over 300 no-op steps;
   - the goal is reachable: set the goal state by hand (joints, object poses), settle, and success becomes True, as in the probe;
   - the objects involved are visible in both cameras and inside the arm's workspace;
   - it does not duplicate a libero_90 or libero_10 task (same scene and same goal predicates);
   - a contact sheet of start states is rendered for a check by eye.
3. **Init states.** 50 per task, taken from sampled resets that pass the checks and saved as `.pruned_init` in LIBERO's format.
4. **Register the suite** (e.g. `libero_ext`) at import in lerobot, with its step limit, so that `lerobot-eval` and the chain run on it unchanged.
5. **After step 3:** measure the libero_90 base's zero-shot score on the new tasks as the novelty check. It should be near 0, as on libero_10.

**Decisions (18 Nov 26):**
- **100 new tasks.** libero_10 is not part of the sequence.
- **At most 3–4 tasks per scene,** so about 30 scenes (25 at the very least). Scenes can be new, or LIBERO's existing ones with new goals.
- **50 single-step tasks and 50 two-step tasks.**

**Asset budget: existing assets are enough; no new 3D models are needed.** These are the 46 usable object classes (49 registered, less the 3 with no meshes):

| Kind | Count | Objects |
|---|---|---|
| Graspable | 26 | 14 groceries; black bowl, white bowl, ramekin, plate; 3 mugs; 2 books; frying pan, moka pot, wine bottle |
| Containers and receivers | 10 | basket, tray, desk caddy (4 compartments), bowl drainer (2 regions), storage box, wooden shelf, two-layer shelf, wine rack, rack, dining set |
| Articulated, with working goal checks | 6 | wooden, white and short cabinets (3 drawers each), microwave, fridge, stove |
| Articulated, with no open/close check | 4 | slide cabinet, window, faucet, basin faucet. Each would need a check written (a few lines), plus testing. |

- **Arenas:** 6 rooms (kitchen, living room, study, plain tabletop, floor, coffee table), with 13 colour styles between them. Using a non-default style needs a small problem subclass registered from our code.
- **Goals.** Distinct object-target pairs run to the hundreds. Across 100 tasks, each graspable object would be the moved object in roughly 5 tasks, about as often as LIBERO-90 reuses its objects.
- **What limits variety is the set of motions, not the objects.** Every goal is pick-and-place, open or close, or turning a knob, as in LIBERO generally. New meshes would not change that; the four unfinished articulated fixtures would, a little.
- **Novelty.** Apart from the 13 unused objects, every object appeared in pretraining. Novelty comes from new layouts and new goals, as with libero_10, whose tasks still score 7.2 zero-shot.
- **If more unseen objects are wanted later,** the cheapest source is the HOPE groceries. LIBERO's grocery class already loads them, and three (cherries, corn, mayo) are registered but missing their meshes.

**Separation between tasks (proposed 18 Nov 26).** Capacity is the thing being measured, so the tasks must be clearly dissimilar. Near-duplicates would share memory slots and inflate the task count. The libero_10 basket family (envs 7, 0, 1: two groceries into the same basket) is the pattern to avoid. On pi05 it was the worst drift cell and accounted for the whole deficit against the specialists.

- **Task signature.** Each step of a task is (motion, moved object, receiver).
  - Motions: place in, place on, place beside, open, close, turn on, turn off.
  - Objects map to categories: box, can, bottle, mug, bowl, plate, book, cookware.
  - Receivers map to classes: any drawer, any caddy compartment, any fridge shelf, the basket, the plate, and so on.
- **Rule 1, no near-duplicates anywhere in the set.** Two tasks are near-duplicates if their steps match once objects are mapped to categories and receivers to classes. That is, they differ only by a within-category swap: mug ↔ cup, ketchup ↔ BBQ sauce, top ↔ bottom drawer, left ↔ right compartment, left ↔ right plate.
- **Rule 2, a scene's 3–4 tasks never repeat a step type** (motion × receiver class) within that scene.
- **Rule 3, a two-step task's two steps differ in motion or receiver class.** "Put both X and Y in the basket" is out.
- **Rule 4, budgets across all 100 tasks,** so that usage is spread evenly:
  - each step type appears in at most about 5 tasks, counting both single-step tasks and steps of two-step tasks;
  - each object instance is the moved object in at most about 5 tasks.
- **Later empirical check.** Once demonstrations exist, run the held-out routing audit on all 100 tasks, as for libero_10. Replace members of any pair whose core overlap stands well above background. This is the memory's own measure of similarity, which is what capacity depends on.

**The binding constraint is the motion vocabulary.** With working goal checks, the assets support about 26 step types:
- place in, 11 receivers: basket, tray, caddy, drainer, drawer, microwave, fridge, shelf, two-layer shelf, wine rack, dining set;
- place on, 7 receivers: plate, stove, cabinet top, shelf top, storage-box top, microwave top, beside an object on the table;
- open or close a drawer, the microwave or the fridge: 6 step types;
- turn the stove on or off: 2 step types.

The 100 tasks contain 150 steps (50 single-step tasks plus 50 × 2), so each step type recurs in about 6 tasks. Rule 1 still holds, because the recurrences differ in object category, scene and partner step. But the tasks share skills at roughly LIBERO-90's level of reuse.

Two cheap levers widen the vocabulary without new models:
- **Finish the four fixtures with no checks.** The window, faucet, slide cabinet and basin faucet need an open/close or on/off check each, giving about 7 more step types.
- **Use the predicates that exist but are unused:** Stack (bowl on bowl) and Up (lift an object clear of the table).

Together these give about 35 step types, so each recurs in about 4 tasks. New predicates written in code (for example "push X to the front" or "stand the book upright") would add more. Their demonstrations would be harder to obtain, though, so they are a last resort.

**Josh's rule (18 Nov 26).** For a given action, two tasks must differ in both the object and the destination. "Put the ketchup in the drawer" and "put the mug in the sink" is fine; "put the ketchup in the drawer" and "put the bowl in the drawer" is not. Destinations are then the binding constraint: each action–destination pair can be used once. That gives about 30 pairs, or about 38 with the four fixtures finished, which is roughly 20–25 tasks at 50/50.

**Does a different scene separate tasks? Measured on the existing audits (18 Nov 26).**
- Method: the held-out routing audit's per-task read distributions for libero_10, from `cell_s2_seq5` (the full layout the memory runs used) and three reduced-layout warm-ups (`wu2_sep2`, `wu2_ref`, `wu2_sep32`). The 45 task pairs are split by how the two tasks relate. Each cell is the weighted read IoU as a multiple of the background: unrelated pairs in different room types.
- The table shows `cell_s2_seq5`, as the range over sites. The three reduced-layout runs agree.

| Pair type | Pairs | Expert sites | VLM sites |
|---|---|---|---|
| Unrelated, different room type (background) | 29 | 1.0× | 1.0× |
| Unrelated, same room type | 11 | 1.1–1.5× | 0.9–1.0× |
| Same scene, same destination, different objects (env 0–1, basket) | 1 | 1.2–2.3×, mean 1.5× | 1.8–3.0×, mean 2.4× |
| Shared object only (env 4–9: the yellow-and-white mug, plate against microwave, different rooms) | 1 | 2.2× | 2.2× |
| Different scene, same destination and a shared object (env 7–0 and 7–1 basket, 4–6 mug on plate, 2–8 moka pot on stove) | 4 | 2.3–6.1× per pair | 2.8–4.7× per pair |

- **The room or scene contributes almost nothing.** Unrelated tasks in the same room type overlap about as much as unrelated tasks in different rooms.
- **What drives overlap is what the tasks share in content.**
  - Sharing a destination alone is mild: about 1.5× at the expert sites, 2.4× at the VLM sites.
  - Sharing an object alone is about 2.2×.
  - Sharing both, even across scenes, is 2.3–6×.
  - The highest pair, env 7–1 (cream cheese into the basket in two different scenes, 6.1× at the expert sites), is the one behind the biggest forgetting event in the ten-task run: env 7 fell from 88 to 68 when env 1 was trained.
- **Overlap tracks instruction wording.** Across all 45 pairs, the share of content words two instructions have in common correlates with routing overlap at r = 0.85–0.86 at the expert sites and 0.90 at the VLM sites. Shared words and shared objects go together in this data, so the two cannot be told apart. Either way, word overlap between instructions is a usable proxy for routing overlap, and it can be computed for a candidate task list before anything is built.
- **Caveats.** Only 6 related pairs. The cross-scene pairs are in similar rooms. Every cross-scene pair also shares an object, and no libero_10 pair shares only a destination across scenes.
- **Reading.** Varying the scene is not a separation lever. Separation has to come from distinct objects and destinations (and wording), which is what Josh's rule targets.

**Agreed rule (18 Nov 26), replacing the stricter rule above:**
- **Hard: no two tasks share both an object and a destination.**
- **Soft: shared destinations are permitted but kept to a minimum.** Each destination recurs in as few tasks as the count allows (about 4 at most), and in different contexts.
- **Proposed, to confirm:**
  - **Objects get the same soft treatment as destinations.** A shared object alone costs about as much as a shared destination alone (2.2× against 1.5× / 2.4×). Reuse is also unavoidable: about 110 placing steps over 26 graspable objects is about 4 uses each.
  - **Fixtures count as both object and destination** in open, close and turn tasks. So "open the fridge" and "close the fridge" cannot both be tasks. The fridge can still recur as a destination ("put the milk in the fridge").
- **Design-time check:** the content-word overlap of every pair of instructions. Target: no pair shares more than one content word, the env 0–1 level (one word, "basket"). The env 7–1 pair shared four words (cream, cheese, box, basket).
- ~~**Expected spread with even reuse:** about 5% of task pairs share a destination and about 3% share an object; none share both.~~ Correction (18 Nov 26): 100 tasks under this rule turned out to be infeasible (feasibility check below). The 60-task draft has 9.2% and 8.1%.

**Fixture check (18 Nov 26).** The four fixtures without goal checks, rendered on a table:
- The faucet (a lever tap) is usable once its on/off check is written.
- The window is a thin slab and is not useful.
- The basin faucet fails to load.
- The slide cabinet has no placement regions, so LIBERO's authoring tool rejects it.
- The dining set is a placemat with cutlery, which makes a usable new destination ("on the placemat").

**Feasibility under the agreed rule (18 Nov 26): 100 tasks at 50/50 cannot be built from the current assets.**
- **Vocabulary.** `scripts/libero_tasks/plan_tasks.py` has a hand-written table of sensible pairings: 21 object kinds × 19 destination kinds, plus one open/close/turn task per fixture. That table gives only **140 placing steps** that make physical sense and are not already in LIBERO-90/10. Of those, 30 are "lift" or "beside", which are kept rare.
- **Why that is not enough.** The rule lets each (object, destination) pair appear in only one task. For a two-step task this covers all four pairings of its two objects with its two destinations, not just its own two steps. 100 tasks at 50/50 need about 145 placing steps and about 240 such pairings.
- **Greedy construction:** the largest set found in 20 randomised builds per variant. A better search might add about 10%.

| Rule | Two-step share | Max tasks |
|---|---|---|
| Agreed rule, no LIBERO-90/10 steps, lift/beside capped | 50% | 64 |
| same | 25% | 84 |
| same | 0% | 120 |
| Agreed rule, LIBERO-90 steps allowed, no caps | 50% | 78 |
| Weaker rule (no two tasks contain the same step; sharing across steps allowed), LIBERO-90 steps allowed, no caps | 50% | 100 |
| One-shared-word limit (stricter than the agreed rule) | 50% | 52–56 |

~~The one-shared-word limit is unreachable at any useful size, so it becomes a reported statistic, not a target.~~ Correction (18 Nov 26): with a repair pass, the 60-task draft below meets it. The figures in the table are greedy builds without repair.

**Decision (18 Nov 26): about 60 tasks, 50/50, under the agreed rule** (option 1). This replaces the earlier decision of 100. Josh's view: if 60 tasks do not saturate the memory, that is a good result in its own right.

**Draft task list (18 Nov 26): `dataset_gen_tasks.md`.** Generated by `scripts/libero_tasks/plan_tasks.py --seed 11 --restarts 400`. Superseded the same day by the built suite (§2b): the physical pairing test changed the vocabulary and the planner was rerun (seed 41).
- **Contents:** 60 tasks (30 single-step, 30 two-step) in 20 scenes of 2–4 tasks: 5 of 4, 10 of 3 and 5 of 2.
- **Rooms:** kitchen 8, study 5, living room 2, coffee table 2, tabletop 2, floor 1.
- **Checks:**
  - No pair shares an object and a destination.
  - No pair of instructions shares more than one content word. The greedy builds alone left 15–20 such pairs; a repair pass then swaps tasks for alternatives that keep the rule.
  - No placing step is already in LIBERO-90/10.
  - No scene repeats a destination or a moved object, exceeds 3 large or 7 small items, or mixes rooms.
  - No two scenes have the same set of fixtures.
- **Reuse:** 9.2% of task pairs share a destination and 8.1% share an object. Each destination is used in at most 6 tasks and each object in at most 6.
- **Method.** A randomised greedy build adds tasks that cover no new object–destination pairing already used, sampling the least-used objects and destinations first. A repair pass follows. Tasks are then grouped into scenes by annealing.
- **Constraints added along the way:**
  - The two destinations of a two-step task fit one kind of room (kitchen, study, living room).
  - The two steps of a task never use the same item, so not "in the microwave and on the microwave".
  - A "beside X" reference is never the task's other destination.
  - The fridge is capped at 5 tasks, since it is kitchen-only.
- **Spares.** There are none: the rule allows only about 64 tasks. If validation or the expert-data step rules tasks out, the planner should regenerate around the tasks that are kept.
- **Next:** Josh reviews the list. Then each scene gets a layout (placement regions), its BDDL files are generated, and the validation checks of the pipeline above run on every task.

---
## 2b. Building the suite (18 Nov 26)

**Where things live.**
- **Runtime extension:** `src/lerobot/envs/libero_ext/__init__.py`, registered when `lerobot.envs.libero` is imported, so it is present in every worker process. It adds three things to hf-libero without changing the package:
  - a `lifted` goal predicate;
  - on/off checks for the faucet;
  - the `libero_ext` suite, read from `suite.json`, with its BDDL and start-state files in `files/`.
- **Use:** `--env.type=libero --env.task=libero_ext`. The episode limit is 520 steps, as for libero_10, until expert demonstrations exist.
- **Authoring tools:** `scripts/libero_tasks/`:
  - `plan_tasks.py`: the task list;
  - `pair_test.py`: physical test of every object–destination pairing;
  - `split_scenes.py`: splits crowded scenes;
  - `scene_lib.py`: layout, BDDL writing, goal forcing and checks;
  - `build_suite.py`: layout search, validation and start states;
  - `smoke_suite.py`: end-to-end test through lerobot's env.
- **Build report:** `dataset_gen_build.md`.

**Physical pairing test** (`pair_test.py`). Each pairing is tested in a two-item scene:
- the goal is false after a settled reset;
- forcing the goal (joints set, the object teleported into or onto its destination, settled) makes the success check fire;
- no object interpenetrates by more than 6 mm.

149 of 154 pairings pass. This changed the vocabulary:
- **Dropped "on the book":** LIBERO's books stand upright.
- **Dropped "on the microwave":** its top_side site sits above the real top, so the On check never fires for an object resting there.
- **"On the placemat" is checked with a table region over the mat,** because the mat's cloth does not collide.
- **Failing pairings:**
  - mug, plate and frying pan in the drainer (its slots are vertical);
  - macaroni in a drawer;
  - salad dressing in the basket.
- **Added pairings:** cabinet top, shelf, storage box, bookcase and bowl accept more objects, to make up for the dropped destinations.
- The planner was rerun on the passing pairings (seed 41). It satisfies the same rules; 5 instruction pairs share two content words.

**Layout and validation** (`scene_lib.py`, `build_suite.py`). Each scene gets a layout, and each of its tasks a BDDL file with its own start conditions (a door or drawer open or closed). Every task must pass all of the following, or the layout is resampled:
- **Stable start:** three resets, each settled for 80 steps. Movable objects drift less than 1 cm over the last 20 steps, none tips over by more than 25°, none ends up below the work surface, and the goal is false.
- **Forced goal reached:** the success check fires, with no interpenetration.
- **In view:** the moved objects and every approach point project inside the agentview image.
- **In reach:** a proportional end-effector controller gets within 5 cm of a point above each moved object and of each destination's approach point:
  - 6 cm in front of a fixture's opening;
  - above an open drawer's pulled-out part;
  - 7 cm in front of a closed drawer;
  - the inner part of a side fixture's top.

Once a layout passes, each task gets 50 start states from settled resets that pass the same checks, saved in LIBERO's `.pruned_init` format.

**What the build measured and changed:**
- **Reach is shallow towards the camera.** Item centres are kept within these bounds (from reach maps, with the end effector within 3 cm at surface + 6 cm):

  | Arena | x | y |
  |---|---|---|
  | kitchen / tabletop | −0.32 to +0.08 | ±0.32 |
  | study | −0.26 to +0.16 | ±0.32 |
  | living room / coffee table | −0.20 to +0.20 | ±0.32 |
  | floor | −0.30 to +0.10 | ±0.32 |

  The y limit comes from the camera's field of view. Side fixtures sit outside these bounds with their front face at |y| = 0.28; only their openings and the inner part of their tops must be in view and in reach.
- **Clearance in front of doors and drawers** (22 cm fridge, 18 cm microwave, 17 cm drawers), only where a task in the scene uses that door or drawer. The fridge door, swung open, blocked both the arm and the objects beside it.
- **LIBERO's table tops are not at the nominal workspace height:** +1.6 cm in the study, +2.4 cm in the living room, +1.2 cm at the coffee table. Objects spawn slightly inside the table and pop out, so settling takes 80 steps, and the `lifted` predicate measures from the real surface.
- **Five crowded scenes could not be laid out in the reachable area,** so each was split in two (`split_scenes.py`): S03, S10, S14, S20 and S05 gave tasks to S21–S25. The suite now has 25 scenes of 1–4 tasks.
- **"Beside" regions are sized by the moved object.** The region now sits clear of the reference by the moved object's own half extent plus 2 cm, not a fixed 7 cm, which had put the 36 cm frying pan on top of the basket it was to sit beside (S23).
- **The reach tolerance is 5 cm** (it was 4 cm). The closed fridge door was reached at 4.6–4.9 cm.
- **Two scenes changed room:**
  - S20 moved to the tabletop. On the floor and coffee-table arenas the table-region marker misses a plate lying on the mat.
  - S22 moved from the study to the tabletop. There the arm could not reach the side cabinet's drawer front.

**Result (18 Nov 26): 60 tasks in 25 scenes, each with 50 validated start states** (`src/lerobot/envs/libero_ext/`, 2.8 MB).
- **Task list:** `dataset_gen_tasks.md`, regenerated from the final `plan.json` with `plan_tasks.py --from-json`.
- **Validation:** `dataset_gen_build.md`.
- **The checks still hold after the splits and moves:**
  - no pair shares an object and a destination;
  - 5 instruction pairs share two content words;
  - no placing step comes from LIBERO-90/10;
  - 10.9% of pairs share a destination and 8.2% share an object.
- **Rooms:** kitchen 9, study 6, tabletop 4, living room 3, coffee table 2, floor 1.
- **End-to-end test** (`smoke_suite.py`, through lerobot's own LIBERO env):
  - the suite registers with all 60 tasks;
  - `LiberoEnv(task="libero_ext")` builds vectorised envs whose observations match libero_10's (two 256×256 cameras, end-effector, gripper and joint state);
  - in all 180 runs (60 tasks × 3 stored start states), success is false from the start state and true once the goal is forced.
- The new code passes the repo's ruff and ruff-format hooks.

**Caveats for the expert-data step.**
- The fridge's shelves (about 8 cm high) and the bookcase's (about 6.5 cm) will be hard to fill with a top-down grasp.
- Validation shows every goal is reachable as a state, and every approach point as a position. No task has yet been demonstrated end to end.

---
## 3. Expert data (to discuss once the environments exist)

**Constraints whatever the source:**
- **Actions** in the space of LIBERO's demonstrations: 7-D OSC_POSE deltas plus the gripper, from the same controller configuration, one action per control step.
- **Observations** matching our libero_10 dataset:
  - 256×256 agentview and wrist images (stored rotated 180°, as in the HF data);
  - 8-D state;
  - 7-D action.

  This lets the `build_libero_suites.py`-style conversion and `replay_libero_actions.py` apply unchanged.
- **Demonstrations:** about 40–50 successful ones per task (libero_10 has 379 episodes over 10 tasks).
- **Privileged state.** The simulator exposes object poses, joint states and the goal predicates. A scripted or RL expert can use them; the policy never sees them.

**Options to weigh then:**
- scripted primitives driven by privileged state;
- RL on privileged state, with a sparse reward from the success predicate or a shaped one, then recorded rollouts;
- a few teleoperated demonstrations expanded MimicGen-style.

**Decision (18 Nov 26): a scripted expert on privileged state.** RL would need a shaped reward per task (as much work as a script), runs at about 1,000 simulator steps/s on the CPU, and gives jerky motion. Teleoperation costs hours of Josh's time. Reasons in full in the chat of 18 Nov 26.

**Variety.** LIBERO's start states vary very little: across our 50 stored states, each object moves within a 2–3 cm box, no object rotates, and the robot always starts in the same pose. In LIBERO's own data the variety comes from the human operator. Near-identical demos would make each task cheaper to store, and so overstate capacity. Variety therefore comes from three places:
- **The expert's parameters,** drawn per episode: speed, approach and transport heights, grasp point and orientation, release height, pauses, and the order of a two-step task's independent steps.
- **DART-style noise:** the executed action is perturbed, and the recorded label is the expert's action from the state actually reached.
- **Wider training starts:** objects nudged a further ±2 cm, round ones turned, the arm's joints jittered by ±0.03 rad. Evaluation keeps the stored states.

**Pilot (18 Nov 26).** `scripts/libero_tasks/expert.py` holds the expert; `pilot_expert.py` runs it from fresh resets, with half the episodes using widened starts and noise.
- **How it works.** The plan comes from the task's steps. The expert picks with a top-down grasp: across the body for boxes, bottles and cans, or on the rim for mugs, bowls, ramekins and plates. Among candidate grasp directions (8) or rim points (12) it chooses one whose gripper model (fingers and a 21 × 6 cm hand) clears every other object's collision geometry. It carries the object above everything along its path. At the drop point it picks a clear hand orientation and the lowest clear release height, and aims the object's origin, not the hand, at the target. It releases there or retries; it never drops the object elsewhere.
- **Insertion** (microwave, fridge, bookcase): the expert grips with the fingers parallel to the fixture's face, turns the wrist, pitches the hand to point into the opening, slides the object in at the region's height, then backs out.
- **Bugs found on the way:**
  - robosuite's grasp check misses real grasps; a grasp now counts if the object rose with the gripper;
  - the hand's 21 cm width hit neighbouring objects, so grasps are now chosen for clearance;
  - a bottle caught the cabinet edge in transport, so transport now clears obstacles by 6 cm;
  - rim-held mugs were placed 5 cm off target, so the expert now aims the object rather than the hand.
- **Success rates** (16 episodes each, the last run):

  | Rate | Tasks |
  |---|---|
  | 100% | cookies on the plate, ketchup on the shelf, lift the milk, wine bottle in front of the basket, popcorn in the basket, mug on the shelf, ramekin in the drainer, wine bottle in the fridge, popcorn in the bookcase |
  | 88–94% | ramekin on the placemat, BBQ sauce in the top drawer |
  | 62% | milk in the tray + wine bottle on the cabinet (two-step) |

  Earlier runs of 8–12 episodes: chocolate pudding in the bowl 92%; bowl in the drainer + mug on the placemat 58%, mostly out of steps; ramekin in the microwave 38%; chocolate pudding in the bookcase 25%; cream cheese, book and mug into the bookcase 0%. Only successful episodes are kept, so rates above about 30% are usable; whether an insertion works depends on the object.
- **Variety against libero_10's human demos** (same statistics on both):

  | Statistic | libero_10, human (10 two-step tasks) | Scripted pilot (12 tasks) |
  |---|---|---|
  | Episode-length coefficient of variation | 0.10 | 0.13 |
  | End-effector path spread at matched task phase | 6.0 cm | 5.1 cm |
  | Std of first-grasp time (fraction of the episode) | 0.031 | 0.040 |

  The scripted demos are as varied as the human ones. Their mean length is 217 steps against 272, as expected with mostly single-step tasks.
- **Not done yet** (as of the pilot; see §3b):
  - routines for the five open/close/turn tasks;
  - a systematic demonstrability test of every pairing;
  - the SmolVLA learnability check.

## 3b. Open/close/turn routines, demonstrability test, replan (18 Nov 26)

**Open/close/turn routines** (`expert.py`):
- **Faucet lever and microwave door:** a closed-loop push. Every step, the closed gripper aims 3 cm ahead, along the part's motion, of its contact point on the part, wherever the part has actually got to (open-loop arcs stalled or kept pushing after the goal). Doors are pushed 4 cm below their top edge, so the hand body passes over it. The push continues until the goal has held for 6 steps, because a door slammed shut bounces open.
- **Stove:** grip the knob's raised ridge from above and turn the wrist.
- **Drawer:** the hand turned horizontal (fingers above and below the handle bar), a 20 cm pull, and a second grip and pull if the drawer stops short of open.
- **Episodes end at the first step where the goal holds,** as LIBERO's evaluation episodes do. A shut microwave door springs back ~0.01 rad once released, so the goal holds only while the gripper is still pressing.
- **Pilot:** faucet 10/10 (58 steps); stove knob 3/3; microwave door 5/5 once pressing continued until shut; drawer 8/12 in fresh layouts.

**Changes to the task set:**
- **"Open the fridge" is dropped.** The short fridge's door is flush with its body: no handle a gripper can hold, and only 5 mm of overhang to pinch. Dragging its top edge by friction was the only thing that moved it, and unreliably. The fridge stays as a destination, with its door open at the start.
- **Drawer opening uses the top drawer only.** The horizontal hand reaches ~10 cm below the handle; at the middle drawer (11 cm above the table) it knocked over objects standing in front of the cabinet. Fixtures the robot opens or closes itself get a larger clear strip in front: cabinet 28 cm, microwave 22 cm (was 17 and 18 cm).
- **LIBERO bug:** the coffee-table arena puts table regions at floor height (its `_load_sites_in_arena` omits the table's z offset), so a "beside" or placemat goal can never hold there. On the floor arena such goals fail for flat objects (a ramekin's origin sits below the region). Scenes with such goals now take the living room or tabletop arena; scenes whose items suit no single room take the tabletop arena.

**Other expert changes:**
- **Grasps use the local width:** the parts of the object between the finger pads at that height, with the open fingers clear of the object's own handle or spout, instead of the whole bounding box. The moka pot, whose handle and spout make its box 15 cm wide, became graspable at its 6.6 cm base.
- **A rim grasp counts as held at a finger gap of 2 mm or more** (was 6 mm; a bowl's wall is ~4 mm thick).
- **Speed:**
  - a move gives up when blocked (no progress for 15 steps) instead of spending its whole step allowance;
  - shorter waits;
  - speed drawn from 0.8–1.0 (was 0.7–1.0).

  One two-step task went from ~470 to ~395 steps.

**Demonstrability test** (`pair_expert.py`):
- **What it covers:** every pairing that passed the physical test (§2b), and each open/close/turn template with each object its placing step may use.
- **Setup:** small scenes with only the items the task needs, laid out at random by the builder's sampler, in the room the planner would give them. Each pairing gets 3 layouts × 3 episodes, one in three widened with DART noise.
- **Results** (200 pairings, 1,776 episodes): 169 pairings kept, at a threshold of ≥ 5/9 episodes for placements and ≥ 1/2 for templates.

| Destination | Kept | Not kept (successes / 9) |
|---|---|---|
| basket, bowl, caddy, drawer (open), pan, plate, tray | all (5, 6, 8, 26, 2, 6, 9) | — |
| beside | 16/18 | frying pan 2, salad dressing 3 |
| cabinet top | 17/18 | moka pot 4 |
| shelf | 27/28 | mug on the lower shelf 1 |
| storage box | 13/15 | moka pot 2, bowl 3 |
| lift | 10/12 | book 4, salad dressing 3 |
| fridge | 6/10 | orange juice 3, butter 3, cream cheese 2, chocolate pudding 3 |
| microwave | 3/4 | ramekin 2 |
| drainer | 3/4 | bowl (right side) 4 |
| placemat | 2/4 | plate 3, bowl 3 |
| bookcase | 1/9 | all but popcorn: 0–1 |
| open the top drawer and put X in it | 5/9 | book 2, cookies 4, cream cheese 2, mug 4 |
| put X in the microwave and close it | 1/4 | popcorn 1, bowl 0, ramekin 1 (macaroni kept, 5/9) |
| turn on the stove and put X in the pan | 2/2 | — |
| turn on the faucet | 1/1 | — |

Insertion through a side opening (bookcase, fridge, microwave) is the expert's weak point. The bookcase works only for the popcorn box.

*Correction (18 Nov 26).* These runs had a bug: a geometry cache served a closed environment's geom ids to the next environment built in the same process (fixed in `81f8c649`).
- Six pairings lost layouts to the resulting errors.
- Other episodes may have failed on wrong object boxes.
- Successes are unaffected, because they come from the environment's own goal check, not the expert's geometry. So every kept pairing is genuinely demonstrable; a few excluded ones may not deserve to be.
- The suite build, which also used the cache (stability and below-surface checks), was redone after the fix.

**Replan** (`plan_tasks.py --seed 4`):
- **Same rules as §2:** 60 tasks, 30 single-step and 30 two-step, no two tasks sharing both an object and a destination, and four open/close/turn tasks (faucet, microwave, drawer, stove).
- **Kept pairings only, and the 5/9 threshold.** At 6/9, too few pairings remain: several destinations have only 1–2 usable objects, and the planner found no complete task set.
- **The cap on tasks per destination is raised from 7 to 9.** The common destinations (shelf, cabinet top, storage box, tray) have the spare objects.
- **25 scenes of 2–3 tasks,** planned directly rather than split after building.
- **Checks:** no instruction pair shares more than 2 content words (11 pairs share 2), and 12.9% of task pairs share a destination.
- The task list is in `dataset_gen_tasks.md`.

**Rebuild:**
- `build_suite.py` adds an expert check to each candidate layout: the expert must succeed in 2 of 3 episodes of every task of the scene, or the layout is rejected.
- 23 of the 25 scenes built. The two crowded kitchen scenes found no layout that fits: S02 (fridge, microwave, five objects) and S06 (cabinet with the 28 cm drawer strip, fridge, tray). They were split in two (S02 → S26, S06 → S27, tasks unchanged).
- **Result: 60 tasks in 27 scenes.** In its accepted layout, the expert succeeded 3/3 on 46 tasks and 2/3 on 14.
- `smoke_suite.py`: 180/180, through lerobot's LIBERO env on 3 stored start states per task (goal false at the start, true once forced).
- Details: `dataset_gen_build.md`.

**Data pipeline:**
- `generate_demos.py` (desk PC, no rendering) runs the expert from fresh and widened starts until 50 successes per task. Per episode it saves the simulator positions before each action, the 8-D state, the expert's action (the label) and the executed action. That is about 100 MB for the whole suite.
- `scripts/datasets/build_libero_ext.py` (cluster GPU node, `cluster/jobs/build_libero_ext.sh`) sets each saved state, reads the environment's own camera observations at 256 × 256 and writes a LeRobot v3.0 dataset in libero_10's format:
  - images rotated 180°, state 8-D, action 7-D;
  - fps 10;
  - task_index = suite order;
  - quantile stats added.
- States re-read from the saved positions differ from the recorded ones by < 1 mm. robosuite caches observations a few physics substeps before the end of each control step, and the recorded state is that cached value, as evaluation sees it.

**Demos generated (18 Nov 26):**
- 3,000 successful episodes, 50 per task; 942 of them from widened starts with DART noise.
- 652k frames. Mean length 217 steps, longest 489, against the 520-step evaluation limit.
- The expert succeeded on 76% of attempts overall.
- Weakest tasks (successes / attempts):
  - book on the cabinet + moka pot in the tray, 50/173 (failed picks of the book and moka pot);
  - cookies in the middle drawer, 50/171;
  - lift the moka pot, 50/147;
  - macaroni in the basket + moka pot right of the plate, 50/103;
  - book in the bottom drawer + orange juice on the shelf, 50/101.
- The demos are in `outputs/libero_ext_demos` (152 MB) and `/SAN/.../data/libero_ext_demos`.

**Datasets built (18 Nov 26, `cluster/jobs/build_libero_ext.sh`, repo `69e2da0`):**
- **`data/libero_ext`:** 3,000 episodes, 652,137 frames, 60 tasks, 68 GB, quantile stats included.
- **`data/libero_ext_learn4`:** suite tasks 0, 31, 42 and 59 as task_index 0–3; 200 episodes, 49,928 frames, 5 GB.
- States re-read from the saved positions differ from the recorded (cached) ones by at most 1.07 cm, on task 59, the drawer opening. That is under the 2 cm bar for a model mismatch.
- The per-task datasets the build resumes from are in `data/_build/libero_ext` (68 GB), and are redundant now.

**Learnability check** (`configs/ext_learn_spec_r128.sh`, submitted 19 Nov 26):
- LoRA specialists with the libero_10 recipe on suite tasks 0, 31, 42 and 59: cookies in the bowl; ketchup on the shelf + cream cheese in the caddy; chocolate pudding in the bowl + milk in the tray; open the top drawer and put the popcorn in it.
- Each specialist is evaluated on its own task.
- **Zero-shot floor** of the SmolVLA libero_90 base on all 60 tasks (`configs/ext_zeroshot.sh`; 5 episodes each, one seed, 10 actions executed per chunk): **1.0%** mean.
  - Only two tasks succeed at all: "turn on the faucet" 2/5 and "lift the alphabet soup" 1/5.
  - The other 58 tasks are 0/5.
  - The tasks are new to the pretrained model, as intended.

## 3c. Learnability failure and the data fixes (19 Nov 26)

**The check.** LoRA r128 specialists, with the libero_10 recipe (5k steps from the SmolVLA libero_90 base), on four tasks; 25 episodes × 2 seeds each.

| Task | Success |
|---|---|
| 0: cookies in the bowl (one step) | 98 |
| 31: ketchup on the shelf + cream cheese in the caddy | 2 |
| 42: pudding in the bowl + milk in the tray | 40 |
| 59: open the top drawer + popcorn in it | 6 |

On libero_10 the same recipe reaches ~85% on two-step human demos. The 60-task memory run (`cell_s2_ext60_rand1`, research_log2 Entry 5) was stopped after 7 tasks.

**Diagnosis.** Rollouts with videos and per-step goal predicates (`diagnose_rollouts.py`), reviewed by a subagent against the expert demos.
- **Every failure runs the full 520 steps.** Expert demos take 90–115 steps (task 0) and 260–320 (the others), so this is not a step limit.
- **The demos were multimodal in their discrete choices**, and the one task the policy learned was the one whose 50 demos all do the same thing.
  - Two-step tasks in a random order: 23/27 on task 31, 21/22 on task 42.
  - Grasp chosen at random among the clear ones: first-grasp wrist yaw over six modes, −68° to +90°, on task 31; 0° or 90° on task 42; within ±3° on task 0.
  - Also: the faucet pushed either way, and the stove knob turned before or after the placing step.
  - The policy dithered: in task 31 its first grasp landed between the two objects.
- **No recoveries in the demos.** The policy closes 1–3 cm off the expert's grasp: sideways on the cookie box, ~2 cm too high on the pudding and milk, 4–6 cm short of the drawer handle. The expert's grasps almost never fail (49/50 single grasps on task 0), so the demos hold no recoveries. After a miss the policy carries an empty hand to the destination, or loops close/open in place while drifting.
- **Smaller issues:**
  - The grasp point slid along the object at random, up to 0.6 of the free length.
  - In scene S19 (coffee table) the tray sometimes jumps 8–23 cm on its own at t ≈ 25–30 (5 of 50 demos), with the arm 15 cm away. This is a physics glitch, still to fix.

**Fixes** (`expert.py`):
- **v2: discrete choices made the same way in every episode.**
  - Two-step tasks in the instruction's order; on the stove task, the knob first.
  - The clear grasp needing the least wrist rotation.
  - The faucet pushed one way.

  Continuous variety stays: speed, heights, release height, pauses, DART noise, widened starts. Regenerated demos are unimodal: one order, first-grasp yaw within ±4°.
- **v3: v2 plus recovery data, and a narrower grasp slide (0–0.25).**
  - A quarter of first grasp attempts and first drawer-handle grabs are executed wrong on purpose: closing 1.5–2.5 cm too high, beside the object, or 3–5 cm short of the handle.
  - Every recorded label is the expert's action from the state reached: on to the real grasp with the gripper open. Then the expert opens and grasps properly.
  - 12–25 of each task's 50 demos contain such a miss.

**Retests**, same recipe and tasks: `configs/ext_learn_spec_r128_v2.sh` and `_v3.sh`, on `libero_ext_learn4_v2` / `_v3`. Success over 25 episodes × 2 seeds (1000 / 2000):

| Task | Original | v2 | v3 |
|---|---|---|---|
| 0: cookies in the bowl | 98 | **100** (100 / 100) | 96 (92 / 100) |
| 31: ketchup on the shelf + cream cheese in the caddy | 2 | **70** (72 / 68) | 64 (68 / 60) |
| 42: pudding in the bowl + milk in the tray | 40 | **50** (48 / 52) | 16 (12 / 20) |
| 59: open the top drawer + popcorn in it | 6 | 10 (16 / 4) | **28** (36 / 20) |
| Mean | 36.5 | **57.5** | 51.0 |

- **Making the discrete choices deterministic (v2) is the main effect:** task 31 goes from 2 to 70, and the mean from 36.5 to 57.5.
- **Recovery data and the narrower slide (v3) are mixed.**
  - Tasks 0 and 31 do not move beyond the noise (~±10 points at 50 episodes).
  - The drawer rises from 10 to 28, likely from the short-of-the-handle misses.
  - Task 42 falls from 50 to 16. A likely cause is the "too high" misses: on the short pudding box, closing on nothing ~2 cm above it looks much like a real grasp, and its labels say "open and go down", which could teach the policy to let go of good grasps.
- **Still below libero_10 specialist level (~85):** task 42 (50) and the drawer (28 at best).

## 3d. Expert v4 and a 12-task learnability panel (25 Nov 26)

**Correction to §3c's rollout diagnosis.** `diagnose_rollouts.py` gave the policy the task's file name (`S19_put_the_…`) as its instruction instead of the instruction itself, which handicapped every rollout it made:
- **Unaffected:** the demo statistics behind §3c's main finding (random step order and grasp choice), and every success rate in §3c's tables, all of which come from the cluster's eval script.
- **Overstated:** the account of how the policy failed, in particular "closes 1–3 cm off and never recovers", which motivated v3's recovery data. Fixed in `6776ca99`. With the instruction, local rollouts match the cluster's evals: v2 task 42 6/10 (cluster 50%); v2 task 59 1/10 (10%); v3 task 59 4/10 (28%).

**Remaining failure modes**, from the corrected rollouts:
- **Task 42 (crowded coffee-table scene S19, 9 items):** sometimes the pudding is never placed; more often the milk fails. The milk stands next to the storage box, the policy retries over it, then drifts away.
- **Task 59 (drawer):**
  - In about half the episodes the hand turns horizontal and faces the cabinet, but hovers without moving in to the handle. Three near-identical handles are in the wrist camera's view.
  - When the drawer is open, the popcorn placement fails about a third of the time.
  - The v3 handle recoveries help, consistently: drawer opened in 6/10 against 4/10.
- **The S19 tray "glitch" is the expert, not physics.** With the robot still, the tray never moves. Its 8–23 cm flips come from a finger clipping its rim while the hand drops diagonally from the start pose to its first approach point.

**v4** (`expert.py`, `generate_demos.py`):
- v2's deterministic discrete choices.
- Deliberate misses only for the drawer handle (25% of first grabs); grasp misses on objects are off.
- The approach to a grasp passes above everything on the way: the tray flips are gone, and a 2 cm tilt remains in 1 of 30 episodes.
- Clear grasps need 1 cm of clearance, not 5 mm.
- The narrower grasp slide (0–0.25).
- LIBERO-like start states: the suite's own reset, with objects anywhere in their ~3 cm init regions and the arm at home. No widened starts; DART noise stays on one episode in three.

**Panel.** LoRA r128 specialists, with the libero_10 recipe, on twelve tasks covering every task type:

| Task | Type |
|---|---|
| 6: turn on the faucet | lever push |
| 11: tomato sauce in the middle drawer | placement into an open drawer |
| 12: macaroni in the microwave and close it | sideways insertion + door push |
| 19: turn on the stove and cream cheese in the pan | knob twist + placement |
| 27: lift the bowl | rim grasp |
| 32: bbq sauce on the lower shelf | sideways insertion |
| 38: bbq sauce in the tray + popcorn in the bookcase | two steps, bookcase insertion |
| 42: pudding in the bowl + milk in the tray | two steps, crowded coffee table |
| 54: wine bottle left of the basket + pudding in the caddy | two steps, beside + caddy |
| 56: mug in the tray + alphabet soup on the storage box | two steps, floor scene |
| 58: wine bottle in the fridge | sideways insertion |
| 59: open the top drawer and popcorn in it | drawer pull + placement |

The expert succeeded on 71–100% of attempts on these tasks; 50 demos each.

**Panel results** (`configs/ext_panel_spec_r128_v4.sh`, 25–26 Nov 26). Success over 25 episodes × 2 seeds (1000 / 2000). The first subset build contained only task 6, because `qsub -v` splits a value on its commas. The job was resubmitted with a ':'-separated list (`2afe2f04`), which cost about 1 h 20 min.

| Task | Type | 1000 / 2000 | Mean | v2 / v3 (§3c) |
|---|---|---|---|---|
| 6: turn on the faucet | lever push | 88 / 96 | 92 | |
| 11: tomato sauce in the middle drawer | placement into an open drawer | 16 / 8 | **12** | |
| 12: macaroni in the microwave and close it | sideways insertion + door push | 72 / 80 | 76 | |
| 19: turn on the stove + cream cheese in the pan | knob twist + placement | 72 / 68 | 70 | |
| 27: lift the bowl | rim grasp | 96 / 84 | 90 | |
| 32: bbq sauce on the lower shelf | sideways insertion | 92 / 100 | 96 | |
| 38: bbq sauce in the tray + popcorn in the bookcase | two steps, bookcase insertion | 28 / 32 | **30** | |
| 42: pudding in the bowl + milk in the tray | two steps, crowded coffee table | 64 / 60 | 62 | 50 / 16 |
| 54: wine bottle left of the basket + pudding in the caddy | two steps, beside + caddy | 68 / 56 | 62 | |
| 56: mug in the tray + alphabet soup on the storage box | two steps, floor scene | 0 / 0 | **0** | |
| 58: wine bottle in the fridge | sideways insertion | 96 / 80 | 88 | |
| 59: open the top drawer + popcorn in it | drawer pull + placement | 8 / 8 | **8** | 10 / 28 |
| Mean | | | 57.2 | |

Eight tasks reach 62–96, including every sideways insertion and three of the four open/close/turn tasks. Four are far below.

**Diagnosis of the four below 60.** Local rollouts (10 episodes, with the cluster's evaluation settings) and open-loop checks of the specialists.
- **Tasks 11 and 38: two grasp modes were left in the demos.** A mode audit (`audit_demos.py`, below) flags three panel tasks.
  - **Task 11:** 36 demos grasp the sauce with the wrist turned 90° first; 14 grasp it unturned and turn 70–90° later, either way.
    - Cause: the tomato sauce is nearly square (6.3 × 6.2 cm), so both grasps fit. "The least rotation among grasps clear by 1 cm" flipped with the layout, because the unturned grasp's clearance ranged from −0.9 to +1.3 cm.
    - The policy never goes down to the sauce. It closes the gripper 9–45 times in the air, turns the wrist part way, and hovers at the drawer empty-handed (1/10 locally).
  - **Task 38:** the popcorn grasp turned the wrist −90° in 39 demos and +90° in 11, a near-tie decided by numerical noise.
    - The tray step succeeded in 10/10 rollouts.
    - In 4 of the 7 failures the popcorn grasp missed: the finger gap 10 steps after closing was 0.8–2.2 cm, against 6.2 cm when the box is held.
    - In the other 3, the box was held but not inserted into the bookcase.
  - **Task 42** (a 37 / 10 split with the same cause) reached 62 regardless.
- **Task 56: the demos did not start as an evaluation does.**
  - lerobot's `LiberoEnv.reset` ends with 10 no-op steps that command the gripper open. robosuite integrates the gripper command from zero after a reset, so an evaluation's first frame shows the fingers at 0.039.
  - Every demo's first frame showed 0.040, because the demos were recorded straight after the settle. An opening of 0.039 is also what the fingers show just after a release.
  - In S25 the tray lies right below the home pose, so after the mug is placed the scene looks much like the start. The specialist used the fingers to tell the two apart.
    - **Open loop**, at demo start states placed in the evaluation env: without the no-op steps the specialist predicts the demos' first move (y −0.79 to −0.92). With them it predicts the opposite (+0.64 to +0.81), towards the alphabet soup. Swapping in the training frame's wrist image alone restores −0.83; the agentview, the state and the instruction are already identical.
    - **Rollouts:** every episode goes to the soup first (0/10). Without the no-op steps the mug is the first target in 9 of 10, but its rim grasp closes about 2 cm too high and still 0/10 succeed. The floor arena puts the objects farther from both cameras than a table does.
  - The mismatch is in the first frame of every task. Its effect on the other tasks is not known.
- **Task 59: drawer opening stays weak.**
  - The drawer opened in 5–6 of 10 rollouts, and the popcorn placement failed after most openings (4/10 locally, 8% on the cluster).
  - This specialist's training loss ended at 0.12–0.14, against 0.01–0.09 for the others.
- **Not a cause: image size.** lerobot's LIBERO env renders at 360 × 360 by default, and the dataset is 256 × 256, as libero_10's is. Evaluating at 256 did not change task 56's first move.

**Expert v5** (`expert.py`, `generate_demos.py`, `audit_demos.py`):
- **A grasp orientation per task, fixed before its episodes.** For each object the task picks, it is the closing direction clear by 5 mm in most of 20 fresh layouts. When no direction is ever clear, it is the one that most often has the most clearance. The grasp choice prefers it, accepting 5 mm of clearance for it and 1 cm for any other.
- **Turns:**
  - a wrist turn of about 90°, which could go either way, always goes anticlockwise;
  - at the drop, the hand keeps its orientation unless that costs more than 4 cm of release height. Choosing the lowest release first had flipped the turn with the layout (task 8: three modes).
- **A mode filter.** Each episode's signature is the wrist yaw at each change of hand orientation between gripper events. Only episodes in the most common mode are kept, and the share of successes in that mode before filtering is recorded.
- **On all 60 tasks** (3,000 demos, 684k frames, expert success 76%): the most common mode held at least 95% of successes on 49 tasks and at least 80% on 58. The lowest were task 16 (61%) and task 23 (73%). After filtering, every task's 50 demos are in one mode.

**v6 data: v5 plus demos that start as an evaluation does.**
- `generate_demos.py` reproduces `LiberoEnv.reset`: a fresh reset, the start state, then 10 unrecorded no-op steps. The first state's gripper is now 0.0388.
- The v5 render (`libero_ext_v5`, 26 Nov) predates this fix and is superseded.

## 3e. Plan: v6 for all 60 tasks, then the 60-task run (26 Nov 26)

1. **Generate and validate v6 locally:** 60 tasks × 50 demos. Check the mode shares (any task below 0.8 gets a look), the first-frame gripper (0.0388) and the lengths (under 520 steps).
2. **Render on the cluster** (`build_libero_ext.sh`, `V=_v6 LEARN_NAME=libero_ext_panel LEARN_TASKS=6:11:12:19:27:32:38:42:54:56:58:59`): `libero_ext_v6` plus the 12-task `libero_ext_panel_v6`, about 5 h.
3. **v6 panel** (`configs/ext_panel_spec_r128_v6.sh`): the same 12 specialists, recipe and evaluation. Pre-registered reads:
   - Tasks 11 and 38 rise to at least 60 (one mode).
   - Task 56 rises above 0 (start fixed). If it stays below 40, scene S25 (tasks 55 and 56, the only floor scene) moves to a table arena and its two tasks are regenerated.
   - Task 59 (drawer opening, the only such task): if it stays below 40, it is replaced by a placement task from the kept pairings.
   - The other eight hold at 60 or more.
   - **Go criterion:** go on to the 60-task run if the panel mean is at least about 70 and no kept task is below 40.
4. **Replace v1 everywhere:**
   - `libero_ext_v6` becomes the suite's dataset on the cluster. v1 is already deleted there and on the desk PC.
   - On the Backup drive, `libero_generated/libero_ext` (v1, 68 GB) and `libero_ext_demos` are replaced by v6: cluster → desk PC → Backup in the background, about 4–5 h. The README is updated.
   - The v4 and v5 datasets, build directories, demos and specialist checkpoints are removed.
5. **Restart the 60-task memory run** under a new NAME (`cell_s2_ext60v6_rand1`) on `libero_ext_v6`, with the same random task order (`ext60_order_rand1.json`).
   - A smoke test comes first (`smoke_ext60` on v6).
   - research_log2 Entry 5 gets a dated note.
   - Evaluation stays at lerobot's 360 × 360 default, as in every panel so far.
