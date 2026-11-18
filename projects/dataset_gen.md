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
