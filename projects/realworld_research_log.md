# Real-World Research Log - VLA Memory

---
## Entry 0 - Context

This log tracks the real-world pi0.5 memory-layer experiments separately from the main simulation-focused `research_log.md`.

The real-world setup uses:
- Pretraining dataset: `/home/josh/lerobot/outputs/realworld_pretrain` with 15 tasks.
- Sequential dataset: `/home/josh/lerobot/outputs/realworld_seq` with 5 tasks.
- Model: pi0.5 with memory layers on expert layers `[8,10,12,14]`.
- Memory values: LoRA rank 2.
- Retrieval: `mem_knn=36`.
- Routing loss: joint-slot locality/separation, not the older half-key proxy.

Sequential task order:

| Task index | Instruction |
|------------|-------------|
| t0 | Put the mustard in the basket |
| t1 | put the red bow on the plate |
| t2 | Place the orange in the black basket |
| t3 | Push over the white lego brick |
| t4 | Stand the grey bottle up |

Important limitation:
- These real-world sequential runs did not include rollout eval blocks.
- The conclusions below are based on training curves plus memory-slot JSON diagnostics.
- The most useful proxy metrics are weighted read IoU and read-through-updated-slots, not raw slot intersection.

---
## Entry 1 - 22 May 26 (First Real-World Follow-Up Batch)

### Runs Compared

Old baseline:

```text
/home/josh/lerobot/outputs/train/realworld_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36_80k
/home/josh/lerobot/outputs/train/realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36_80k
```

New follow-up runs:

```text
/home/josh/lerobot/outputs/train/realworld_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k
/home/josh/lerobot/outputs/train/realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k

/home/josh/lerobot/outputs/train/realworld_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k
/home/josh/lerobot/outputs/train/realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k

/home/josh/lerobot/outputs/train/realworld_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_8000_knn_36_50k
/home/josh/lerobot/outputs/train/realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_8000_knn_36_50k
```

Scripts:

```text
/home/josh/lerobot/job_scripts/nebius/combined/realworld_pi05_4_layer_film_lora2_sample_contrastive_50k.sh
/home/josh/lerobot/job_scripts/nebius/combined/queue_realworld_pi05_memory_followups.sh
```

The first script ran the `contrastive=0.01, max_support=4096` pair. The queue script then ran:

```bash
run_pair "0.05" "4096"
run_pair "0.01" "8000"
```

### Original Real-World Failure Mode

The old real-world run had several coupled failures:

1. **Contrastive loss dominated the objective.**
   - Final total loss: `2.363`
   - Final MSE: `0.0132`
   - Raw/weighted contrastive: `2.342`
   - The action loss was only a tiny fraction of the training objective.

2. **Memory was narrow and gate-saturated.**
   - Final gate mean: `0.962`
   - Batch `mem_usage_effnum_mean`: `3450`
   - Batch `mem_used_frac_mean`: `0.0875`
   - The model was highly dependent on a relatively small effective memory region.

3. **Sequential writes landed in slots older tasks still read heavily.**
   - L14 `t2` reading slots updated by `t0`: `53.3%`
   - L14 `t4` reading slots updated by `t3`: `21.5%`
   - L14 `t4` reading slots updated by any prior task: `38.9%`

4. **Task 4 had the clearest adaptation weakness.**
   - Task 4 cold-start MSE: `0.1428`
   - Task 4 final MSE: `0.0818`
   - Other tasks finished much lower, typically `0.036-0.053`.

5. **The 80k schedule wasted late training.**
   - LR was effectively at the floor for a large fraction of the run.
   - MSE had already reached the useful range by roughly 32k.

### Pretrain Summary

| Config | Weighted contrastive | Pretrain MSE | Gate | Effnum | Support | Inter-task routing sim |
|--------|----------------------|--------------|------|--------|---------|------------------------|
| old `c=1, sup=2048, 80k` | `2.342` | `0.0132` | `0.962` | `3450` | `3098` | `0.0298` |
| `c=0.01, sup=4096` | `0.0239` | `0.0154` | `0.936` | `5398` | `5837` | `0.0673` |
| `c=0.05, sup=4096` | `0.1179` | `0.0172` | `0.924` | `4404` | `4930` | `0.0516` |
| `c=0.01, sup=8000` | `0.0240` | `0.0155` | `0.936` | `5718` | `5899` | `0.0660` |

Interpretation:

- Reducing contrastive fixed the most obvious objective imbalance.
- `contrastive=0.01` makes the weighted contrastive term comparable to MSE rather than 100x larger.
- `contrastive=0.05` still puts substantial pressure on the contrastive objective and gives worse MSE.
- The new runs use memory more broadly than the old run, but gate is still high.

### Sequential Summary

| Config | Mean seq weighted IoU | Task 4 final MSE | Notes |
|--------|------------------------|------------------|-------|
| old `c=1, sup=2048, 80k` | `0.0564` | `0.0818` | Narrow routing, severe pairwise read-through |
| `c=0.01, sup=4096` | `0.0982` | `0.0667` | Better fit, too much broad shared overlap |
| `c=0.05, sup=4096` | `0.0825` | `0.0583` | Best final MSE, contrastive still heavy |
| `c=0.01, sup=8000` | `0.0858` | `0.0603` | Best pairwise read-through improvement |

Task 4 final MSE improved in all new runs. That is a meaningful sign that the old run's final-task adaptation issue was partly caused by the bad pretrain regime.

However, mean sequential IoU worsened in every new run. The new models are broader and more plastic, but also expose more cumulative write/read interference.

### Key Read-Through Diagnostics

L14 individual read-through:

| Config | `t2 <- t0` | `t4 <- t3` | `t4 <- any prior` |
|--------|------------|------------|-------------------|
| old `c=1, sup=2048, 80k` | `53.3%` | `21.5%` | `38.9%` |
| `c=0.01, sup=4096` | `35.4%` | `7.1%` | `46.9%` |
| `c=0.05, sup=4096` | `40.4%` | `14.5%` | `46.3%` |
| `c=0.01, sup=8000` | `26.7%` | `6.3%` | `44.0%` |

Main finding:

- `c=0.01, sup=8000` best fixes the original worst pairwise collisions.
- But cumulative read-through for task 4 remains high.
- So the new problem is not simply one bad semantic collision. It is broader cumulative exposure to previous writes.

### Interpretation

The new runs moved in the right direction for pretraining:

- Contrastive no longer dominates the objective.
- Effective memory usage increased.
- The old obvious pair collisions were reduced, especially in the `c=0.01, sup=8000` run.
- Task 4 final MSE improved substantially.

But the new runs also changed the failure mode:

- Broader routing means more tasks read through regions that prior tasks have updated.
- This increases mean weighted IoU and cumulative read-through.
- The system appears too plastic under the current sequential write settings.

This is why the real-world answer is not simply "push routing apart harder." There is enough slot capacity in theory, but the router is not a hard task allocator. It is a continuous function trained to fit actions on the pretrain tasks. Future sequential tasks are unseen and naturally land near similar pretrain/online tasks in language, vision, and action space.

The simulation diary also showed that stronger separation has a non-monotonic tradeoff:

- Too little separation causes shared-slot interference.
- Too much separation hurts useful sharing and pretrain fit.
- Raw slot intersection is a poor target because many low-weight tail slots are touched incidentally.
- Weighted read-through and weighted IoU are more meaningful.

### Why More Routing Separation Is Not The Main Next Bet

The relevant pretraining knobs are:

```bash
--policy.memory_layer.routing_inter_task_separation_weight
--policy.memory_layer.routing_intra_task_locality_weight
--policy.memory_layer.routing_intra_task_max_support
--policy.memory_layer.routing_global_balance_weight
```

The indirect query-space knob is:

```bash
--policy.memory_layer.contrastive_loss_weight
```

Reasons not to make harder separation the primary next move:

1. **Future tasks are unseen during pretraining.**
   - Even perfect separation among the 15 pretrain tasks does not guarantee private regions for the 5 sequential tasks.
   - Similar new tasks can still be routed near older task regions.

2. **Useful sharing is real.**
   - Robot tasks share primitives: reach, grasp, lift, align, place, stabilize, push, and pose-correct.
   - Forcing every task into private slots can reduce interference but also reduce transfer and fit.

3. **The new runs already reduced the old pairwise collisions.**
   - The best new run reduced L14 `t2 <- t0` from `53.3%` to `26.7%`.
   - It reduced L14 `t4 <- t3` from `21.5%` to `6.3%`.
   - The remaining issue is cumulative read-through, not just the original pairs.

4. **Broad routing plus unchanged write budget is the current mismatch.**
   - The new pretrains created broader read footprints.
   - Sequential still uses `tfidf_top_t=512` and `memory_value_lr=0.001 -> 0.0001`.
   - That may be too aggressive for these broader real-world routing distributions.

### Current Recommendation

For robot testing:

1. Test `c001_sup8000` first.
   - Best reduction in original pairwise read-through.
   - Good task 4 final MSE.
   - Most directly addresses the old diagnosed failure.

2. Test `c005_sup4096` second.
   - Best final task 4 MSE.
   - Somewhat lower mean IoU than `c001_sup4096`.
   - But contrastive is still more dominant than desired.

3. Do not prioritize `c001_sup4096`.
   - It has the highest mean sequential IoU.
   - It improves MSE but appears most broadly overlapping.

Packaged test models:

```text
/home/josh/lerobot/outputs/train/realworld_v2
```

This package contains:

```text
realworld_v2/
  c001_sup4096/{003000,006000,009000,012000,015000}/
  c005_sup4096/{003000,006000,009000,012000,015000}/
  c001_sup8000/{003000,006000,009000,012000,015000}/
```

### Next Low-Cost Ablation

Before spending on another full pretrain, run sequential-only ablations from the best pretrain, likely `c001_sup8000`.

Primary sequential-side knobs:

```bash
--tfidf_top_t=256        # current 512
--tfidf_top_t=128        # stronger under-writing test
--memory_value_lr=0.0005 # current 0.001
--memory_value_lr_end=0.00005
```

Rationale:

- The new pretrains made routing broader and more plastic.
- The old write settings may now update too much of the shared read surface.
- Lowering `top_t` or memory LR should test whether the remaining failure is excessive write-time drift rather than insufficient separation.

Suggested first sequential-only tests:

1. `c001_sup8000` pretrain with `tfidf_top_t=256`, same LR.
2. `c001_sup8000` pretrain with `tfidf_top_t=128`, same LR.
3. `c001_sup8000` pretrain with `tfidf_top_t=512`, `memory_value_lr=0.0005 -> 0.00005`.

What to inspect before robot eval:

- Mean weighted IoU.
- L14 `t2 <- t0`.
- L14 `t4 <- t3`.
- L14 `t4 <- any prior`.
- Task 4 final MSE.
- Whether older-task proxies degrade less without making current-task MSE much worse.

### Possible Pretrain Ablation

If testing pretraining-side separation anyway, keep it controlled:

```bash
--policy.memory_layer.contrastive_loss_weight=0.01
--policy.memory_layer.routing_inter_task_separation_weight=0.35
--policy.memory_layer.routing_intra_task_max_support=4096  # or 8000
```

This should be treated as an ablation, not the main bet. The expected risk is worse useful sharing and worse pretrain fit. It should only be robot-evaluated if weighted read-through improves without a clear MSE/pretrain-quality regression.

---
## Entry 2 - 27 May 26 (Real-World v2 Failure Analysis + Config Mismatch)

### Context

After the Entry 1 follow-up batch, we trained a new v2 pair:

```text
/home/josh/lerobot/outputs/train/realworld_v2_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_8000_knn_36_50k
/home/josh/lerobot/outputs/train/realworld_v2_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_8000_knn_36_50k
```

Changes relative to v1:

- Pretrain dataset increased from 15 tasks to 25 tasks.
- Sequential task mixture changed.
- Sequential v2 task order:

| Task index | Instruction |
|------------|-------------|
| t0 | Put the mustard in the basket |
| t1 | put the red bow on the plate |
| t2 | Stack the baskets |
| t3 | Push over the white lego brick |
| t4 | Place the scredriver in the tub |

Robot testing outcome:

- Good: mustard in basket, white lego push.
- OK-ish: screwdriver in tub.
- Almost failed: stack baskets, red bowl on plate.
- Important observation: red bowl on plate was fine in v1.

### Important Correction: Real-World Did Not Actually Use The Sim Best Sequential Write Budget

We realised the real-world v2 run accidentally kept `tfidf_top_t=512`.

The intended plan was to apply the best current simulation operating point to real-world pi0.5. In the simulation research log, the best later setting was:

```text
layers=[8,10,12,14]
lora_rank=2
mem_knn=36
routing_loss_topk=36
routing_inter_task_separation_weight=0.25
routing_intra_task_locality_weight=0.25
routing_intra_task_min_support=128
routing_intra_task_max_support=2048
tfidf_top_t=1536
memory_value_lr=0.001 -> 0.0001
```

The v2 real-world script instead used:

```text
mem_knn=36
routing_loss_topk=36
routing_intra_task_max_support=8000
tfidf_top_t=512
memory_value_lr=0.001 -> 0.0001
```

So the strongest sim lesson - `knn=36` needs a larger matching write budget - was not actually tested in real-world v2.

This supersedes part of Entry 1. The earlier suggestion to test smaller `tfidf_top_t` was based on the real-world read-through concern before noticing that the v2 run had already under-shot the sim best write budget by 3x.

### Other Config Discrepancies From The Sim Best

Besides `tfidf_top_t`, the main differences are:

| Knob | Sim best / script | Real-world v2 |
|------|-------------------|---------------|
| `tfidf_top_t` | `1536` | `512` |
| `routing_intra_task_max_support` | `2048` | `8000` |
| `contrastive_loss_weight` | `1.0` | `0.01` |
| pretrain steps | `100000` | `50000` |
| pretrain warmup / decay | `10000 / 80000` | `4000 / 50000` |
| sequential batch size | `64` | `32` |
| eval env | configured in sim | absent in real-world |

Interpretation:

- `tfidf_top_t=512` is the clearest accidental mismatch.
- `routing_intra_task_max_support=8000` is also a meaningful difference: it allows much broader task routing than the sim best `2048`.
- `contrastive_loss_weight=0.01` is a deliberate real-world scaling choice, not necessarily a bug. Earlier real-world `c=1` made the contrastive term dominate the objective.
- `50000` pretrain steps may be too short for v2 because the dataset grew substantially.

### Wandb And Memory JSON Findings

Pretrain comparison:

| Run | Tasks | Frames | Effective epochs | Final MSE | Effnum | Used frac | Gate |
|-----|-------|--------|------------------|-----------|--------|-----------|------|
| v1 `c001_sup8000` | 15 | 267547 | about `6.0` | `0.0155` | `5718` | `0.128` | `0.936` |
| v2 `c001_sup8000` | 25 | 381156 | about `4.2` | `0.0188` | `6771` | `0.149` | `0.932` |

The v2 router/memory is broader and slightly more separated by routing similarity, but the action MSE is worse. This is consistent with undertraining after increasing the dataset size.

Sequential final comparison:

| Run | Final MSE | Effnum | Used frac | Gate | Mean weighted IoU |
|-----|-----------|--------|-----------|------|-------------------|
| v1 `c001_sup8000` | `0.0603` | `3144` | `0.121` | `0.964` | `0.0858` |
| v2 `c001_sup8000` | `0.0378` | `1524` | `0.082` | `0.954` | `0.0806` |

V2 sequential MSE is lower, but the memory read distribution is much narrower at the end despite the broader pretrain. The model still gates memory very heavily. This is a warning sign: rollout success is not tracking MSE directly, and the final adapted policy may rely on a smaller set of high-impact LoRA transforms.

Per-task sequential MSE:

| Run | Task | Start MSE | End MSE |
|-----|------|-----------|---------|
| v2 | mustard | `0.0784` | `0.0305` |
| v2 | red bowl | `0.0866` | `0.0378` |
| v2 | stack baskets | `0.1111` | `0.0591` |
| v2 | white lego | `0.0567` | `0.0203` |
| v2 | screwdriver | `0.0690` | `0.0378` |

Red bowl did not obviously fail to fit while it was the current task. Its end MSE is actually better than v1 red bowl. The likely failure is later drift / rollout brittleness rather than inability to learn the red bowl data.

Stack baskets is different. It has the worst current-task fit in v2 and then also collides strongly with screwdriver.

### Read-Through Diagnostics

The clearest harmful pair in v2 is stack baskets / screwdriver at L14:

| Read task | Updated-by task | L14 read-through |
|-----------|-----------------|------------------|
| screwdriver | stack baskets | `40.5%` |
| stack baskets | screwdriver | `27.2%` |
| screwdriver | any prior task | `60.8%` |

Other important L14 v2 read-through values:

| Task | Future read-through | Non-self read-through |
|------|---------------------|-----------------------|
| mustard | `10.3%` | `10.3%` |
| red bowl | `24.6%` | `30.1%` |
| stack baskets | `28.5%` | `49.1%` |
| white lego | `18.2%` | `45.6%` |
| screwdriver | `0.0%` | `60.8%` |

This supports two failure modes:

1. **Frozen-router generalization is not strong enough for arbitrary new tasks.**
   - The router learned a partition over the pretraining task manifold.
   - New tasks can land in shared routing basins.
   - TF-IDF can restrict writes, but it cannot stop old tasks from reading slots that later tasks update.

2. **Per-task adaptation capacity is too low under `knn=36, top_t=512`.**
   - `knn=36` creates broad read mixtures.
   - With `top_t=512`, sequential training updates too small a slice of the read footprint.
   - This is exactly the write-budget issue found in sim before `top_t=1536` became the best setting.

### Interpretation Of The Robot Results

The observed robot results are consistent with the diagnostics:

- Mustard works because it has strong support and low future read-through.
- White lego works because it is simpler and fits very well, even though it has some prior overlap.
- Screwdriver is OK-ish because it is the newest task and gets direct final adaptation, but it heavily reads stack-updated slots.
- Stack baskets fails because it is both hard to fit and strongly entangled with screwdriver.
- Red bowl likely learns initially but is brittle after future task writes. This explains why it could be fine in v1 but bad in v2.

### Updated Experimental Direction

The next direction is now revised. Initially, the conservative read was "do not remove the newly added v2 data yet" because v2 had two obvious confounds:

- the accidental `tfidf_top_t=512` mismatch;
- more pretrain data without more pretrain steps.

After further discussion, the more precise issue is that the newly added v2 pretrain tasks are not necessarily useful support for the sequential distribution. They were targeted at eliciting specific multi-strategy VLA behavior. That may be valuable for a different question, but for this continual-learning benchmark it can shape the router around regions that are not relevant to the held-out sequential tasks.

So the cleaner next primary experiment is to revert to the original 20-task real-world pool and create a new 15/5 split.

Proposed source-task split, using 1-indexed task numbers:

| Split | 1-indexed source tasks | 0-indexed source task ids |
|-------|-------------------------|----------------------------|
| sequential | `1, 2, 8, 11, 15` | `[0, 1, 7, 10, 14]` |
| pretrain | `3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20` | `[2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]` |

Local metadata for `outputs/realworld_all_tasks` gives this approximate scale:

| Split | Episodes | Frames |
|-------|----------|--------|
| pretrain | `753` | `271602` |
| sequential | `252` | `76062` |

This is close to the original v1 real-world scale, so `50k` pretrain steps is much less confounded than in v2. It also tests the router-coverage hypothesis more directly: the sequential tasks should be held out, but still live inside the old 20-task real-world distribution rather than next to unrelated multi-strategy additions.

The recommended first config:

```text
layers=[8,10,12,14]
lora_rank=2
contrastive_loss_weight=0.01
mem_knn=36
routing_loss_topk=36
routing_inter_task_separation_weight=0.25
routing_intra_task_locality_weight=0.25
routing_intra_task_min_support=128
routing_intra_task_max_support=4096
tfidf_top_t=1536
memory_value_lr=0.001 -> 0.0001
```

Rationale:

- `tfidf_top_t=1536` corrects the accidental real-world bug and matches the best sim lesson for `knn=36`.
- `contrastive_loss_weight=0.01` should stay fixed because high contrastive weight clearly distorted real-world pi0.5 training.
- `knn=36` should be kept, but only together with the larger write budget. The bad point was `knn=36, top_t=512`.
- `routing_intra_task_max_support=4096` is a plausible middle ground. With `mem_n_keys=384`, each head has `384^2 = 147456` slots, and `147456 / 4096 ~= 36` idealized task-sized support regions. This is not a hard allocation guarantee, because the support loss is an entropy pressure rather than a slot reservation, but it is a reasonable capacity prior.

If there is budget for one cheap sibling, reuse the same pretrain and run a lower sequential write LR:

```bash
--tfidf_top_t=1536 --memory_value_lr=0.0005 --memory_value_lr_end=0.00005
```

This tests whether the larger write budget needs smaller update magnitude to avoid red-bowl-style future drift.

Implementation note: after creating the new pretrain and sequential datasets, the task ids inside each split may be remapped to local contiguous ids. That is fine for training, but the source 20-task ids should be recorded in the dataset names, script comments, or a split manifest to avoid confusing this run with the older `realworld_pretrain` / `realworld_seq` split.

---
## Entry 3 - 2 Jun 26 (v3 Analysis + Direct Forgetting Eval: Concentration vs Spread)

### Context

v3 is the Entry 2 recommendation actually executed: the sim-best operating point ported to real-world on a clean 15/5 split of the original 20-task pool. It fixes all three v2 confounds at once.

```text
/home/josh/lerobot/outputs/train/realworld_v3_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k
/home/josh/lerobot/outputs/train/realworld_v3_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k
```

Config verified from the checkpoints (the key point is that the v2 write-budget bug is gone):

| Knob | v2 | v3 |
|------|----|----|
| `tfidf_top_t` | `512` (the bug) | **`1536`** (= sim best) |
| `routing_intra_task_max_support` | `8000` | `4096` |
| pretrain pool | 25 tasks (multi-strategy) | 15 held-out from the 20-pool |
| pretrain effective epochs | `4.20` | `5.87` |
| `mem_knn` / `routing_loss_topk` | `36 / 36` | `36 / 36` |
| `contrastive_loss_weight` | `0.01` | `0.01` |
| `lora_rank`, batch, grad_accum | `2`, `32`, `1` | `2`, `32`, `1` |

So v3 simultaneously corrects: the accidental `top_t=512`, the undertraining (25 tasks at 50k = 4.2 epochs -> 15 tasks at 50k = 5.9 epochs), and the irrelevant multi-strategy pretrain pool. The v2 problem tasks (stack baskets, screwdriver) are now in **pretrain**, not the sequential benchmark.

### v3 Sequential Task Set (new 15/5 split)

| Task index | Instruction |
|------------|-------------|
| t0 | Put the mustard in the basket |
| t1 | Put the red bowl on the plate |
| t2 | Push over the white lego brick |
| t3 | Remove the blender lid and place it on the red plate |
| t4 | Place the red brick in the tub |

### Pretrain Comparison

| Metric | v2 (`c001_sup8000`) | v3 (`c001_sup4096`) |
|--------|---------------------|---------------------|
| Effective epochs | `4.20` | `5.87` |
| Final MSE | `0.0188` | `0.0195` |
| Gate mean | `0.932` | `0.941` |
| `effnum_mean` | `6771` | `6428` |
| Inter-task routing similarity | `0.0553` | `0.0609` |
| `routing_intra_task_support_mean` | `4727` | `5348` |

Pretrain MSE is not directly comparable (different task mixtures). The meaningful fact is that v3 trains ~5.9 epochs (close to v1's 6.0), removing the v2 undertraining confound, while keeping routing well separated (similarity ~0.06; ~1.0 would be identical).

### Sequential Memory Dynamics

| Metric | v2 | v3 |
|--------|----|----|
| Final-task MSE | `0.0378` | `0.0355` |
| `effnum_mean` | `1524` | `1966` |
| `effnum_L14` | `2319` | **`3675`** |
| Gate mean | `0.954` | `0.958` |
| Mean weighted read IoU (all layers) | `0.0806` | **`0.0515`** |
| L14 weighted IoU mean / max | `0.0915` / `0.2116` | `0.0689` / `0.1620` |

Two clear improvements: the larger write budget broadens the final effective memory usage (`effnum_L14` `2319 -> 3675`), directly addressing the Entry 2 warning that v2's final policy leaned on a narrow set of high-impact LoRA transforms; and read overlap drops substantially (mean IoU `-36%`, worst-pair L14 IoU `0.212 -> 0.162`).

### Read-Through Diagnostics (the important nuance)

Recomputed with the same definitions as Entry 1/2 (my numbers reproduce the Entry 2 v2 values exactly: screwdriver `<- stack 40.5%`, screwdriver `<- any prior 60.8%`, etc.).

L14 future read-through (fraction of an earlier task's read weight on slots later tasks overwrite = forgetting exposure):

| | t0 | t1 | t2 | t3 | mean(t0-t3) |
|--|----|----|----|----|-------------|
| v2 | `10.3` | `24.6` | `28.5` | `18.2` | `20.41%` |
| v3 | `18.1` | `45.4` | `13.8` | `5.5` | `20.70%` |

Key finding: **weighted IoU improved a lot, but aggregate read-through did not.** The plasticity<->drift tradeoff is the reason: `top_t=1536` writes ~3x more slots per task, so the union of later-task updates re-covers the more-separated read footprints. v3's worst single pair is **red-bowl (t1) <- blender-lid (t3) = 41.9% at L14** (both are "place X on the (red) plate"), comparable to v2's worst (screwdriver <- stack `40.5%`). The collision moved to a *semantically coherent* pair and **concentrated** onto one task (t1=45%), whereas v2 spread it (24/28/18).

### Per-Task Fit (current-task MSE at end of each task window)

| | t0 | t1 | t2 | t3 | t4 |
|--|----|----|----|----|----|
| v2 | `0.0305` | `0.0378` | `0.0591` | `0.0203` | `0.0378` |
| v3 | `0.0295` | `0.0356` | `0.0325` | `0.0426` | `0.0355` |

v3 has no hard-to-fit outlier; v2's stack-baskets (`0.0591`) was the clear weak point. (These are the **only** rollout-free signals the run logs; there are still no eval rollouts for real-world.)

### Direct Forgetting Eval (new method, the decisive result)

Since read-through is *direction-blind* (it cannot tell destructive overwrite from useful shared-primitive refinement), we measured forgetting directly: load all 5 sequential checkpoints and recompute each task's flow-matching MSE under each checkpoint (paired noise seeding; harness reproduces the logged training losses on the diagonal). Each cell = task MSE under that checkpoint; **bold** = just-trained.

v3 (`top_t=1536`):

| model after | t0 | t1 | t2 | t3 | t4 |
|-------------|----|----|----|----|----|
| t1 (6k) | 0.032 | **0.046** | 0.44 | 0.46 | 0.62 |
| t2 (9k) | 0.032 | 0.046 | **0.029** | 0.46 | 0.61 |
| t3 (12k) | 0.032 | **0.142** | 0.029 | **0.037** | 0.61 |
| final (15k) | 0.032 | **0.143** | 0.029 | 0.038 | **0.037** |

v2 (`top_t=512`):

| model after | t0 | t1 | t2 | t3 | t4 |
|-------------|----|----|----|----|----|
| final (15k) | 0.036 | 0.043 | **0.079** | 0.035 | 0.035 |

Forgetting summary (trained -> final):

| run | profile | per-task forgetting |
|-----|---------|---------------------|
| v2 | **spread** | mustard +14%, red bow +11%, **stack +60% (->0.079)**, white lego +27%; none pristine |
| v3 | **concentrated** | **red bowl +209% (0.046->0.143)**; mustard/white-lego/blender-lid all +0-2% (pristine) |

The read-through metric predicted the worst offender in both runs (v3 red bowl 45% -> +209%; v2 stack 28% -> +60%), and every low-read-through task was retained. So forgetting here is **threshold-y and concentrated**, not the smooth ~20% mean - what matters is the worst task's concentrated exposure, not the average.

### The sharing is real *and* destructive (resolves the interference-vs-reuse question)

From the same run: training red bowl (t1) **lowered blender-lid's loss before blender-lid was ever trained** (t3 column: `0.63 -> 0.46`). That is genuine forward transfer through the shared "place-on-red-plate" slots - the reuse we want. But when blender-lid then trains, it re-specializes those same rank-2 slots to its own instantiation and overwrites red bowl's version (red bowl jumps `0.046 -> 0.142` exactly at the t3 window). Slot-level: red bowl's top 1% of L14 slots carry 62% of its read weight; t3 updates 66% of red bowl's top-50 L14 slots (94% at L12), hard (median ~427/3000 updates at L14, ~1036/3000 at L12).

Conclusion: **a rank-2 LoRA slot cannot co-host two tasks' specializations of a shared primitive**, so under frozen-router sequential training the shared region becomes a zero-sum overwrite and the frozen earlier task loses. Therefore the right lever is **NOT more routing separation** (which would destroy the transfer) - it is **per-slot co-hosting capacity (raise `lora_rank`)**, or history-aware write-protection of prior tasks' hot slots.

### Caveat: MSE forgetting != robot success

Cross-checking against the v2 robot results: stack-baskets "almost failed" and has the worst MSE forgetting (+60%) - consistent. But v2 red bow also "almost failed" yet its MSE forgetting is mild (+11%) - so that failure was task brittleness / marginal fit, not catastrophic forgetting. Treat the MSE matrix as necessary-but-not-sufficient. Implication for v3: if red bowl fails on the robot, this matrix says it will be *genuine forgetting* (caused by blender-lid), a more targetable failure than v2's red bow.

### New tooling: `--eval.type=loss`

Added loss-based eval to `lerobot-sequential-train` so this forgetting matrix is produced *in-run* (no post-hoc checkpoint reloading). After each task it recomputes MSE on all seen tasks and logs per-task loss + forgetting (vs the task's just-trained baseline) to wandb (`eval/loss/task_*`, `eval/forgetting/task_*`, `eval/avg_forgetting_prior`), `eval/loss_results.jsonl`, and a cumulative chart.
- `--eval.type` in {`env` (default, rollouts), `loss`, `none`}; `--eval_loss_n_batches` (default 20).
- Code: `EvalConfig.type` (`configs/default.py`); `_eval_loss_on_seen_tasks` / `_render_loss_eval_chart` / `_append_loss_results_jsonl` + the eval branch in `scripts/lerobot_sequential_train.py`. Backward compatible (default `env` runs the unchanged rollout path).
- Add `--eval.type=loss --eval_loss_n_batches=20` (and drop `--env.*`) to the real-world sequential scripts to get retention curves live.

### Answers to the Three Questions

1. **Does v3 look better than v2?** Yes on every controllable dynamic: fixed the `top_t` bug, fixed undertraining (5.9 vs 4.2 epochs), fixed the pretrain-pool relevance, broadened final memory usage (`effnum_L14` +58%), lowered read overlap, slightly better final-task fit, and removed the hard-to-fit outlier.
2. **Are we addressing the v2 issues?** Mostly. The config/undertraining/pool issues are fixed. The forgetting channel is *re-shaped, not eliminated*: v3 retains 4/5 tasks near-perfectly but concentrates the damage into one severe collision (red bowl), whereas v2 degrades all tasks mildly.
3. **Stronger performance than v2?** Likely yes by typical task (4 reliable vs 5 mediocre), worse by worst-case (red bowl 0.143 > stack 0.079). Net better *if* the one collision is closed - and because the collision rides on real transfer, the lever is rank, not separation.

### Next Steps

1. **Rank-4 sequential rerun** from the same v3 pretrain, then re-run the forgetting matrix (now just `--eval.type=loss`). Prediction to falsify: red bowl's `0.046 -> 0.143` jump shrinks substantially while blender-lid keeps the forward-transfer benefit. If red bowl still collapses at rank 4, the problem is deeper than co-hosting capacity and reorder/replay becomes the conversation.
2. Robot test priorities: mustard / white lego / blender-lid / red brick should be solid; **red bowl is the predicted weak point** (genuine forgetting from the later red-plate task).

### Update (2 Jun 26) — currently running: v4 task-swap POC

Decided against the rank-4 idea (next step #1 above) for now: with the router frozen and only values trainable, there is no mechanism allocating sub-dimensions per task — each task's gradient fills the whole rank at a shared slot, so the later task overwrites the earlier one at rank 4 just as at rank 2. Rank changes per-slot capacity, not allocation.

Instead, banking a clean POC first by **curating the tasks** so no two share a primitive. Swap: move "Remove the blender lid and place it on the red plate" into pretrain, and bring "Place the grey water bottle in front of the red water bottle" into the sequential set (taking the t3 slot). This removes the only dangerous pair (red-bowl <- blender-lid was the sole >9% future read-through; everything else was clean), so the prediction is ~5/5 retention.

- New datasets (merged from `outputs/vla-wm-real` task folders via `merge_datasets.py`): `realworld_pretrain_v4` (15 tasks, 753 eps / 268k frames) and `realworld_sequential_v4` (5 tasks, 252 eps / 79k frames). v4 sequential order: mustard, red bowl, white lego, **grey water bottle**, red brick.
- Script: `job_scripts/nebius/combined/realworld_v4_pi05_4_layer_film_lora_2_sample_contrastive_4k_50k.sh` — v3 configs verbatim, v4 datasets, plus `--eval.type=loss --eval_loss_n_batches=20` on the sequential stage (forgetting matrix logged live, no post-hoc checkpoint eval).
- Launched 2 Jun (tmux `v4train`, log `outputs/v4_train.log`). Pretrain ~48h, then sequential auto-runs.

The "real" separation lever (deferred, for the collision case rather than this POC) is **collision-aware write protection**: extend the sequential write mask to also exclude prior sequential tasks' top read-weight slots. Note this is *not* what online IDF already does — IDF down-weights by pooled, task-agnostic batch-frequency (DF) and is overridable by the current task's TF, so a 2-task collision (low DF, high current demand) slips through; that is exactly why red bowl was overwritten *with* online IDF enabled. A prior-task read-footprint mask is task-identity-aware and can hard-veto.
