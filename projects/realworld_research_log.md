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
