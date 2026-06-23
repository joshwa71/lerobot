# Research Log - VLA Memory

---
## Entry 0 (context / progress so far)
- Goal: reduce catastrophic forgetting in sequential training using PKM-style memory layers on SmolVLA (continual multi-task adaptation).
- Found: **standard PKM with static value vectors** could achieve ~zero interference, but performance plateaued (~30% success/task) → likely insufficient expressivity.
- Switched to **PKM “values” as mixture of tiny LoRAs** (`value_type=lora`, rank 2–4). Capacity constraints (fewer slots) coincided with interference returning; later also observed interference even when slot count matched.
- Tried mitigations:
  - **Query contrastive loss** (centroid) to separate routing → little benefit.
  - **Slot output corruption** for robustness to drift → limited benefit.
  - Started experimenting with **seeding sequential TF‑IDF/IDF from pretrain memory usage stats**.
- Off the table: hard task masking / task-boundary slot allocation, EWC, replay.

## Entry 1 — 11 Feb 26 (analysis + next experiments)
### Summary of the problem
- Sequential training updates only memory values/LoRA params (router frozen). We observed **high slot overlap at read-time** and a **shared “hot core”** used by all tasks (router collapse / MoE-like load imbalance).
- TF‑IDF masking mostly prevents *writing* into that shared core, but tasks still *read* heavily from it, and overlap remains large outside the strict all-task intersection.

### Evidence from sequential logs (tasks 6–9; layers 14 & 15)
- Access overlap (intersection/min) across tasks (treating `(layer,slot)` as distinct): ~**0.62–0.72**.
- Update overlap is lower with TF‑IDF: ~**0.16–0.31** (intersection/min).
- Shared hot core exists:
  - Layer 14: intersection of top-512 accessed across tasks = **134 slots**, contributing **~26–34%** of all accesses per task.
  - Layer 15: intersection of top-512 accessed across tasks = **172 slots**, contributing **~43–48%** of all accesses per task.
- TF‑IDF mostly protects the core (fraction of updates landing in access-core top-512 intersection is small, typically **≤3%**; and intersection of top-512 updated slots across all 4 tasks is **empty**).

### Design decision / hypothesis
- Forgetting is driven less by “writing into the global core” (TF‑IDF already reduces this) and more by **router collapse / shared read-time dependence** + overlap in the non-core set (worse with LoRA-values since shared slots implement shared transforms).
- Aim: keep a small universal core but increase **conditional routing separation** (task/instruction-conditioned hot sets), mainly during pretraining (contrastive is ineffective in single-task sequential batches).

### Next experiment being run (pretraining)
- Script: `job_scripts/smolvla-memory/pretrain/2_layer/pretrain_2_wide_film_dropout_high_lora_4_sample_contrastive_0.5.sh`
- Key knobs:
  - Memory on expert layers **[14,15]**, `value_type=lora`, `lora_rank=4`
  - `mem_n_keys=384`, `mem_heads=4`, `mem_knn=16`, `mem_k_dim=512`
  - Language-conditioned queries: `lang_to_query=true`, `fuse_method=film`, `embedding_model=all-mpnet-base-v2`
  - Increased exploration: `dropout_prob=0.1`
  - Stronger separation objective: `contrastive_method=sample`, `contrastive_loss_weight=0.5`

---

## Entry 2 - 25 Feb 26 (Simplified Experiments)

### Summary of the problem
Previously I was not systematically investigating the effects of each new feature on the performance. I have now run ablations across:
- Contrastive loss type (centroid vs sample-wise)
- Contrastive loss weight (0.5, 1, 2)
- Dropout probability (0.05, 0.1, 0.2)

Expirments run with Lora=4 and layers 12 and 14.

### Observations
- **Contrastive loss type:** No clear winner here. Centroid contrastive appears to exhibit less interference at later layers (e.g. layer 14) but slighly more in earlier layers. This is interesting - I thought we'd expect strictly more intersection from centroid.
- **Contrastive loss weight:** We appear to see the largest intersection for 0.5 (as expected), then significantly reduced intersection for 1.0, then oddly increased intersection for 2.0. Non-monotonic increase is confusing.
- **Dropout probability:** 0.05 experiment still running, but 0.2 performs better than 0.1 in terms of interference. Appears less degredation in previous task per, less intersection. This makes sense as some attempts to access slots important for previous tasks will be nullified by higher dropout.

### Thoughts
- Claude tells me that for sample-wise contrastive loss we are including some terms which encourage uniformity of the queries inside a given task **as well** as distance across tasks. This could be problematic if we saturate the query space. Could explain the non-monotonic behaviour of the sample contrastive loss weight. To test we are trying a loss that removes the positive pairs from the denominator.
- I also want to test the change caused by the lora r. Testing 2 to see if it makes any difference.
- I also want to test the effect of initialising the idf stats with the pretraining stats with different weights. Testing denom = 16 (e.g. 2x a single sequential task), 33 (1x seq task), 66 (0.5x seq task)

### Future
- In future I should test the effect of batch size. Due to memory constraints I'm limited in the batch size I can use for lora=4 which could affect the training dynamics of the query projections if not all the tasks are in the batch. Perhaps worth adding gradient checkpointing or something so I can increase batch size.
- Test corruption again. Seemed promising.

---

## Entry 3 - 4 Mar 26 (Small Test: Cross-Batch Contrastive Queue)

- Added a small pretraining test to increase effective contrastive pool without increasing micro-batch memory: cross-batch query FIFO (`contrastive_query_queue=2048`).
- Goal: improve sample-wise contrastive negatives/positives under `batch_size=32` by reusing recent query embeddings from prior batches.
- Using layers **[12,14]** in the new script:
  - `job_scripts/smolvla-memory/pretrain/2_layer/contrastive_accumulation/pretrain_12_14_film_lora_2_sample_contrastive_1.sh`
- Added matching sequential script for this run family (same sequential setup pattern):
  - `job_scripts/smolvla-memory/sequential/2_layer/contrastive_accumulation/sequential_12_14_film_lora_4_sample_contrastive_1.sh`

---

## Entry 4 - 4 Mar 26 (Dynamics Notes + Next Test: r=1 with More Slots)

### Updated dynamics interpretation
- There is a consistent tradeoff between expressivity and stability in the value parameterization:
  - **Static value vectors** gave lower interference but lower ceiling performance.
  - **LoRA values** increased current-task performance but also increased forgetting.
- Working hypothesis: LoRA slot overlap is more destructive than vector overlap because each slot is a **transform** of the hidden state, not just an additive template. When shared slots are updated by a later task, those updates can alter behavior broadly for prior tasks.
- This is compatible with the observed pattern where TF-IDF reduces direct write overlap but forgetting still appears once additional tasks are introduced (especially by task 4): the model still reads overlapping regions, and overlapping LoRA slots are high-impact.

### Why TF-IDF may still be insufficient
- Current online masking is based on **frequency of slot access per batch** (TF over counts), which can still repeatedly prioritize globally frequent shared slots.
- IDF helps, but if TF dominance is strong, medium-importance task-specific slots may still be under-updated.
- In other words, masking reduces some overwrite, but not necessarily enough to prevent drift in shared high-impact LoRA slots.

### Suggested mitigation directions (non-EWC / non-replay / non-hard-mask)
- Use **contribution-weighted TF** (weight by retrieval weights) instead of pure access counts so low-weight incidental touches do not dominate slot selection.
- Use a **saturating TF transform** (e.g., sqrt/log scaling) before TF-IDF ranking to reduce repeated wins by the same hot slots.
- Consider **per-head update budgeting** to reduce global head collapse where dominant heads consume most of the update budget.
- Add **soft plasticity decay** per slot (continuous reduction in update magnitude for heavily updated slots, not hard exclusion).
- Keep query/router adaptation offline (pretraining) and avoid online query updates if they destabilize old-task routing.

### New experiment being launched
- We are now testing **LoRA rank 1 with more memory slots** to probe whether many weaker experts are more stable than fewer stronger experts.
- To approximately double slot count relative to `mem_n_keys=384` (slots = `n_keys^2`), we set:
  - `mem_n_keys=544` (`544^2` is ~2x `384^2`)
- New scripts:
  - `job_scripts/smolvla-memory/pretrain/2_layer/lora_r_exp/pretrain_12_14_film_lora_1_2xslots_sample_contrastive_1.sh`
  - `job_scripts/smolvla-memory/sequential/2_layer/lora_r_exp/sequential_12_14_film_lora_1_2xslots_sample_contrastive_1.sh`

---

## Entry 5 - 6 Mar 26 (Routing Compactness + Corruption Fix)

- Recent discussion clarified two separate ideas:
  - **Weighted TF-IDF** for sequential updates: rank slots by retrieval-weighted contribution rather than raw access counts.
  - **Routing regularization** during pretraining: act on actual PQ sub-key usage, not just query embeddings.
- Key point: more diverse queries do **not** necessarily imply more diverse slot accesses. Different queries can still route into the same hot late-layer region, so query contrastive is only an indirect proxy for what we care about.
- Current hypothesis:
  - within a task, routing should stay fairly localized
  - across tasks, those localized routing regions should differ
- To test this, I initially added a one-sided **routing compactness** loss on PQ marginals with CLI control via `routing_compactness_weight` (default `0.0`).
- Initial compactness-control scripts were created under:
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_locality_exp/`
  - `job_scripts/smolvla-memory/sequential/2_layer/routing_locality_exp/`
- After further discussion, the more important anti-forgetting term is **routing separation**, not compactness alone:
  - compactness says each task should route locally
  - separation says different tasks should route to different regions
- So compactness is now being treated as a **secondary control ablation**, while the main next test is a routing-separation sweep.
- Initial routing-separation scripts were created under:
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_inter_task_separation_exp/`
  - `job_scripts/smolvla-memory/sequential/2_layer/routing_inter_task_separation_exp/`
- Also fixed the LoRA corruption path so corruption is applied to the **adapter output before the shared gating/aggregation path**, which is a better match to the failure mode we want to test than corrupting the low-rank hidden activations.

---

## Entry 6 - 7 Mar 26 (Routing Loss Interpretation + New Sweep)

- We discovered that the logged `routing_separation_mean` metric was actually the **mean pairwise cosine similarity** between task routing distributions, not a higher-is-better separation score.
  - `~0.998` means tasks are routing almost identically.
  - `~0.02` means tasks are routing very differently.
- In the first routing-separation sweep, `0.1` was too weak to change routing much, while `0.5` and `1.0` drove the similarity way down but also collapsed **intra-task routing entropy** to about `0.10` and `0.06`.
- That means the separation term was working, but it was achieving separation by pushing each task toward an almost one-hot PQ subkey pattern. This is too sharp for the intended behavior.
- Main adjustment:
  - replace the old one-sided “compactness” term with an explicit **intra-task locality** loss
  - define locality as a **support/entropy band**, so it penalizes routing that is both too diffuse and too concentrated
  - keep **inter-task separation** as a separate objective
- CLI has been updated to reflect this distinction:
  - `routing_intra_task_locality_weight`
  - `routing_inter_task_separation_weight`
  - `routing_intra_task_min_support`
  - `routing_intra_task_max_support`
- New planned sweeps:
  - **Intra-task locality sweep** with support band `8-32` and locality weights `0.1 / 0.25 / 0.5`
  - **Inter-task separation sweep** with locality fixed on (`weight=0.25`, support band `8-32`) and separation weights `0.15 / 0.25 / 0.35`
- Goal of the new sweep: find a regime where tasks separate in routing space **without** collapsing each task onto 1-2 subkeys per PQ half.

---

## Entry 7 - 9 Mar 26 (Routing Loss Misalignment Diagnosis + Joint-Slot Fix)

### Findings from the Entry 6 sweep

Ran 3 locality sweeps (`locality_weight` 0.1 / 0.25 / 0.5, support band `[8, 32]`) and 3 separation sweeps (`sep_weight` 0.15 / 0.25 / 0.35, locality 0.25, support `[8, 32]`). Results:

1. **Sequential training performance did not improve.** Best separation run (sep=0.15) tied with corruption baseline at ~16.5% success.
2. **The routing loss was operating on the wrong distribution.** The `_compute_routing_losses` method computed soft distributions over all `n_keys=384` subkeys per PQ half, then measured entropy and pairwise similarity on those half-marginals. But retrieval only uses the **top-k** subkeys and forms **Cartesian-product joint slots**. The loss could be satisfied by moving tail mass in the half-distributions without changing the top subkeys or the final retrieved slots.
3. **Evidence of misalignment:**
   - Half-level support was broad (routing_intra_task_support_mean ~216–384) even as the **actual final-slot effective number** in layer 14 was very concentrated (effnum ~30–82 for sep 0.15/0.25).
   - sep=0.25 drove half-similarity down, but layer-14 weighted access IoU between tasks stayed at ~0.316. It achieved "separation" by peeling one task off onto a different tiny hot core, while the other 3 tasks still shared.
4. **Task 7 "zero overlap" anomaly:** In sep=0.25, task 7 was isolated onto a different layer-14 core (weighted IoUs ~1e-5 against other tasks) while tasks 6/8/9 still shared heavily (IoUs 0.60–0.69). This is a pathological asymmetric solution allowed because the pairwise loss has no global-balance pressure.
5. **Last-task slot-usage instability:** Task 9's slot-usage metric oscillated between ~0.020–0.037 across steps. Since the router is frozen during sequential training, this is **batch/episode heterogeneity** within task 9, not learning instability.
6. **Logging note:** The sequential `train/loss` includes routing auxiliary terms even though they have no gradient path to the trainable value params. `mse_loss` is the meaningful optimisation signal.

### Root cause

The routing regulariser operated on the two PQ half-distributions separately (`s1_full`, `s2_full`, each `n_keys`-way). But:
- Retrieval takes top-k in each half, forms the k×k Cartesian product, then selects top-k final slots.
- The loss on soft half-marginals is a **weak proxy**: it can be cheaply satisfied by reshuffling tail mass while the actual top subkeys (and final slots) stay shared.
- The support band `[8, 32]` targeted half-subkey support, not final-slot support. With `n_keys=384`, these are completely different scales.

### Solution implemented

Rewrote `_compute_routing_losses` in `memory_lite.py` to operate on the **joint product-key candidate distribution**:
1. Take top-M subkeys per PQ half (M = `routing_loss_topk` or `knn`, default matches retrieval)
2. Form M×M Cartesian-product candidate scores and slot IDs
3. Softmax over those M² joint candidates per sample per head
4. Scatter per-task distributions into a compact slot histogram (slot IDs remapped via `searchsorted` for memory efficiency)
5. Compute locality (entropy band) and separation (cosine similarity) on those **full-slot distributions**

This directly regularises the distribution retrieval actually uses. Gradient flows through: loss → histogram → joint softmax → joint scores → top-k values → PQ half scores → query projection + keys.

Also fixed a NaN gradient bug: `torch.where(p > eps, p.log(), 0)` computes `log(0)` gradients for the unused branch. Replaced with `p.clamp(min=eps).log()`.

Added `routing_loss_topk` config param (default 0 = use `knn`). Support bounds now refer to **effective final slots** in the `n_keys²` space, not half-subkey support.

### Support band rationale

Old band `[8, 32]` was in half-subkey space over 384 subkeys. New band is in final-slot space over 147,456 (`384²`) slots. With 35 pretrain tasks:
- Want to allow **generalist slots** (shared priors across tasks), so don't need complete separation
- Want to prevent **collapse** onto a tiny hot core (the pathology we observed)
- Uniform partition: 147K / 35 ≈ 4,200 slots/task, so even max_support=2048 is well within budget

### Next experiments

**Locality sweep** (locality_weight=0.25, no separation loss):
- `[64, 512]` — moderate band
- `[64, 1024]` — recommended default
- `[128, 2048]` — broad, allows generous generalist core

**Separation sweep** (locality_weight=0.25, support `[128, 2048]`):
- `sep_weight` = 0.15 / 0.25 / 0.35

Scripts under:
- `job_scripts/smolvla-memory/pretrain/2_layer/routing_locality_exp/`
- `job_scripts/smolvla-memory/pretrain/2_layer/routing_inter_task_separation_exp/`
- Matching sequential scripts in `sequential/2_layer/` directories

---

## Entry 8 - 12 Mar 26 (Joint-Slot Separation Results + Next Steps)

### Results from Entry 7 experiments

Ran 6 pretraining runs (3 locality-only, 3 locality+separation) and 5 sequential runs (sep_0.25 sequential added later). All use layers [12,14], LoRA rank 2, support band [128, 2048], locality_weight=0.25.

#### Pretraining eval success (libero_spatial, 4 episodes):

| Run | Eval % | MSE (final) |
|-----|--------|-------------|
| loc [64,512] | 77.5 | 0.0133 |
| loc [64,1024] | 70.0 | 0.0117 |
| loc [128,2048] | 77.5 | 0.0119 |
| sep=0.15 | 80.0 | 0.0117 |
| **sep=0.25** | **90.0** | **0.0112** |
| sep=0.35 | 80.0 | 0.0133 |

sep=0.25 is the best pretrained model by a clear margin.

#### Sequential training (tasks 6-9, avg_pc_success_seen after 4 tasks):

| Run | Success % | Weighted IoU | Task 9 reads from 8's updates (L12/L14) |
|-----|-----------|--------------|------------------------------------------|
| loc [64,512] | 12.0 | 0.263 | — |
| loc [64,1024] | 10.0 | 0.242 | — |
| loc [128,2048] | 16.5 | 0.204 | 37% / 72% |
| sep=0.15 | 30.5 | 0.057 | 18% / 16% |
| **sep=0.25** | **34.5** | **0.044** | **15% / 13%** |
| sep=0.35 | 26.5 | 0.041 | 15% / 13% |

### Key findings

**1. The joint-slot routing fix (Entry 7) works.** Unlike the half-distribution approach (Entry 6) which achieved spurious separation by reshuffling tail mass, the joint-slot loss produces genuine routing differences that translate to less forgetting. sep=0.25 achieves 34.5% vs 16.5% for the best locality-only run.

**2. Separation loss fixes the L14 collapse.** Without separation, L14 concentrates into ~400 effective slots (effnum) vs L12's ~2500. With separation, both layers equalize at ~3000-4000. The L14/L12 effnum ratio goes from 0.16 to ~1.10. This was the key pathology from Entry 7 — now resolved.

**3. Write overlap is essentially solved.** TF-IDF masking (top_t=512) combined with separation gives near-zero write overlap between tasks. Pairwise update-set IoU is 0-2% for sep runs (vs up to 53% of task 6's updated slots overwritten by task 8 in locality-only at L14).

**4. The remaining interference is read-time.** Task 9 reads 13-16% of its retrieval weight through slots that task 8 modified. The tasks read from different slot regions (weighted IoU ~0.044), but LoRA value updates to any shared-read slot change the transform seen by all tasks that read from it. This is the dominant remaining forgetting channel.

**5. Binary slot overlap vs weighted overlap — a subtlety.** Raw set overlap is high for sep runs (~80-90% of slots touched by every task) because routing is diffuse. But the access weight distribution is highly concentrated: 50% of all retrieval weight sits in ~1,400 slots (0.94% of total) for sep_0.15 L14. The weighted IoU (which uses access counts, not binary presence) is the meaningful interference metric. The long tail of incidentally-touched slots carries negligible weight.

**6. Per-task expressivity is the other bottleneck.** Sequential per-task MSE converges to 0.058-0.091 vs pretrain MSE of 0.011. Each LoRA slot is a rank-2 transform of the 720-dim expert hidden state. With top_t=512 and ~2,000-3,000 unique slots updated per task, effective capacity is ~7.2M params per task — but the rank-2 constraint severely limits per-slot expressivity. Task 8 is consistently the hardest (MSE ~0.089-0.091 across all runs).

**7. sep=0.25 is the sweet spot.** Higher separation (0.35) achieves even lower IoU (0.041 vs 0.044) but lower performance (26.5% vs 34.5%) due to worse pretrain quality. Lower separation (0.15) has slightly higher IoU but also lower pretrain quality (80% vs 90%). sep=0.25 balances routing separation with pretrain fit.

### Challenges remaining

Two distinct problems limit performance:
1. **Forgetting (read-time interference):** 13-16% of read weight flows through modified slots. Separation + TF-IDF solved write overlap but can't prevent tasks from reading modified slots.
2. **Per-task fit (expressivity ceiling):** Rank-2 LoRA at 2 layers gives each task ~7.2M effective params. Per-task MSE is 5-8x higher than pretrain. More capacity is needed.

### Next experiments

**Batch 1 — Isolation of forgetting vs capacity interventions (2-layer pretrains):**

3 pretraining runs combining sep=0.25 with corruption noise to address read-time robustness:
- `corruption_prob=0.05, 0.1, 0.2` (all with `corruption_std=0.1`)
- Corruption adds Gaussian noise to retrieved LoRA outputs during pretraining, teaching the model to tolerate the kind of value drift that occurs when other tasks update shared-read slots.
- The 0.1 corruption_prob is calibrated to roughly match the ~13% read-from-written interference measured at L14.
- Scripts: `job_scripts/smolvla-memory/pretrain/2_layer/sep_and_corruption/`

**Batch 2 — Capacity via depth (3-layer pretrains):**

3 pretraining runs adding layer 10 to the existing [12,14] setup, with sep=0.25:
- Adds ~50% more effective capacity per task (3 layers × ~2,500 unique updated slots each)
- Interference budget is per-layer, so a third layer doesn't compound interference
- Will test whether the per-task MSE gap (0.06-0.09 vs pretrain 0.011) narrows with more depth
- Scripts: `job_scripts/smolvla-memory/pretrain/3_layer/separation/`

**Quick test — IDF seeding from pretrain stats:**

1 sequential run using the existing sep=0.25 pretrained checkpoint with `idf_stats_path` pointing to its `memory_usage.json` and `idf_stats_denom=33` (pretrain prior worth ~1 sequential task). This tests whether down-weighting globally popular pretrain slots in TF-IDF reduces the remaining write-to-read interference. No new pretraining needed.

---

## Entry 9 - 16 Mar 26 (3-Layer & Corruption Results + Top-T Sweep)

### Results from Entry 8 experiments

Ran 3 pretraining + sequential pairs for **3-layer [10,12,14]** (separation sweep: 0.15/0.25/0.35) and 3 pairs for **corruption** (prob 0.05/0.1/0.2 on 2-layer [12,14] with sep=0.25). IDF seeding from pretrain stats tanked performance and is omitted.

Baseline for comparison: 2-layer [12,14], sep=0.25, no corruption (from Entry 8): pretrain MSE 0.0112, sequential 34.5%, IoU 0.044.

#### Pretrain summary

| Run | MSE | Gate L10 | Gate L12 | Gate L14 | Effnum |
|-----|-----|----------|----------|----------|--------|
| **Baseline (2L)** | **0.0112** | — | 0.74 | 0.54 | 3257 |
| 3L sep=0.15 | 0.0086 | 0.36 | 0.49 | 0.45 | 3069 |
| 3L sep=0.25 | 0.0086 | 0.42 | 0.54 | 0.43 | 2675 |
| 3L sep=0.35 | 0.0086 | 0.46 | 0.57 | 0.44 | 2404 |
| corr=0.05 | 0.0116 | — | 0.76 | 0.55 | 3218 |
| corr=0.1 | 0.0128 | — | 0.75 | 0.55 | 3217 |
| corr=0.2 | 0.0122 | — | 0.75 | 0.56 | 3232 |

#### Sequential eval progression (avg_pc_success_seen after each task)

| Run | T6 (3K) | T7 (6K) | T8 (9K) | T9 (12K) | IoU | Seq MSE |
|-----|---------|---------|---------|----------|-----|---------|
| **Baseline (2L)** | 22 | 31 | 41.3 | **34.5** | 0.044 | 0.066 |
| 3L sep=0.15 | 26 | 34 | 37.3 | **43.5** | 0.050 | 0.070 |
| **3L sep=0.25** | **26** | **38** | **41.3** | **44.0** | **0.042** | **0.071** |
| 3L sep=0.35 | 28 | 31 | 47.3 | **42.5** | 0.036 | 0.063 |
| corr=0.05 | 22 | 24 | 37.3 | **22.0** | 0.049 | 0.063 |
| corr=0.1 | 26 | 29 | 44.7 | **30.5** | 0.047 | 0.060 |
| corr=0.2 | 22 | 19 | 38.7 | **30.5** | 0.043 | 0.063 |

### Key findings

**1. 3-layer is the clear winner (+10pp over baseline).** All three 3L runs beat the 2L baseline by a wide margin (42.5–44% vs 34.5%). Pretrain MSE drops from 0.0112 to 0.0086 (−23%), confirming the **capacity hypothesis**: the per-task MSE gap (0.06–0.09 vs pretrain) was driven by insufficient expressivity at 2 layers.

**2. 3L eliminates the task-9 forgetting pattern.** The baseline shows a forgetting signature: success climbs to 41.3% after 3 tasks then drops to 34.5% on task 9. The 3L sep=0.25 run shows monotonically improving eval: 26 → 38 → 41.3 → 44.0. The extra layer provides enough slot capacity that task 9 doesn't overwrite earlier tasks' important slots.

**3. Corruption hurt more than it helped.** All corruption runs underperform the baseline (22–30.5% vs 34.5%). Corruption noise during pretraining degrades base model quality (pretrain MSE 0.0116–0.0128 vs 0.0112) without sufficient compensating robustness. corr=0.1 peaks at 44.7% after 3 tasks but then crashes to 30.5% on task 9 — same forgetting pattern as baseline, just from a worse starting point.

**4. Separation sweet spot remains ~0.25.** Consistent with the 2L finding from Entry 8. sep=0.25 gives the best balance across all 3L runs.

**5. The 3L benefit comes from capacity, not reduced overlap.** IoU is comparable across conditions (0.036–0.050). The third layer provides more exclusive slot budget per task without needing to further reduce routing overlap.

**6. Layer 10 gates conservatively.** L10 gate values (0.36–0.46 pretrain, 0.30–0.41 sequential) are notably lower than L12/L14, acting as supplementary capacity rather than a primary memory site.

### Remaining bottleneck

Sequential per-task MSE is still 0.063–0.071 vs pretrain 0.0086 — a 7–8× gap. Each task only updates ~2,000–3,000 unique slots (out of 147K per layer) because `tfidf_top_t=512` limits gradient updates to 512 slots per batch.

### Next experiments

**Top-T sweep (no new pretraining needed):**

3 sequential runs reusing the 3L sep=0.25 pretrained checkpoint with increased `tfidf_top_t`:
- `top_t=768` (1.5× baseline)
- `top_t=1024` (2× baseline)
- `top_t=1536` (3× baseline)

Higher top_t allows more slots to receive gradient updates per batch, directly increasing per-task capacity. Forward/backward compute is unchanged — top_t only masks gradients after backward. The write-overlap risk is managed by existing separation + IDF.

Scripts: `job_scripts/smolvla-memory/sequential/3_layer/top_t/`

**Future: gradient checkpointing for higher capacity pretrains.**

SmolVLA currently lacks gradient checkpointing (PI0 has it). Adding it to the expert transformer layers would reduce activation memory by ~50–60%, enabling:
- LoRA rank 4 (2× per-slot expressivity) at 3 layers
- 4-layer configurations (e.g. [8,10,12,14])

Both address the per-task MSE gap from different angles — rank increases per-slot expressivity while depth increases slot budget. Implementation would follow PI0's pattern: wrap expert layer forward passes in `torch.utils.checkpoint.checkpoint()` with `use_reentrant=False`.

---

## Entry 10 - 17 Mar 26 (Gradient Checkpointing + Capacity Pretrains)

Implemented gradient checkpointing for SmolVLA (`--policy.gradient_checkpointing=true`). Checkpoints each full transformer layer (attention + MLP/memory) during training, freeing intermediate activations. No change to inference.

**3 new pretraining runs launched** — all use sep=0.25, loc=0.25, support [128,2048], and gradient checkpointing enabled:

| Run | Layers | LoRA rank | Rationale |
|-----|--------|-----------|-----------|
| 2L r=8 | [12,14] | 8 | 4× per-slot expressivity at baseline depth |
| 3L r=4 | [10,12,14] | 4 | 2× expressivity + 3-layer slot budget (best depth from Entry 9) |
| 4L r=2 | [8,10,12,14] | 2 | Maximum slot budget, baseline expressivity |

Scripts:
- `job_scripts/smolvla-memory/pretrain/2_layer/lora_r_exp/pretrain_12_14_film_lora_8_..._sep_0.25_...`
- `job_scripts/smolvla-memory/pretrain/3_layer/lora_r/pretrain_10_12_14_film_lora_4_..._sep_0.25_...`
- `job_scripts/smolvla-memory/pretrain/4_layer/sep/pretrain_8_10_12_14_film_lora_2_..._sep_0.25_...`

---
**Update 18 Mar 26:** r=4 and r=8 OOMed despite gradient checkpointing. Added gradient accumulation support (`--gradient_accumulation_steps=N`) to both `lerobot-train` and `lerobot-sequential-train`, using Accelerate's `accumulate()` context. Memory usage stats, TF-IDF masking, and online IDF all accumulate correctly across micro-batches. The r=4 and r=8 scripts now use `batch_size=16, gradient_accumulation_steps=2` (effective batch=32). Rerunning.

---

## Entry 11 - 23 Mar 26 (Contrastive Pool Confound in Capacity Comparison)

### Observations and analysis

- Reviewed the latest `22_3_26` runs against the original baseline runs from `16_3_26` using the local wandb parser and the per-task memory-slot JSON dumps.
- The main empirical picture still looks consistent:
  - **4-layer [8,10,12,14], rank 2** is the strongest capacity result so far in sequential training (~55% final seen-task success vs 44% baseline).
  - **LoRA rank 4 at [10,12,14]** underperforms the baseline sequentially (~37.5% final), despite fitting the current task well.
  - The **top_t sweep** mostly changes write breadth, not routing/read overlap: larger `top_t` increases how many slots get updated and how much later tasks read through previously updated slots, which explains the non-monotonic retention pattern.
- From the slot JSONs, the most useful distinction is:
  - **read overlap / read-through-updated-slots** tracks forgetting
  - **write breadth / updated-slot count** tracks plasticity
- The 4-layer result improves because it adds lower-overlap adaptation capacity. The extra layer provides additional slot budget without increasing read overlap as much as rank-4 does.

### Code issue discovered

- While checking the fairness of the rank-4 comparison, I verified a training-loop detail in code:
  - `accelerator.accumulate(...)` delays optimizer stepping, but the **contrastive loss is still computed independently on each microbatch**
  - with `contrastive_query_queue=0`, the sample contrastive pool is therefore only the **current microbatch**, not the effective accumulated batch
- Relevant code paths:
  - `src/lerobot/scripts/lerobot_train.py`
  - `src/lerobot/policies/modules/memory_lite.py`
- This matters because the **rank-4 pretrain used `batch_size=16` with `gradient_accumulation_steps=2`**, whereas the baseline and 4-layer pretrains used a real batch size of 32.
- Therefore the current rank-4 result is **confounded**: its contrastive objective had a smaller effective pool than the baseline/4-layer runs. This weakens the claim that rank 4 is intrinsically worse.
- The 4-layer result is still more trustworthy under this specific issue because its pretrain did not rely on gradient accumulation.

### Decision / next step

- To remove this confound, we are rerunning the two capacity-expansion pretrains from this experiment family with a **contrastive query queue** enabled:
  - **3-layer [10,12,14], LoRA rank 4**
  - **4-layer [8,10,12,14], LoRA rank 2**
- For these reruns, `contrastive_query_queue=128` has been added to the job scripts so later microbatches can at least reuse recent queries when the real batch must stay small.
- After these reruns complete, we can re-evaluate whether:
  - rank-4 is genuinely worse than baseline / 4-layer
  - the earlier conclusion should instead be that rank-4 was mainly hurt by a contrastive-pool mismatch during pretraining

---

## Entry 12 - 27 Mar 26 (Global Balance Sweep — Targeting Read-Time Interference)

### Motivation

The dominant remaining forgetting channel is **read-time interference**: tasks read 13–16% of their retrieval weight through slots that a later task has updated. Write overlap is already near zero (TF-IDF + separation solved that in Entry 8), and the weighted TF-IDF variant barely moved the read-share numbers — strong evidence that sequential-side write heuristics are too blunt for this problem.

The existing pairwise separation loss (`routing_inter_task_separation_weight`) reduces cosine similarity between per-task routing distributions, but it can be satisfied asymmetrically (e.g., the "task 7 anomaly" from Entry 7 where one task split off while the rest still shared a hot core). It does not directly penalize **aggregate slot collapse** — the failure mode where all tasks concentrate reads onto a small shared core.

### Intervention

`routing_global_balance_weight` is an existing but previously untested loss term. It computes the task-averaged joint-slot probability distribution and penalizes low entropy via `(1 - normalized_entropy)`. Unlike pairwise separation, it directly requires the full slot table to be utilized across tasks, preventing the shared hot-core collapse.

This term works in conjunction with sep and loc — they are independent additive terms in the total loss:
- **Locality** keeps each task's routing compact
- **Separation** pushes task-pair routing apart
- **Global balance** spreads aggregate slot usage, preventing the shared core

### Experiment design

**3-point pretrain sweep** on the 4-layer [8,10,12,14] configuration (the best capacity result from Entry 9/11), adding global balance on top of the existing sep=0.25, loc=0.25 setup:

| Run | `routing_global_balance_weight` | Everything else |
|-----|---|----|
| gb=0.05 | 0.05 | sep=0.25, loc=0.25, sup [128,2048], lora_rank=2, contrastive_query_queue=128 |
| gb=0.1 | 0.1 | same |
| gb=0.2 | 0.2 | same |

Each pretrain has a matching sequential run (same sequential config as the 4-layer baseline: tfidf_top_t=512, online IDF, tasks [6,7,8,9]).

### What to watch

1. **Pretrain success** — stronger anti-collapse can hurt fit; need to confirm it doesn't degrade below the 4-layer baseline
2. **`eval/avg_pc_success_seen`** after 4 sequential tasks — the primary metric
3. **Task-9 read share through task-8-updated slots**, especially L12/L14 — the direct measure of read-time interference
4. **Weighted access IoU** (not binary overlap) — the meaningful interference metric per Entry 8 finding 5
5. **`routing_global_entropy`** during pretraining — should increase with higher weight; watch for saturation or training instability

### Scripts

- Pretrain: `job_scripts/smolvla-memory/pretrain/4_layer/routing_global_balance/`
- Sequential: `job_scripts/smolvla-memory/sequential/4_layer/routing_global_balance/`

### What we are not doing

- More weighted-TF variants (already showed weak leverage)
- Online query/key training during sequential (risks changing routing for old tasks)
- Stronger pairwise separation alone (doesn't address aggregate collapse)

---

## Entry 13 - 31 Mar 26 (Global Balance Results + Interpretation)

### Results from Entry 12 experiments

Ran the full 4-layer [8,10,12,14] `routing_global_balance_weight` sweep with matching sequential runs. One important comparison note:

- The saved `baseline_pretrain` / `baseline_sequential` runs in `31_3_26` are the older **3-layer [10,12,14]** best config.
- The **clean control** for this sweep is therefore the 4-layer rerun with the same settings and **`routing_global_balance_weight=0`**.
- Historical baseline is still useful for context, but it mixes layer-count / queue effects with global balance.

#### 4-layer global-balance sweep summary

| Run | Pretrain eval % | Pretrain effnum | Pretrain support | Final seq seen % | Seq weighted IoU | Task 9 reads from 8's updates (L12 / L14) |
|-----|-----------------|-----------------|------------------|------------------|------------------|-------------------------------------------|
| gb=0.0 | 82.5 | 2910 | 1727 | 46.0 | 0.041 | 7.8% / 6.0% |
| **gb=0.05** | **72.5** | **3932** | **2188** | **50.5** | **0.060** | **6.6% / 6.1%** |
| gb=0.1 | 72.5 | 4832 | 2553 | 39.0 | 0.073 | 7.2% / 6.1% |
| gb=0.2 | 75.0 | 6585 | 3084 | 39.0 | 0.099 | 7.1% / 6.2% |

Historical reference:
- saved 3-layer baseline pretrain = **87.5%**
- saved 3-layer baseline sequential = **44.0%**

So the best run from this sweep is **gb=0.05**:
- `+4.5pp` over the proper 4-layer control (`50.5` vs `46.0`)
- `+6.5pp` over the saved 3-layer baseline (`50.5` vs `44.0`)

### What global balance actually did

The global-balance term behaved exactly as intended in one narrow sense: it **reduced aggregate collapse**.

As `routing_global_balance_weight` increased:
- `routing_global_entropy_mean` increased monotonically
- `mem_usage_effnum_mean` increased strongly
- `mem_used_frac_mean` increased
- `mem_usage_top1_share_mean` decreased

This means the model was using a broader portion of the table and relying less on a tiny shared hot core.

However, the more important finding is that this did **not** translate into cleaner task-specific read subsets.

As global balance increased:
- per-task support expanded substantially (`1727 -> 2188 -> 2553 -> 3084` in pretraining)
- sequential weighted access overlap **increased**, not decreased (`0.041 -> 0.060 -> 0.073 -> 0.099`)
- gate usage dropped (`0.425 -> 0.406 -> 0.375 -> 0.374` in pretraining)

Interpretation:
- **Global balance spreads aggregate usage**
- but it also makes each task's routing footprint broader
- and those broader footprints overlap more at read time

So global balance is solving **aggregate load imbalance**, but it is not directly solving **harmful pairwise read overlap**.

### Performance interpretation

The best point, `gb=0.05`, improved final sequential success, but the task trajectory shows that it did **not** solve oldest-task forgetting.

Final per-env success on `[8, 1, 3, 5]`:
- gb=0.0: `30 / 48 / 48 / 58`
- **gb=0.05: `10 / 46 / 78 / 68`**
- gb=0.1: `16 / 28 / 50 / 62`
- gb=0.2: `10 / 34 / 44 / 68`

This means the `gb=0.05` gain came mainly from stronger performance on the newer / middle tasks (especially envs `3` and `5`), not from improved retention of the oldest task (`8`).

The strongest evidence that high global balance is the wrong extreme is `gb=0.2`:
- it starts with the strongest first-task score (`36%`)
- then forgets that task hardest (`10%` final)

That is the signature of a model with broad, shared routing that remains plastic for new tasks but does not preserve older task-specific reads.

### Main conclusions

**1. Global balance and anti-interference are different objectives.**
- Preventing aggregate hot-core collapse is not the same as reducing harmful pairwise read overlap.

**2. Mild global balance helps, but only in a narrow regime.**
- `gb=0.05` is the only useful point in this sweep.
- It gives a modest gain in final average seen-task success.

**3. Stronger global balance overshoots.**
- `gb >= 0.1` spreads routing too broadly, increases read overlap, lowers gate trust, and hurts final sequential performance.

**4. Read-time interference is still not solved.**
- The oldest-task problem remains.
- The right target metric is not aggregate entropy or mean table usage alone; it is harmful read overlap through updated slots, especially worst-case task pairs.

### Next steps

The result of this sweep is that we should **not** keep pushing global balance upward.

The clean next experiment is to test a knob that changes **actual retrieval breadth** rather than aggregate entropy:

- **`mem_knn` sweep on the 4-layer sep=0.25 control**
  - `mem_knn = 8 / 12 / 16`
  - this directly changes how many slots each query reads
  - unlike global balance, it attacks read breadth itself rather than trying to de-hotspot the whole table

To keep this sweep as controlled as possible, the pretrain scripts fix:
- `routing_loss_topk=16`

so changing `mem_knn` changes the actual retrieval set while leaving the routing-regularizer candidate pool fixed.

Scripts prepared under:
- `job_scripts/smolvla-memory/pretrain/4_layer/mem_knn/`
- `job_scripts/smolvla-memory/sequential/4_layer/mem_knn/`

### What we are not doing next

- More broad `routing_global_balance_weight` sweeps; the response is already clearly non-monotonic
- Another wider `routing_inter_task_separation_weight` sweep; higher separation already showed the tradeoff between lower overlap and worse useful sharing
- More sequential weighted-TF variants; they remain weak leverage against read-time interference

---

## Entry 14 - 3 Apr 26 (mem_knn Results + Next Isolated Robustness Tests)

### Results from Entry 13 experiments

Ran the planned 4-layer [8,10,12,14] `mem_knn` sweep with:
- `mem_knn = 8 / 12 / 16`
- `routing_loss_topk = 16` fixed
- same sep=0.25 / loc=0.25 / support [128,2048] / LoRA rank 2 / queue 128 setup as the current 4-layer control

#### Pretrain summary

| Run | Pretrain eval % | Pretrain MSE | Gate mean | Active effnum | Active used frac |
|-----|-----------------|--------------|-----------|---------------|------------------|
| knn=8 | 67.5 | 0.0175 | 0.323 | 1579 | 0.0447 |
| knn=12 | 77.5 | 0.0162 | 0.383 | 2289 | 0.0679 |
| **knn=16** | **90.0** | **0.0150** | **0.422** | **2949** | **0.0909** |

#### Sequential summary

| Run | Final seq seen % | Seq MSE | Gate mean | Weighted access IoU | Avg task-9 read share through task-8 updates |
|-----|------------------|---------|-----------|----------------------|----------------------------------------------|
| knn=8 | 42.5 | 0.0776 | 0.452 | 0.0390 | 7.84% |
| knn=12 | 43.5 | 0.0728 | 0.487 | 0.0447 | 7.37% |
| **knn=16** | **45.5** | **0.0728** | **0.535** | **0.0406** | **4.97%** |

Final per-env success after 4 tasks:
- knn=8: `12 / 40 / 50 / 68`
- knn=12: `20 / 36 / 50 / 68`
- knn=16: `24 / 38 / 54 / 66`

### Main findings

**1. Lowering `mem_knn` did not reduce the harmful interference channel enough to offset the loss in fit.**
- `knn=16` remains the strongest run in both pretraining and final sequential performance.
- Pretrain quality drops monotonically as `knn` is reduced (`90.0 -> 77.5 -> 67.5`).
- Sequential final seen-task success also drops (`45.5 -> 43.5 -> 42.5`).

**2. Smaller `knn` reduces useful mixture capacity faster than it reduces forgetting.**
- Lower `knn` gives lower gate usage and lower active effnum.
- This indicates the model is relying on memory less confidently and using a smaller effective set of high-weight slots.
- For LoRA values, this is especially costly because each query is assembling a weighted mixture of transforms rather than additive vectors.

**3. Plain read/read overlap is not the right target by itself.**
- `knn=8` actually has slightly lower weighted access IoU than `knn=16` (`0.039` vs `0.041`), but performs worse.
- The more useful metric is **read share through later-updated slots**.
- On that metric, `knn=16` is clearly best:
  - avg `8 reads 7 updates`: `1.56%` vs `2.37%` (`knn=8`) and `2.77%` (`knn=12`)
  - avg `9 reads 8 updates`: `4.97%` vs `7.84%` (`knn=8`) and `7.37%` (`knn=12`)

**4. The failure mode from smaller `knn` looks like concentration, not clean task separation.**
- As `knn` decreases, retrieval becomes more peaked and the model depends more on a narrower set of high-weight slots.
- That makes any shared updated slot more behaviorally important.
- So reducing retrieval breadth directly does not solve the overwrite problem; it can make the same overlap more damaging.

**5. There is a possible loss-mismatch confound for `knn < 16`, but it likely does not explain the whole result.**
- `routing_loss_topk` was held fixed at 16 while actual retrieval used 8 or 12 slots.
- That means the routing regularizer was not perfectly aligned to the real retrieval set for those runs.
- However, the consistent degradation in pretrain fit, gate usage, and harmful read-through suggests the main story is still that smaller `knn` is simply a worse operating point here.

### Interpretation

The original hypothesis was that reducing `mem_knn` might reduce interference by shrinking the read set. The sweep suggests the opposite tradeoff dominates:
- lower `knn` reduces the richness of the LoRA mixture
- this lowers pretrain fit and memory trust
- and it concentrates behavior onto fewer high-weight slots
- so shared overwritten slots matter **more**, not less

In other words, the current 4-layer system appears to benefit from a fairly rich mixture at read time. The next move should not be another lower-`knn` run.

### New experiments

To isolate the next directions cleanly, we are launching **three new pretrain + sequential pairs**, each with exactly one change from the current 4-layer `knn=16` control.

#### 1. `knn=24` pair

Goal:
- test whether **more** retrieval breadth improves fit and reduces the importance of any one overwritten slot

Reasoning:
- The `mem_knn` sweep indicates that decreasing read breadth hurts more than it helps.
- If the main problem is concentration of behavioral weight on a small number of shared slots, then increasing `knn` may help by spreading the LoRA mixture over more slot outputs.
- This should improve pretrain fit almost automatically; the key question is whether sequential retention also improves or whether read footprints become too broad.

Expected result:
- likely better pretrain fit than `knn=16`
- sequential outcome could be a modest win or a wash
- if it helps, it will probably do so by reducing the effective importance of shared overwritten slots rather than by lowering raw overlap

Scripts:
- `job_scripts/smolvla-memory/pretrain/4_layer/mem_knn/pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_24.sh`
- `job_scripts/smolvla-memory/sequential/4_layer/mem_knn/sequential_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_24.sh`

#### 2. `dropout_prob=0.1` pair

Goal:
- test robustness to **missing retrieved slots**

Reasoning:
- Dropout zeroes some retrieved slot contributions during training and renormalizes the remaining weights.
- This trains the policy to succeed when some normally-important slots are unavailable.
- That is not exactly the same failure mode as overwrite, but it is a cheap way to reduce dependence on a brittle small subset of slots.

Expected result:
- mild hit to pretrain fit
- possible retention gain if it reduces over-reliance on a few shared slots
- effect is likely smaller and gentler than corruption

Scripts:
- `job_scripts/smolvla-memory/pretrain/4_layer/dropout/pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_16_dropout_0.1.sh`
- `job_scripts/smolvla-memory/sequential/4_layer/dropout/sequential_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_16_dropout_0.1.sh`

#### 3. `corruption_prob=0.05` pair

Goal:
- test robustness to **drifted / overwritten slot outputs**

Reasoning:
- Corruption is a closer match to the real failure mode than dropout: later tasks do not delete shared slots, they change the retrieved LoRA outputs seen by earlier tasks.
- In the current `knn=16` control, harmful read-through is about `5-6%`, so `corruption_prob=0.05` is a better-calibrated starting point than the earlier `0.1` corruption setting used in the 2-layer experiments.
- This keeps the intervention milder than the older corruption sweep, which appeared to over-regularize.

Expected result:
- more targeted than dropout, but also riskier
- if it works, the win should come mainly from better oldest-task retention rather than higher current-task fit
- if it fails, it will likely fail by degrading pretrain quality without sufficiently reducing later forgetting

Scripts:
- `job_scripts/smolvla-memory/pretrain/4_layer/corruption/pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_16_corruption_0.05.sh`
- `job_scripts/smolvla-memory/sequential/4_layer/corruption/sequential_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_16_corruption_0.05.sh`

### What these runs will tell us

This batch is designed to separate three hypotheses:

1. **`knn=24` wins**:
   read-time interference is best addressed by reducing concentration and increasing mixture expressivity, not by shrinking the read set

2. **dropout wins**:
   the model is too dependent on a brittle subset of slots, and simple missing-slot robustness is enough to help retention

3. **corruption wins**:
   the main remaining problem is specifically value drift in shared-read slots, and robustness to overwritten LoRA outputs is the right target

### What we are not doing next

- another lower-`knn` sweep below 16; the direction already looks wrong
- combined dropout+corruption or `knn`+robustness runs yet; first isolate the effects
- another broad global-balance or separation sweep; the new question is robustness to shared-slot drift, not routing entropy

---

## Entry 15 - 9 Apr 26 (Isolated Robustness Results + Next Aligned Higher-`knn` Sweep)

### Results from the `9_4_26` batch

This batch compared the current 4-layer `[8,10,12,14]` control against three isolated changes:
- `knn=24`
- `dropout_prob=0.1`
- `corruption_prob=0.05`

Important comparison note:
- the saved `baseline_pretrain` / `baseline_sequential` runs in `9_4_26` are the current 4-layer control
- config: LoRA rank 2, `mem_knn=16`, `routing_loss_topk=16`, `sep=0.25`, `loc=0.25`, support `[128,2048]`, `contrastive_query_queue=128`

#### Pretraining summary

| Run | Eval % | MSE | Gate mean | Used frac | Effnum |
|-----|--------|-----|-----------|-----------|--------|
| **baseline (`knn=16`)** | **90.0** | 0.0150 | 0.422 | 0.0909 | 2949 |
| `knn=24` | 72.5 | **0.0145** | **0.492** | **0.1352** | **4383** |
| `dropout=0.1` | 67.5 | 0.0162 | 0.493 | 0.0956 | 3052 |
| `corruption=0.05` | 72.5 | 0.0161 | 0.416 | 0.0913 | 2874 |

#### Sequential summary

| Run | Final seen % | Seq MSE | Gate mean | Weighted access IoU | Avg `9 reads 8 updates` |
|-----|--------------|---------|-----------|----------------------|-------------------------|
| **baseline (`knn=16`)** | 45.5 | 0.0728 | 0.535 | **0.0406** | **4.97%** |
| **`knn=24`** | **49.5** | **0.0716** | **0.588** | 0.0437 | 5.41% |
| `dropout=0.1` | 42.5 | 0.0708 | 0.617 | 0.0480 | 6.60% |
| `corruption=0.05` | 43.5 | 0.0730 | 0.510 | 0.0467 | 7.00% |

Sequential eval progression (`avg_pc_success_seen`):
- baseline: `30.0 -> 32.0 -> 44.7 -> 45.5`
- `knn=24`: `24.0 -> 32.0 -> 40.0 -> 49.5`
- `dropout=0.1`: `28.0 -> 30.0 -> 46.0 -> 42.5`
- `corruption=0.05`: `28.0 -> 41.0 -> 47.3 -> 43.5`

Final per-env success after 4 tasks:
- baseline: `24 / 38 / 54 / 66`
- `knn=24`: `22 / 38 / 68 / 70`
- `dropout=0.1`: `12 / 36 / 60 / 62`
- `corruption=0.05`: `26 / 34 / 54 / 60`

### Main findings

**1. `knn=24` is the best result in this batch.**
- It improves final sequential performance from `45.5%` to `49.5%`.
- The gain comes mainly from stronger later-task / current-task performance, especially envs `3` and `5`, rather than from rescuing the oldest task.
- Oldest-task retention is slightly worse than baseline at the end (`22` vs `24` on env `8`), but the newer-task gains are larger.

**2. Higher `knn` helped by reducing concentration, not by reducing raw overlap.**
- In pretraining, `knn=24` substantially broadened memory usage:
  - used fraction `0.0909 -> 0.1352`
  - effnum `2949 -> 4383`
  - gate mean `0.422 -> 0.492`
- In sequential training it kept this pattern:
  - used fraction `0.095 -> 0.139`
  - effnum `2179 -> 3380`
  - gate mean `0.535 -> 0.588`
- Access overlap did **not** decrease overall (`0.0406 -> 0.0437`), so the improvement is not a “cleaner separation” story.
- The more plausible mechanism is that larger `knn` creates a richer LoRA mixture, spreading behavior over more slot outputs and improving effective per-task capacity.

**3. There is a loss-misalignment confound in the `knn=24` result, but it strengthens the case for testing larger `knn`, not weaker.**
- The `knn=24` pretrain kept `routing_loss_topk=16`, so the routing regularizer was still aligned to a 16-candidate retrieval objective while the actual read path used 24 slots.
- Despite this mismatch, `knn=24` was still the strongest sequential run.
- This means the direction looks promising even under a partially misaligned setup.

**4. `dropout_prob=0.1` is the wrong robustness mechanism here.**
- It hurts final sequential performance (`42.5%`).
- It also produces the clearest oldest-task collapse:
  - env `8`: `28 -> 18 -> 16 -> 12`
- Read-time interference worsens substantially:
  - avg `9 reads 8 updates`: `4.97% -> 6.60%`
  - weighted access IoU: `0.0406 -> 0.0480`
- Interpretation: making the model robust to **missing** slots is not helping with the real failure mode, which is **drifted shared slots**.

**5. `corruption_prob=0.05` gives a small retention-style effect, but not enough to beat the baseline.**
- It is the best run on the oldest task at the final checkpoint (`26` on env `8` vs baseline `24`).
- It is also strongest after task 2 and task 3 in average seen-task success.
- But it gives back too much current-task / newer-task performance by the end, finishing at `43.5%`.
- This looks more like a mild robustness-to-drift effect than a routing improvement.

**6. Rollout metrics are more informative than MSE alone in this regime.**
- `knn=24` has slightly better pretrain MSE than the baseline while much worse pretrain eval success.
- `dropout=0.1` has slightly better sequential MSE than the baseline while worse final success.
- This reinforces that eval rollouts plus overlap / read-through metrics are the right decision signals, not loss alone.

### Updated interpretation

The current picture is:
- reducing `knn` below 16 was the wrong direction
- increasing `knn` above 16 looks promising
- the benefit is coming from **less concentrated, more expressive read-time mixtures**
- not from cleaner routing overlap in the simple pairwise-IoU sense

This means the next bottleneck to probe is the tradeoff between:
- more expressive / less concentrated retrieval mixtures
- versus eventually over-broad read footprints if `knn` gets too large

### Next experiments

We are now moving to a **higher-`knn` sweep with aligned routing loss**.

Configs prepared:
- `knn=24`, `routing_loss_topk=24`
- `knn=36`, `routing_loss_topk=36`
- `knn=48`, `routing_loss_topk=48`

Rationale:
- `knn=24` already won despite the `routing_loss_topk=16` mismatch
- aligning the routing loss to the actual retrieval set is the cleanest next test
- if the main benefit is reduced concentration / richer LoRA mixtures, increasing `knn` further may continue to help
- if read footprints become too broad, the larger `knn` runs should reveal where that tradeoff turns over

What to watch:
1. `eval/avg_pc_success_seen` after 4 tasks
2. final per-env success, especially oldest task (`8`) versus newer tasks (`3`, `5`)
3. avg `9 reads 8 updates`
4. `memory_iou/all_modules_mean`
5. gate mean, top-1 share, used fraction, and effnum

### What we are not doing next

- we are **not** reintroducing corruption yet
- first we want to establish the best `knn` operating point under aligned routing loss
- if the higher-`knn` sweep confirms a new best setting, corruption can be revisited later on top of that stronger base

---

## Entry 16 - 14 Apr 26 (Aligned Higher-`knn` Results + Write-Budget Diagnosis)

### Results from the `14_4_26` batch

This batch reran the higher-`knn` sweep with the routing loss aligned to the actual retrieval set:
- `knn=24`, `routing_loss_topk=24`
- `knn=36`, `routing_loss_topk=36`
- `knn=48`, `routing_loss_topk=48`

Important comparison note:
- the saved `baseline_pretrain` / `baseline_sequential` runs in `14_4_26` are the earlier `knn=24`, `routing_loss_topk=16` control
- so the cleanest comparison for the aligned `knn=24` rerun is against that saved `24/16` baseline

#### Pretraining summary

| Run | Eval % | MSE | Gate mean | Used frac | Effnum | Routing support |
|-----|--------|-----|-----------|-----------|--------|-----------------|
| baseline `24/16` | 72.5 | 0.0145 | 0.492 | 0.135 | 4383 | 1747 |
| **aligned `24/24`** | **80.0** | 0.0149 | **0.516** | 0.110 | 4223 | 2937 |
| `36/36` | 77.5 | 0.0154 | 0.601 | 0.142 | 6041 | 4945 |
| `48/48` | 80.0 | 0.0148 | 0.637 | 0.176 | 7723 | 7100 |

#### Sequential summary

| Run | T6 | T7 | T8 | T9 | Final seen % | Seq MSE | Mean IoU |
|-----|----|----|----|----|--------------|---------|----------|
| baseline `24/16` | 24.0 | 32.0 | 40.0 | 49.5 | 49.5 | 0.0716 | 0.0437 |
| **aligned `24/24`** | **38.0** | 29.0 | **42.0** | **51.5** | **51.5** | **0.0616** | 0.0332 |
| `36/36` | 38.0 | **34.0** | 41.3 | 46.0 | 46.0 | 0.0646 | 0.0255 |
| `48/48` | 32.0 | 33.0 | 42.0 | 46.5 | 46.5 | 0.0627 | 0.0223 |

Final per-env success after 4 tasks:
- baseline `24/16`: `22 / 38 / 68 / 70`
- **aligned `24/24`: `36 / 40 / 66 / 64`**
- `36/36`: `26 / 34 / 68 / 56`
- `48/48`: `26 / 34 / 52 / 74`

### Main findings

**1. Aligning the routing loss at `knn=24` is a real improvement.**
- The aligned `24/24` run is the best result in this batch.
- It improves final sequential performance from `49.5%` to `51.5%`.
- Unlike the earlier `knn=24` win from the `9_4_26` batch, this gain is no longer confounded by a `routing_loss_topk=16` mismatch.

**2. The aligned `24/24` gain comes mainly from better oldest-task retention, not better newest-task fit.**
- Final env `8` improves strongly: `22 -> 36`.
- Newer-task performance is slightly lower than the old baseline on envs `3` and `5`: `68 -> 66`, `70 -> 64`.
- So the net win is a retention-style improvement with a better balance across tasks, not a pure current-task-performance gain.

**3. Increasing `knn` above 24 keeps reducing overlap metrics, but performance turns over.**
- Sequential mean IoU drops monotonically:
  - `24/16`: `0.0437`
  - `24/24`: `0.0332`
  - `36/36`: `0.0255`
  - `48/48`: `0.0223`
- Pairwise update-set IoU from the slot JSONs also drops monotonically.
- `task9 reads task8 updates` at layer 12/14 also falls overall as `knn` increases.
- But final seen-task success falls at `36` and `48`.

**4. That means overlap reduction is no longer the active bottleneck beyond `knn=24`.**
- If read-time interference were still the dominant limiter in this regime, `36/36` and `48/48` should have outperformed `24/24`.
- They do not.
- So once routing is aligned and overlap is pushed down to the `24/24` level, further overlap reductions have diminishing returns.

**5. The new bottleneck appears to be a write-budget mismatch.**
- As `knn` increases, the model reads from broader, more trusted mixtures:
  - sequential gate mean rises `0.595 -> 0.668 -> 0.707`
- But sequential TF-IDF still updates only `top_t=512` slots per batch.
- From the memory slot JSONs, mean unique updated slots per task/layer actually falls with larger `knn`:
  - baseline `24/16`: roughly `2532 / 2750 / 3363 / 3335`
  - aligned `24/24`: `2189 / 2444 / 3065 / 3403`
  - `36/36`: `1815 / 2053 / 2556 / 3023`
  - `48/48`: `1531 / 1776 / 2188 / 2761`
- Interpretation:
  - larger `knn` spreads read mass over more slots
  - fixed `top_t=512` then captures a smaller fraction of the read footprint for gradient updates
  - the model becomes less plastic even while it trusts memory more

**6. Layer 14 still matters most.**
- In both pretraining and sequential training, layer 14 remains the highest-overlap and highest-trust layer.
- Pretrain weighted overlap at layer 14:
  - baseline `24/16`: `0.026`
  - aligned `24/24`: `0.023`
  - `36/36`: `0.029`
  - `48/48`: `0.033`
- Sequential mean IoU is still highest at layer 14 for every run.
- So late-layer routing concentration has improved, but the late layer remains the main leverage point.

### Updated interpretation

The picture is now:
- alignment of `routing_loss_topk` to actual retrieval was worth doing and produced a genuine gain at `knn=24`
- pushing `knn` higher than `24` reduces overlap further but does **not** improve rollout performance
- the main tradeoff has shifted from:
  - overlap / interference
to:
  - read breadth vs update breadth

The system now looks **write-limited**:
- larger `knn` gives broader, more expressive mixtures at read time
- but with fixed `tfidf_top_t=512`, sequential training cannot update enough of the slots that those mixtures depend on

### Next experiments

The clean next test is a **sequential-only top-`t` sweep** on the stronger aligned checkpoints:

- `knn=24`, `top_t = 768 / 1024 / 1536`
- `knn=36`, `top_t = 768 / 1024 / 1536`

Rationale:
- if `24/24` is still the best retrieval operating point, larger `top_t` may improve plasticity further without needing new pretraining
- if `36/36` was mainly hurt by the fixed write budget, increasing `top_t` should recover more of its potential
- this directly tests the new write-budget hypothesis instead of returning to robustness or routing regularization

What to watch:
1. `eval/avg_pc_success_seen` after 4 tasks
2. final per-env success, especially env `8` vs envs `3` and `5`
3. mean unique updated slots per task/layer
4. `memory_iou/all_modules_mean`
5. `task9 reads task8 updates`, especially layers 12 and 14

### What we are not doing next

- we are **not** increasing `knn` beyond `48` yet
- we are **not** revisiting dropout or corruption yet
- first we want to test whether the current limitation is simply that `top_t=512` is too small for the broader aligned read footprints

---

## Entry 17 - 15 Apr 26 (Top-`t` Sweep Results + New `knn=36` Control)

### Results from the `15_4_26` batch

This batch was a **sequential-only** sweep over the TF-IDF write budget:
- `knn=24`, `top_t = 768 / 1024 / 1536`
- `knn=36`, `top_t = 768 / 1024 / 1536`

Important comparison note:
- no new pretrains were run in this batch
- the `knn=24` sequentials reuse the aligned `24/24` pretrained checkpoint
- the `knn=36` sequentials reuse the aligned `36/36` pretrained checkpoint
- so the pretrain-side differences are inherited from Entry 16, while the new signal here is how much extra sequential write budget each pretrained routing regime can actually exploit

#### Inherited pretrain comparison

| Pretrain | Eval % | MSE | Gate mean | Used frac | Effnum | Routing support |
|----------|--------|-----|-----------|-----------|--------|-----------------|
| `24/24` | 80.0 | 0.0149 | 0.516 | 0.110 | 4223 | 2937 |
| `36/36` | 77.5 | 0.0154 | 0.601 | 0.142 | 6041 | 4945 |

So before any new sequential training:
- `36/36` already reads from a broader and more trusted memory mixture than `24/24`
- the question in this batch was whether larger `top_t` lets sequential training update enough of that broader footprint to recover performance

#### Sequential summary

| Run | T6 | T7 | T8 | T9 | Final seen % | Seq MSE | Mean IoU |
|-----|----|----|----|----|--------------|---------|----------|
| baseline `24/24, top_t=512` | 38.0 | 29.0 | 42.0 | 51.5 | 51.5 | 0.0616 | 0.0332 |
| `24/24, top_t=768` | 28.0 | 28.0 | 48.0 | 51.5 | 51.5 | 0.0597 | 0.0337 |
| `24/24, top_t=1024` | 22.0 | 34.0 | 41.3 | 42.0 | 42.0 | 0.0592 | 0.0342 |
| `24/24, top_t=1536` | 32.0 | 33.0 | 49.3 | 48.0 | 48.0 | 0.0569 | 0.0352 |
| baseline `36/36, top_t=512` | 38.0 | 34.0 | 41.3 | 46.0 | 46.0 | 0.0646 | 0.0255 |
| `36/36, top_t=768` | 32.0 | 39.0 | 40.7 | 50.5 | 50.5 | 0.0600 | 0.0260 |
| `36/36, top_t=1024` | 34.0 | 40.0 | 38.7 | 52.0 | 52.0 | 0.0597 | 0.0265 |
| **`36/36, top_t=1536`** | **38.0** | **38.0** | **49.3** | **57.5** | **57.5** | **0.0529** | **0.0276** |

Final per-env success after 4 tasks:
- baseline `24/24, top_t=512`: `36 / 40 / 66 / 64`
- `24/24, top_t=768`: `30 / 46 / 66 / 64`
- `24/24, top_t=1024`: `14 / 40 / 58 / 56`
- `24/24, top_t=1536`: `22 / 38 / 62 / 70`
- baseline `36/36, top_t=512`: `26 / 34 / 68 / 56`
- `36/36, top_t=768`: `40 / 40 / 60 / 62`
- `36/36, top_t=1024`: `26 / 48 / 58 / 76`
- **`36/36, top_t=1536`: `28 / 46 / 76 / 80`**

### Main findings

**1. Entry 16's write-budget diagnosis was correct.**
- The `36/36` family improves monotonically as `top_t` increases:
  - `46.0 -> 50.5 -> 52.0 -> 57.5`
- The old conclusion that `36/36` was a worse operating point than `24/24` was therefore incomplete.
- What was actually true is:
  - `36/36` was worse under the old write budget
  - but once we increase `top_t`, it overtakes `24/24` clearly

**2. `36/36, top_t=1536` is the new best sequential setting so far.**
- It improves final seen-task success from `51.5%` to `57.5%` relative to the aligned `24/24` baseline.
- It also improves strongly over the original `36/36, top_t=512` control (`46.0 -> 57.5`).
- Sequential MSE also improves materially (`0.0646 -> 0.0529` within the `knn=36` family).

**3. The `24/24` family does not benefit from larger `top_t` in the same way.**
- `top_t=768` only ties the baseline.
- `top_t=1024` collapses badly (`42.0%` final, env `8` down to `14`).
- `top_t=1536` partially recovers but still underperforms baseline.
- So for `knn=24`, larger write budgets quickly become too interference-heavy.

**4. `top_t` is behaving mainly as a write-budget knob, not a read-footprint knob.**
- Within each fixed-`knn` family, the mean number of accessed slots changes only slightly as `top_t` increases.
- But the number of updated slots grows dramatically.

Mean unique updated slots per task/layer:
- `24/24`:
  - `top_t=512`: `2189 / 2444 / 3065 / 3403`
  - `top_t=768`: `3356 / 3708 / 4632 / 5266`
  - `top_t=1024`: `4588 / 5064 / 6210 / 7103`
  - `top_t=1536`: `6990 / 7842 / 9398 / 10755`
- `36/36`:
  - `top_t=512`: `1815 / 2053 / 2556 / 3023`
  - `top_t=768`: `2730 / 3114 / 3844 / 4501`
  - `top_t=1024`: `3710 / 4223 / 5197 / 6080`
  - `top_t=1536`: `5733 / 6460 / 7910 / 9338`

This is exactly what we hoped to test in Entry 16:
- broader `knn` increases read breadth
- higher `top_t` restores enough write breadth to match it

**5. But extra write budget also increases the harmful read-through channel.**
- `task9 reads task8 updates` rises sharply with `top_t` in both families.

At layers 12 / 14:
- `24/24`:
  - `top_t=512`: `4.21% / 3.78%`
  - `top_t=768`: `5.94% / 5.29%`
  - `top_t=1024`: `8.15% / 8.26%`
  - `top_t=1536`: `12.50% / 14.32%`
- `36/36`:
  - `top_t=512`: `2.49% / 3.25%`
  - `top_t=768`: `3.67% / 4.42%`
  - `top_t=1024`: `4.85% / 6.60%`
  - `top_t=1536`: `9.19% / 11.35%`

So larger `top_t` is always buying plasticity by accepting more eventual drift through updated slots.

**6. The reason `36/36` wins is that it has the right starting regime for this tradeoff.**
- The `36/36` pretrain already uses broader mixtures: higher gate, higher support, higher effnum.
- With `top_t=512` that regime was under-updated.
- With `top_t=1536`, it finally gets enough write coverage to exploit that broader read footprint.
- By contrast, the `24/24` regime already sits closer to the plasticity/interference boundary, so higher `top_t` mostly pushes it into overshoot.

**7. The new best run still has a retention weakness.**
- `36/36, top_t=1536` gets its gain mainly from the newer and middle tasks:
  - env `3`: `76`
  - env `5`: `80`
- Oldest-task retention is not solved:
  - env `8` still drops from `38` after task 3 to `28` after task 4
- So the system now has much better plasticity, but not clean immunity to shared-slot drift.

### Updated interpretation

The picture is now:
- `knn=36` was not intrinsically worse than `knn=24`
- it was **write-limited** under `top_t=512`
- increasing `top_t` reveals that `36/36` is a better read-time operating point once sequential updates can keep up

This means the main remaining bottleneck has shifted again:
- from “not enough write budget” to
- “how to keep the strong plasticity of `36/36, top_t=1536` while reducing drift in shared-read slots”

We now have a new sequential control:
- **`knn=36`, `top_t=1536`**

### Next experiments

We are now launching two follow-up directions.

#### 1. Extend the `knn=36` top-`t` sweep upward

New sequential-only runs:
- `knn=36`, `top_t=2048`
- `knn=36`, `top_t=3072`

Rationale:
- performance is still improving at `top_t=1536`
- this tells us the peak of the plasticity/interference curve has not been located yet
- the right next step is therefore to continue along the same axis before changing anything else

What to watch:
1. final `eval/avg_pc_success_seen`
2. env `8` retention versus envs `3` and `5`
3. `task9 reads task8 updates`, especially L12/L14
4. whether updated-slot counts keep rising faster than performance

#### 2. Revisit corruption on top of the stronger `36/36, top_t=1536` base

New pretrain + sequential sweep:
- `corruption_prob = 0.05 / 0.1 / 0.15`
- fixed `corruption_std = 0.1`
- fixed `knn=36`, `routing_loss_topk=36`
- sequential runs use `top_t=1536`

Rationale:
- in the new best run, the residual failure mode now looks like **drifted shared slots**, not missing-slot robustness
- `task9 reads task8 updates` is now roughly `9-11%` at the critical late layers in the best run
- so `corruption_prob=0.1` is now the natural center point, with `0.05` and `0.15` bracketing it
- this is a cleaner test than the earlier corruption sweeps, which were run on weaker bases and before the write-budget issue was resolved

### What we are not doing next

- we are **not** spending more runs on `knn=24` with larger `top_t`; that branch already looks overshot
- we are **not** revisiting dropout; the earlier evidence still says missing-slot robustness is the wrong target
- we are **not** adding more routing-loss sweeps yet; the dominant open question is now post-update drift under the stronger `36/36` operating point

---

## Entry 18 - 2 Jun 26 (REGIME CHANGE: pi05 + libero_goal, 10-task sequential — forgetting analysis)

### Regime change (IMPORTANT — read first)

Entries 0–17 were all **SmolVLA, 4 sequential tasks** (libero_spatial 6/7/8/9). This entry moves to a **new, harder regime**, so cross-entry numbers are not directly comparable:
- **Model: π₀.₅ (pi05)** — gemma_2b VLM + gemma_300m action expert (NOT SmolVLA). 6.6B total params, 2.4B trainable memory values, chunk_size 50.
- **10 sequential tasks** (not 4): the full **libero_goal** suite (dataset task_index 10–19), **held out of pretraining**.
- **Pretrain: libero_minus_goal** (30 tasks: long 0–9, object 20–29, spatial 30–39). Held-in eval (libero_spatial) = **65%**.
- Sequential eval = libero_goal, **50 episodes/task**.
- Config = the aligned knn=36 setup from Entry 16, applied 4-layer: layers [8,10,12,14], lora_rank 2, n_keys 384, **mem_knn=36 / routing_loss_topk=36**, sep=0.25, loc=0.25, support [128,2048], contrastive=0.01 (queue 128), **tfidf_top_t=512**, online IDF (denom 1, exp 1, weighting raw), train_memory_keys=False, value_lr 1e-3→1e-4, bs 32, 3000 steps/task.

Runs analysed:
- pretrain `libero_minus_goal_pi05_..._knn_36_30k`
- sequential `libero_goal_sequential_pi05_..._knn_36_30k`

**Cross-regime caveat:** absolute MSE and the Entry-17 57.5% are NOT comparable here (different model, action space, suite, task count). All causal claims below use *within-run* structure (per-task diagonal vs final, pairwise slot overlap, collapse timing).

### The carry-over miss

Entry 17 established knn=36 needs **top_t≈1536** (it is write-limited at 512). This run kept **top_t=512** → write-limited by construction. Two PI05-specific facts also matter: **gate ≈ 0.99** at every layer (vs 0.42–0.71 on SmolVLA — memory output dominates the residual), and pretrain held-in is only 65%.

### Headline

Final avg over 10 tasks = **45.8%**. Train order (env ids): 8, 9, 3, 6, 2, 5, 7, 1, 4, 0.

| ord | env | init % | final % | retained | seq MSE | read-thru overwrite | uniq updates (4L) | effnum L14 |
|----:|----:|-------:|--------:|---------:|--------:|--------------------:|------------------:|-----------:|
| 0 | 8 | 82 | 68 | 83% | 0.114 | 36.4% | 12.7K | 10337 |
| 1 | 9 | 32 | **0** | 0% | 0.208 | 45.1% | 24.3K | 12921 |
| 2 | 3 | 44 | **0** | 0% | 0.197 | 43.7% | 36.0K | 13710 |
| 3 | 6 | 40 | 24 | 60% | 0.127 | 29.9% | 32.6K | 13287 |
| 4 | 2 | 78 | 40 | 51% | 0.126 | 42.7% | 38.3K | 13317 |
| 5 | 5 | 52 | 10 | 19% | 0.108 | 29.0% | 34.2K | 9777 |
| 6 | 7 | 80 | 76 | 95% | 0.136 | 25.7% | 20.1K | 9507 |
| 7 | 1 | 86 | 92 | 107% | 0.122 | 15.0% | 32.2K | 18204 |
| 8 | 4 | 94 | 88 | 94% | 0.133 | 6.6% | 45.6K | 15786 |
| 9 | 0 | 60 | 60 | 100% | 0.195 | 0% | 33.8K | 10230 |

("read-thru overwrite" = fraction of that task's read weight on slots that *later* tasks updated, mean over 4 layers, from the memory_by_task JSONs.)

Validation that the slot analysis reads internals correctly: computed mean pairwise **weighted-read Jaccard = 0.1690** == the logged `memory_iou/all_modules_mean = 0.169`. So the logged IoU is **read** overlap — and it is **4–6× the 4-task runs (0.025–0.04)**. Write-set binary IoU is only **0.048**: TF-IDF still controls *writes*; the damage is on the *read* side.

### Two-factor model of forgetting

**Factor 1 — plasticity / transfer ceiling (upstream).** env 0 trained LAST (zero forgetting pressure) reaches only **60%** with MSE 0.195; env 9 trained 2nd (minimal prior interference) reaches only **32%** init. Several tasks fit to MSE ~0.20 vs pretrain 0.099. Frozen backbone + memory-only adaptation + top_t=512 write budget cannot fit ~half the held-out goal tasks even when current. A large chunk of "low scores" is NOT forgetting — it is the method's transfer ceiling on the OOD goal suite (same write-budget limitation from Entry 16/17, now surfacing as poor *current-task* fit because knn=36's broad reads are under-updated by top_t=512).

**Factor 2 — read-time overwrite of shared slots.** Reads are extremely broad (L14 touches 120–134K of 147K slots; weighted-read IoU 0.169). top_t=512 still grows each task's *unique* updated set to 12–46K slots, and because everyone reads broadly, those updates land in earlier tasks' read regions → **36–45% of every early task's read weight is overwritten by later tasks**. gate≈0.99 makes each overwrite maximally destructive.

**Interaction = the outcome.** final ≈ f(basin depth, overwrite exposure). Collapse needs BOTH a shallow basin AND high exposure:
- shallow + high exposure → 0%: env 9 (init 32, exposure 45%), env 3 (44, 44%)
- a deep basin absorbs exposure: env 8 (82, 36%)→68, env 2 (78, 43%)→40
- low exposure (recent) survives regardless: env 7/1/4 → 76/92/88

**Smoking gun (pairwise overwrite — frac of row task's reads on col task's updates):**
- **e9 ← e2 = 20.2%** → env 9 collapses (32→0) at *exactly* step 15000, the moment env 2 is trained. Direct, timing-matched.
- e2 ← e1 = 30.7%, e3 ← e0 = 18.4%, e8 ← e4 = 16.2%.
- Heavy overwriters = env 4/2/0/1 (update 32–46K slots); gentle = env 8/7 (13–20K). Update breadth tracks read breadth.

### Root cause of the abnormally high read overlap

Router/keys are FROZEN after a pretrain that **excluded the goal suite**. The 10 goal tasks are mutually similar ("put X on/in Y", FiLM-on-language queries) and OOD for the frozen router, so they all route into the same slot regions → 0.169 read IoU. The routing separation that worked at 4 tasks does not transfer to the held-out goal suite (IoU jumped from ~0.03 to 0.17 despite identical sep/loc/knn).

### Why the 4-task playbook (raise top_t) may backfire here

4-task runs had read IoU ~0.025, so top_t=1536 was safe. Here read IoU=0.169, so tripling the write budget triples overwrite onto shared reads → faster early-task collapse. Read overlap should be reduced first; THEN write budget can be raised to recover plasticity. The regime is now interference-dominated layered on a plasticity ceiling — the priority order flips vs the 4-task setting.

### Drivers of forgetting (answer to "what drove it")

1. **Plasticity ceiling** (under-updated broad reads + rank-2 LoRA + OOD frozen backbone): half the tasks can't be fit even when current.
2. **Read-time overwrite** of shared, high-gate slots: 36–45% of early-task read weight is later overwritten; amplified by gate≈0.99 and a frozen router that fails to separate the held-out goal tasks.

### Next experiments

**Running now (sequential-only, reuses the knn=36 pretrain):**
- **top_t=1536** rerun — `libero_goal_sequential_pi05_..._knn_36_30k_top_t_1536` (wandb run e1006ra1). Tests whether the write-budget lever recovers the plasticity-limited tasks (env 0/9/3) at 10-task scale, vs worsening early-task overwrite (e9←e2 already 20% at top_t=512). Watch env-8/9/2/5 retention and `task9-reads-task8-updates`.

**Staged plan after that:**
- IDF slot protection (idf_exponent 1→2, or contribution/saturating TF off `raw`) to keep later tasks' top_t off earlier tasks' high-weight slots — sequential-only, low risk, attacks the e2→e9 channel.
- knn=24 / routing_loss_topk=24 + top_t≈768 (new pretrain) — Entry-16 retention-friendly point; narrower reads → lower overlap → safe to add write budget.
- Make the held-out router separable: stronger separation/contrastive on held-in goal-like tasks, OR small `memory_keys_lr` during sequential (train_memory_keys=true) so routing specialises the goal tasks (medium risk).
- Raise the floor: more steps/task (3000→5000; loss still falling at task boundaries), lora_rank 2→4 (grad-ckpt already on), stronger/longer pretrain (held-in only 65%).
- Amplifier: gate saturates at 0.99 (no modulation) — probe a gate regulariser / per-task update budgeting to tame heavy overwriters.

---

## Entry 19 - 9 Jun 26 (libero_90 → LIBERO-Long results + write-protection mechanism autopsy + 10k probe plan)

### Runs analysed

- pretrain `libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k` (90 tasks, 40k steps)
- sequential `libero_10_sequential_pi05_..._knn_36_40k_top_t_1536` (LIBERO-Long, 10 tasks × 3000 steps, `tfidf_top_t=1536`, 50 eval eps/task)
- script: `job_scripts/nebius/libero_90/combined/pi05_libero_10_4_layer_film_lora2_knn36_40k_c0.01_topt1536.sh`

Deltas vs Entry 18: pretrain libero_90 (90 tasks, 40k) instead of libero_minus_goal (30 tasks, 30k); sequential suite is LIBERO-Long (libero_10) instead of libero_goal; `top_t=1536` from the start. Cross-suite absolute numbers are NOT comparable (LIBERO-Long is the hardest suite — six of ten tasks need two full pick-place cycles).

### Pretrain summary

- Held-in eval: `76.4%` @20k → `81.1%` @40k (vs 65% for the 30-task pretrain). MSE `0.196` @20k → `0.133` @40k. **Both still improving when the schedule hit zero → 40k undertrains libero_90.**
- Gate ≈ `0.98` at all four layers (pi05 saturation again — overwrites pass at full strength).
- Usage is broad and healthy at the aggregate level: table-wide effnum 50–62K of 147K per layer, top-1 share ~0.0006. **No hot-core collapse — the 90-task pretrain fixed the Entry 18 global-routing problem.**
- Inter-task routing sim 0.10. Intra-task support mean ~5.8K vs locality band [128, 2048]:
  - **L8 compacted (8,012 → 4,298, −46%) but L14 never moved (7,930 → 7,398, flat from ~10k on, 3.6× above band max).** Locality weight 0.25 contributes ~0.002 to a 0.17 objective — MSE outbids it exactly at the layer that matters most.

### Sequential headline

Final avg over 10 tasks = **34.4%**. Trajectory: 26 → 32 → 29.3 → 39 → 31.2 → 29.3 → 25.1 → 22.5 → 25.3 → 34.4. Logged read IoU `0.107` (slot-JSON recomputation matches exactly — pipeline validated).

Per-task (train order; env id per `ds_to_env_map`):

| ord | env | task | init | peak | final | ret% |
|----:|----:|------|-----:|-----:|------:|-----:|
| t0 | 4 | two mugs → plates | 26 | 30 | 18 | 69 |
| t1 | 6 | mug → plate + pudding | 34 | 48 | 30 | 88 |
| t2 | 9 | mugs → microwave + close | 20 | 24 | **6** | 30 |
| t3 | 2 | stove on + moka pot | 68 | 68 | 54 | 79 |
| t4 | 7 | soup + cheese → basket | **8** | 8 | 4 | 50 |
| t5 | 0 | soup + sauce → basket | 30 | 30 | **2** | 7 |
| t6 | 8 | both mokas → stove | 32 | 52 | 52 | 162 |
| t7 | 1 | cheese + butter → basket | 28 | 28 | 24 | 86 |
| t8 | 3 | bowl → drawer + close | 74 | 76 | 76 | 103 |
| t9 | 5 | book → caddy | 78 | 78 | 78 | 100 |

**mean init = 39.8, mean peak = 44.2, mean final = 34.4.** Even perfect retention caps this run at ~40% — the diagonal is the binding constraint; forgetting costs ~5.4pp on top (9.8pp vs peak) but is catastrophic where it occurs.

### Failure mode 1 — plasticity ceiling on dual-cycle tasks

- The diagonal splits by task structure, not order: all six two-full-pick-place tasks fit at 8–34%; the three one-cycle(+short-second-step) tasks fit at 68–78. Order is disconfirmed within the basket family (inits 8 → 30 → 28 with order).
- Worst sequential MSE plateaus: t2 `0.255`, t7 `0.233`, t4 `0.210` vs pretrain `0.133`. MSE still falling at the end of nearly every 3000-step block while the per-task LR has decayed to 1e-4 (LR is a per-block linear 1e-3 → 1e-4 sawtooth; wandb sampling aliases it).
- t5 is the rollout-vs-MSE cautionary case: MSE `0.089` (2nd best) but 30% init — long-horizon compounding. Rollouts remain the only trustworthy fit signal.
- LIBERO-90 contains only single-step tasks → the two-cycle composition itself is OOD; rank-2 LoRA mixtures on a frozen backbone don't close that gap in 3000 steps.

### Failure mode 2 — family-clustered read-time overwrite, amplified by top_t=1536

- Weighted read IoU: background ~0.09–0.12, but the **basket family** (t4/t5/t7, near-identical "put both X and Y in basket" instructions) sits at **0.335 / 0.300 / 0.228**. Also t2↔t8 = 0.171 (both "place + close articulated container"). L14 worst layer (mean 0.155, max 0.389).
- `top_t=1536` made every task a heavy writer: 47K–112K unique slots updated per task (4-layer sum; t7 112K, t8 104K ≈ 26% of the L14 table each) — ~3× the write breadth of top_t=512 runs.
- Read-through any-later for early tasks: **47–86%** of read weight on later-updated slots (t4 86%, t2 75%, t5 67%; Entry 18 at top_t=512 was 36–45%).
- Timing-matched collapses: t5 30→4 and t0 18→4 at exactly step 24000 (t7's block; rt t5←t7 = 45.5%, t0←t7 = 24.5%); t2 12→6 at t8's block (rt t2←t8 = 44.1%).
- Direct write-budget control on libero_goal: **top_t=1536 = 43.0% vs top_t=512 = 45.8%** — raising the budget at 10-task scale (read IoU ≥ 0.1) is net-negative, as Entry 18 predicted.
- Self-write coverage 58–87%: the write mechanism works (tasks write where they read) — it's mis-targeted, not undersized. Positive transfer is real (t6 32→52 after later tasks; t8/t9 retain ≥100%).

### What improved vs Entry 18 / what's spent

Read IoU 0.169 → 0.107, held-in 65 → 81.1, no hot core. **The pretrain-diversity lever worked and is now mostly spent**: the remaining overlap is semantic (lookalike instructions route together under a frozen FiLM-on-language router, by construction). Don't buy another pretrain expecting IoU to drop further from data diversity alone.

### Mechanism autopsy (new analyses, from the per-task slot JSONs + code)

**1. TF-IDF protection is functionally a TF mask at this scale.**
- Code (`lerobot_sequential_train.py`): DF increments are binary-per-batch (`df_vec[used] += 1.0` — within-batch intensity discarded); IDF = `log((B+1)/(DF+1))^e`, recomputed only at task boundaries; mask score = in-batch raw-count TF × IDF, top-1536.
- Reconstructed the exact IDF in effect during t7's block (L14, B=21,000, DF = Σ batch_accesses of t0–t6): t4/t5 core-50% slots median IDF **0.49** vs t7's other written slots **1.84** → a **3.0× penalty**, while TF on those same slots runs **~86×** the accessed-median. TF wins; the log crushes a 60:1 DF ratio into 3×.
- Counterfactuals: at e=1, **100%** of t7's 2,913 writes into t4/t5 cores survive the write margin; **e=2 keeps 97%**; e=4 keeps 60%. Mass-weighted DF is worse (per-batch mass shares ≈ 0 → IDF uniform ~9.9 → zero protection). Only ~400 universal slots (DF ≥ 0.9B → IDF → 0) are genuinely hard-protected.
- **Decision: the `idf_exponent=2` plan is dropped** — measured as a near-no-op before spending a run on it.

**2. Hard protection is zero-sum under this routing.** Veto "slots present in >τ of any single prior task's batches": τ=0.7/0.5/0.3 vetoes 37/50/64% of the table, rescues 100% of the measured core damage at every τ — but blocks 68/79/88% of t7's write events, and **83–96% of t7's own read mass sits on vetoed slots**. Any write policy strong enough to protect predecessors starves the new task. Write-side knobs alone cannot produce the target behaviour.

**3. Incidental vs intrinsic damage (L14).** Cross-family damage is incidental for the writer (t2←t8: hit slots carry 21.7% of t2's mass but only 10.3% of t8's; t3←t8: 16.1/7.2; t0←t7: 12.2/4.5) → protectable cheaply in principle. Within-family it is contention (t4←t7: 34.8/24.3; t5←t7: 29.5/18.2; t4←t5: 37.4/32.3) → the writer needs the slots it damages; only routing separation can fix this channel.

**4. Routing breadth, measured at the right granularity.** The "94% of table touched per task" union is coupon-collector arithmetic (~4.8M queries × ≤144 slots each). The meaningful numbers: a **single same-task batch touches ~31K unique L14 slots** (12.9K at L8); per-task effnum 19K; core-50 ~3.3K. Combined with the L14 locality stall (above): **the router never learned per-task chunking at late layers even in-distribution, and held-out tasks can't be allocated chunks at all under frozen routing.**

**5. Gradient-dilution hypothesis tested and rejected as the diagonal driver.** Update events ~119/slot mean for the heavy writers, but concentration is uncorrelated with fit: t8 is most diffuse (118 ev/slot) with init 74%; t0 most concentrated (286 ev/slot) with init 26%; t5 fits to near-best MSE with the 2nd-largest write set. The libero_goal 512-vs-1536 pair (3× concentration difference) shows no init advantage either. The diagonal is task structure + expressivity, not gradient spreading.

**6. SupCon facts (code, `memory_lite.py:_compute_sample_contrastive_loss`).** The sample contrastive is SupCon at τ=0.07 → gradient concentrates exponentially on the *most similar* cross-task pairs (hard-negative emphasis is built in — the property needed for lookalike families). Caveats: it operates on the per-sample mean query (task-centroid geometry, not slot footprints — the Entry 5/7 proxy gap applies); default SupCon keeps same-task terms in the denominator (intra-class spreading at high weight — `contrastive_negatives_only=true` exists to remove it); batch 32 + queue 128 is thin negative coverage for 90 classes.

### Constraints fixed in discussion (design space)

- **Router stays frozen, permanently.** Slow key-training during sequential was tried previously: per-task plasticity improved greatly, prior-task performance was decimated (frozen old queries × moved keys silently re-points old tasks' retrieval). Off the table.
- **No per-task parameters, no added parameters of any kind.**
- **Pretrain-task forgetting is acceptable.** Pretraining is purely prior-construction for clean sequential adaptation; only the 10 sequential tasks need protecting from each other.
- Still off the table: EWC, replay, hard task-boundary slot allocation.
- Target property: *each sequential task inherits a compact, separated footprint from the frozen router; cross-task overlap only at benign/synergistic edges.* Everything must be bought at pretrain time via the routing losses.

### Plan: two 10k single-lever probes (running now)

Both are **truncated full runs** (same warmup 4000 / decay 40000 schedule, `--steps=10000`, `save_freq=10000`, eval never fires) so a passing probe continues to 40k with `--resume=true` at zero wasted compute. ~11h each on the H200, run sequentially in tmux session `probes`; logs in `outputs/probe_logs/`.

Scripts: `job_scripts/nebius/libero_90/probes/`
- `probe_10k_pretrain_loc_1.0.sh` → run `libero_90_pi05_8_10_12_14_probe10k_loc_1.0`
  - Single knob: `routing_intra_task_locality_weight 0.25 → 1.0`.
  - **Pass:** L14 intra-task support < ~5.5K by 10k and still descending (control: flat ~7.9–8.2K); MSE@10k within ~10% of control's 0.261; audit shows held-out L14 core-50/effnum down ~2×. Family IoU recorded but not required (compaction ≠ de-coincidence).
  - **Fail route:** L14 flat at 1.0 → per-layer locality (heavy on 12/14), not a bigger global weight.
- `probe_10k_pretrain_c_0.05_negonly_q512.sh` → run `libero_90_pi05_8_10_12_14_probe10k_contrastive_0.05_negonly_q512`
  - Contrastive arm as a package: weight 0.01 → 0.05, `contrastive_negatives_only=true`, queue 128 → 512 (flag = the high-weight pathology guard; queue = SupCon statistics enabler). Locality stays 0.25.
  - **Pass:** held-out audit family IoU (basket t4/t5/t7, L14) down ≥~30% vs control audit; MSE guardrail as above.
  - **Fail route:** query_inter_sim moves but slot-level family IoU doesn't (the Entry 5/7 proxy failure) → similarity-weighted *separation* (slot-space pairwise term weighted by language-embedding similarity; small code change, no new params).

**Review instrument (to write before evaluating — first TODO at review):** held-out routing audit = forward the libero_10 demos through a frozen checkpoint with per-task usage logging (no training), dump the slot JSONs, compute per-task L12/L14 support, core-50, effnum, and the pairwise weighted IoU matrix. Run it on the **control checkpoint (20k and/or 40k) first** to fix the baseline — control routing plateaued early, so control@20k is a fair comparator for probe@10k. Minutes-to-an-hour per checkpoint vs 45h for a sequential run.

### After the probes

1. Stack whatever passes → `--resume` to 40k (or relaunch stacked) → full audit at 40k.
2. One sequential run on the graduating prior:
   - *[SUPERSEDED — a sequential-side write-rule change was floated here; sequential-side anti-forgetting schemes are OFF THE TABLE per project constraint. Disregard. The fix stays in the prior.]*
   - **`top_t` re-derived from audited footprint size** — if core-50 shrinks ~3×, 1536 per batch is proportionally far too destructive; do not carry it over (that mistake is how this run got its forgetting cliff).
3. Orthogonal plasticity track (untouched by all routing work, needed for the diagonal): steps/task 3000 → 5000 with LR floor ~2e-4; longer pretrain (40k undertrains — MSE/eval both still improving); `lora_rank=4` only if the ceiling persists after those (memory-heavy: 2.4B → ~4.8B trainable values).

### What we are not doing

- No `idf_exponent` sweep, no weighted-DF variant — both measured as ≈no-ops offline before spending GPU.
- No router/key training during sequential (decided, see constraints), no per-task parameters.
- No further `top_t` increases; no new pretrain bets without passing the held-out routing audit first.

---

## Entry 20 - 11 Jun 26 (Probe results + held-out audit: SupCon package is the breakthrough; locality loss is dead)

### Probe outcomes (in-run, @10k, trailing-1k window, vs control @10k)

Both probes completed cleanly (~11h each, `outputs/probe_logs/`).

| gate metric | control | probe L (loc=1.0) | probe C (SupCon pkg) |
|---|---|---|---|
| L14 intra-task support | 7,837 (flat) | 6,841 (−13%, descending) | **3,446 (−56%)** |
| L8/L10/L12 support | 5,692/6,590/6,979 | 4,899/5,519/5,887 | **2,456/2,569/2,848** |
| inter-task routing sim (slot-level) | 0.138 | 0.155 | **0.039** |
| query inter / intra sim | 0.13 / 0.79 | 0.17 / 0.82 | **−0.02 / 0.99** |
| MSE (guardrail ≤ ~0.27) | 0.243 | 0.233 ✓ | 0.246 ✓ |

- **Probe L FAILS its gate.** First config ever to get L14 support *descending* (control was flat), at zero MSE cost — but 4× weight bought only 13%. Locality-as-a-loss has weak leverage against MSE.
- **Probe C achieved both probes' targets at once.** Mechanism: SupCon's pull-together drove same-task sample-mean queries to cosine 0.99 → per-task routing compacts (locality's job, done better via the query map); negatives-only push-apart drove cross-task centroids to ~orthogonal; and — unlike every prior query-space attempt (Entries 1, 5, 7) — it moved the **slot-level** metrics (inter-task sim 0.138 → 0.039). Not the proxy failure this time.
- Bookkeeping: probe C's `train/loss` is negative (≈ −0.15) — the negatives-only SupCon term is a negative log-sum-exp (weighted ≈ −0.40). Gradients unaffected; compare runs on `mse_loss` only.
- Aggregate-usage check (checkpoint `memory_usage.json`): probe C L14 aggregate effnum 49.6K with per-task support 3.4K → aggregate/per-task ratio ~14× vs control ~8×. **Compaction + separation, not collapse onto a shared core.**

### Held-out routing audit (the decisive test)

Instrument: stream the libero_10 demos through the FROZEN checkpoint via `lerobot-sequential-train` with `memory_value_lr=1e-12` (numerically inert), no env (no eval), no checkpointing, 100 batches/task — dumps the standard `memory_by_task` JSONs measuring the **pristine prior's** held-out footprints. ~35 min/checkpoint. Scripts: `job_scripts/nebius/libero_90/probes/{audit_heldout_routing.sh,run_audits_seq.sh}`; outputs `outputs/train/audit_heldout_{control_40k,probeC_10k,probeL_10k}/`.

Baseline caveat: control audited @40k (20k checkpoint was accidentally deleted); probes @10k. The training-amount confound is covered by probe L, which acts as a same-duration control (near-control config, audited @10k, shows ≥control overlap).

Held-out per-task footprints (L14, mean over 10 tasks):

| run | effnum | core50 | vs control |
|---|---|---|---|
| control@40k | 14,991 | 2,643 | — |
| **probeC@10k** | **2,070** | **351** | **~7× smaller** |
| probeL@10k | 12,346 | 2,214 | 0.84× (negligible) |

Held-out pairwise weighted IoU (L14):

| run | t4-t5 | t4-t7 | t5-t7 | FAMILY mean | t2-t8 | off-diag mean | background (non-family) |
|---|---|---|---|---|---|---|---|
| control@40k | 0.391 | 0.355 | 0.302 | 0.349 | 0.235 | 0.142 | 0.127 |
| **probeC@10k** | **0.236** | **0.229** | **0.104** | **0.190 (−46%) PASS** | 0.096 | 0.053 (−63%) | 0.043 (−66%) |
| probeL@10k | 0.494 | 0.346 | 0.369 | 0.403 (+15%) **FAIL** | 0.264 | 0.186 | 0.171 |

Sanity: control audit family IoUs (0.39/0.36/0.30) match the sequential-run measurements (0.34/0.30/0.23) to within the writes/duration differences — pipeline consistent.

### Conclusions

1. **The SupCon package (weight 0.05 + `negatives_only=true` + queue 512) is the first intervention in the whole project that improves held-out slot-level routing.** It generalizes: basket-family separation −46%, background −66%, footprints ~7× compacter — on tasks it never saw, at +1.2% MSE.
2. **The locality loss is dead.** At 4× weight it barely compacts in-distribution and generalizes *backwards* (held-out overlap +15% vs control). Per-layer locality (the old fail-route) is moot given C. Plausible reading: locality pressure makes routing less query-sensitive overall, which makes lookalike held-out tasks route *more* identically.
3. Family overlap is reduced, not eliminated (0.190 vs background 0.043 → still ~4.4× background). The basket family remains the top residual interference channel — but with footprints 7× smaller and IoU halved, the write-collateral arithmetic is transformed.
4. Risk to watch: does compaction/separation hold through full training (LR decay, MSE pressure for capacity), and does held-in fit keep tracking control?

### Decisions / next steps

1. **Resume probe C → 40k** (`job_scripts/nebius/libero_90/probes/resume_probeC_to_40k.sh`; ~33h). Gates: MSE tracks control (0.196 @20k / 0.133 @40k), held-in eval @20k/@40k vs control 76.4/81.1, support & query-sim stability.
2. **Re-audit at 40k** (existing audit script, ~35 min) — confirm held-out compaction/separation survives full training.
3. **Sequential run** on the audited 40k checkpoint with `top_t` re-derived from the new footprints: held-out core50 dropped ~7.5× (2.6K → ~350), so `top_t=1536` is ~an order of magnitude oversized for this regime — start at **256–512** (decide from the 40k audit). Minimal-change config (everything stays pretraining-side; the only sequential knob is the existing TF-IDF top_t).
4. Probe L checkpoint: keep for reference, no further investment.

**Update (launched 11 Jun, tmux `pipeline`):** all three stages packaged in `job_scripts/nebius/libero_90/probes/pipeline_probeC_full.sh` (idempotent stages, auto-fallback if the resume `--steps` override is ignored). Decisions baked in:
- `top_t=512` pre-committed rather than audit-gated: protection now comes from separated/compact footprints, not the write mask; per-batch accessed slots shrink ~7× with this prior, so 512 is *relatively* more generous than 1536 was in the old regime, and libero_goal showed 512 safe at far worse IoU (0.17) than this prior's held-out 0.05 bg / 0.19 family. The 40k audit (stage 2) is informational.
- **Contrastive weight held at 0.05** for this cycle: 7× compaction at +1.2% MSE is already near the useful ceiling, weight responses have been non-monotonic throughout the project, and the open risk is fit-side (which more pressure worsens). Weight 0.1 becomes a 10k probe only if the 40k re-audit shows separation eroding.
- Sequential config otherwise identical to the failed top_t=1536 run (3000 steps/task, lr 1e-3→1e-4, 50 eval eps, same env mapping) for clean attribution: pretrain recipe + top_t are the only deltas.
- Sequential run name: `libero_10_sequential_pi05_8_10_12_14_contrastive_0.05_negonly_q512_40k_top_t_512`. ETA ≈ 3.3 days (33h resume + 35min audit + ~45h sequential).
- What to look at first when it lands: retention matrix vs Entry 19's (esp. t5 after t7's block, the step-24000 cliff), diagonal inits on the dual-cycle tasks (plasticity should be roughly unchanged — this cycle attacked interference, not the ceiling), `memory_iou/all_modules_mean`, read-through-any-later for t0–t5, and 40k-audit family IoU vs the 10k audit's 0.190.
- **Pre-registered next lever if retention is good but absolute perf lands ~50%:** the bottleneck is then the plasticity ceiling, and the first response is optimization budget, not architecture — more steps/task (3000 → 5000; Entry 19 showed MSE still falling at every block end) and higher memory-value LRs (floor 1e-4 → ~2e-4 first; peak 1e-3 → ~2e-3 second, watching within-block stability — t8's MSE rose late in its own block at the current peak). These are sequential-only, cheap, and now safer to push because the compact/separated footprints mean extra write pressure leaks far less into other tasks' cores than it would have pre-probe-C.

**Update 2 (11 Jun, correction — resume abandoned, fresh 40k launched):**
- **The "truncated full run / resume for free" probe design was wrong.** lerobot auto-scales the LR schedule when `steps < scheduler_decay_steps` (`schedulers.py:111`; probe logs confirm: "Scaling warmup: 4000 → 1000, decay: 40000 → 10000"). Both probes therefore ran a **compressed full cosine**, fully decayed by 10k.
- Consequences for Entry 20 conclusions: **none material.** Probe L vs probe C shared the identical compressed schedule, so that contrast is clean; the held-out audit gaps (7× compaction, −46% family IoU, with probe L as a same-schedule control moving the opposite direction) dwarf any schedule artifact. Bonus: the SupCon effects survived a complete LR decay — they are end-of-training properties, not high-LR transients. One nuance: probe-vs-control *in-run* comparisons at 10k carried an LR-position confound (probes at LR floor, control at ~0.85×peak), which slightly flattered probe MSE.
- The resume (launched this morning) rebuilt the scheduler **unscaled** (steps=40000 ≥ decay), so its LR jumped from floor to ~0.85×peak at step 10001 — an SGDR-style sawtooth, not comparable to control. Killed at ~step 10.4k (no checkpoints written; probe C's 10k checkpoint pristine; the resume's brief wandb re-attachment to run `hdbpetb9` is cosmetic).
- **Replacement: fresh 40k pretrain of the recipe** with a clean schedule (steps=40000 == decay → warmup 4000 / decay 40000 honored, exactly matching control): run `libero_90_pi05_8_10_12_14_contrastive_0.05_negonly_q512_40k`, script `probes/pretrain_c_0.05_negonly_q512_40k.sh`, checkpoints at 10k/20k/30k/40k, evals at 20k/40k. Pipeline v2 (`pipeline_probeC_full.sh`) now: fresh pretrain → 40k audit (`audit_heldout_c005_40k`) → sequential top_t=512 (unchanged). Cost vs resume: +~11h; chain ETA ≈ 44h + 35min + 45h ≈ 3.7 days.
- Extra validation the fresh run gives for free: whether SupCon's compaction/separation holds under the uncompressed schedule — check `routing_intra_task_support_*` and query sims at 10k/20k against the probe's values, and the 10k-checkpoint audit can be compared like-for-like against `audit_heldout_probeC_10k` if needed.

---

## Entry 21 - 14 Jun 26 (SupCon recipe at full scale: forgetting SOLVED, plasticity DESTROYED — over-compaction autopsy)

### Headline

The full 40k SupCon pretrain (`contrastive=0.05`, `negatives_only=true`, queue 512) + sequential libero_10 at `top_t=512` was **killed at step ~18k (task 6/10)** — performance was clearly worse than the failed top_t=1536 run. Diagnosis is unambiguous and clean: **we eliminated read-time interference and destroyed per-task capacity at the same time, via the same knob. Net is much worse.**

- Sequential **cold-start (diagonal) avg over the 5 completed tasks = 14.0%** vs the old top_t=1536 run's 31.2% on the same 5. Running 5-task avg @15k = **12.4%** vs old 31.2%. Less than half.
- This is despite **forgetting being essentially solved** (see below). The loss is entirely current-task FIT, not retention.

### The disconnect (this is the whole entry)

| axis | metric | old run (c0.01, top_t=1536) | NEW run (SupCon, top_t=512) |
|---|---|---|---|
| **interference** | seq pairwise read IoU (mean) | 0.107 | **0.0165** (6.5× lower) |
| | read-through-overwrite (t0–t4) | 59–86% | **0–6%** |
| | held-out audit family IoU (L14) | 0.349 | **0.133** |
| **capacity** | seq L14 read effnum | ~14–22K | **~2.0–2.7K** (7–9× lower) |
| | seq L14 core50 | ~3,000 | **~345** |
| | seq per-task MIN MSE | 0.089–0.255 | **0.18–0.34** (1.5–2.6× worse) |
| **outcome** | cold-start diagonal (5-task) | 31.2% | **14.0%** |

Every interference metric improved by 5–9×. Every capacity metric regressed by 5–9×. They move together because **they are the same quantity — routing breadth — read from two directions.** A task's read footprint IS its interference surface AND its expressive capacity (number of rank-2 LoRA transforms it can mix). Compacting it cuts both.

### Evidence chain (pretrain → audit → sequential)

**1. The pretrain prior was already weaker, and we had the warning.** Held-in libero_90 eval: **58.6% @20k / 73.6% @40k** vs control **76.4 / 81.1** (−18pp / −7.5pp). Pretrain MSE 0.164 vs 0.136 (+20%). gate 0.92 vs 0.98 (trusts memory less). Aggregate effnum 4968 vs 10641, used_frac 0.094 vs 0.206 (uses <half the table). The held-in regression was visible at the 20k eval and was NOT gated — the pipeline ran straight through to sequential.

**2. Query collapse confirmed at all layers.** `query_intra_sim` = **0.990–0.996** (control 0.83), `query_inter_sim` = **−0.016** (near-orthogonal). Measured on the per-sample routing query `z` (mean over tokens×heads, `memory_lite.py:785`). The `negatives_only=true` flag removed same-task samples from the SupCon denominator — i.e. it removed the **intra-task uniformity pressure** that was the only force keeping per-task queries spread. With only the positive pull and no counter-pressure, same-task queries collapsed to near-parallel → routing compacts to a tiny footprint.

**3. The 40k audit "passed" harder than the probe — and that was the trap.** Held-out L14: c005@40k core50 **511** (control 2,643), family IoU **0.133** (control 0.349), background **0.042** (control 0.127). The audit measured exactly the structure we designed for and confirmed it. **But the audit had no capacity gate.** The 7× footprint shrink sat right there in the probe-C audit table (Entry 20: effnum 2,070 / core50 351) and we read "7× smaller footprints" as purely good (less collateral) — missing that it is identically a 7× capacity cut.

**4. Sequential: forgetting solved.** Pairwise read IoU 0.0165 (old 0.107). Read-through-overwrite 0–6% (old 59–86%). Self-write coverage 78–93%. The interference problem chased across Entries 1–19 is gone.

**5. Sequential: plasticity destroyed.** Per-task min MSE 1.5–2.6× worse than old on every task, while actively training on it (t5: 0.205 vs 0.089; t3: 0.248 vs 0.161). The signature is structural: the previously-EASY task collapsed most (t3 stove+moka cold-start 68→28, −40), while the previously-HARDEST task was unchanged (t4 soup+cheese basket 8→10, +2). Capacity loss hammers tasks that needed capacity; the already-starved hard task has nothing more to lose.

### Root cause

`negatives_only=true` at weight 0.05 over-compacted routing. The flag was added (Entry 2) to remove intra-task "query-space saturation," but that uniformity was **load-bearing for capacity**: it kept each task's routing broad enough to mix enough rank-2 transforms. Removing it let SupCon collapse intra-task queries (0.994), which shrank footprints 7×, which crushed per-task expressivity in the frozen-backbone adaptation regime.

Why the held-in regression (−7.5pp) under-predicted the sequential collapse (−55%): held-in eval trains the **whole backbone**, which compensates for weak/narrow memory. Sequential **freezes the backbone** and adapts memory values ONLY, so the memory capacity loss is fully exposed and unmasked.

### top_t=512 is NOT the cause (and raising it back won't help)

New-run uniq updated slots/task at L14 = ~2,500–3,700, despite `top_t=512`. Old run at top_t=1536 = ~28–39K. If top_t were binding, new would be ~9–13K (½–⅓ of old). It's far less — because the compact PRIOR only routes meaningful TF mass to ~2–3K distinct slots, so there aren't 512 high-TF slots to fill, let alone 1536. **Write budget is slaved to the read footprint; the read footprint is set by the prior.** Reverting top_t is not the lever.

### Reframe: we have now mapped both ends of the routing-breadth axis

- **Broad routing** (control prior, top_t=1536): high capacity, high interference → 34.4% final, **interference-limited**.
- **Compact routing** (SupCon negatives_only prior): ~zero interference, low capacity → ~12% partial, **capacity-limited**.

The target is between. The mistake was treating interference as the sole objective and validating against an interference-only proxy. The audit was necessary but **not sufficient — it needs a capacity/plasticity gate**, and the held-in eval gate must actually block the pipeline.

### Methodology fixes (adopt before the next pretrain)

1. **Add a capacity gate to the held-out audit:** per-task L14 effnum/core50 must stay within ~2× of control, not just family IoU down. Reject any prior with core50 < ~1,500 (≈ control/1.8).
2. **Gate the pipeline on held-in eval @20k:** if pretrain held-in < ~0.9× control at 20k, stop before sequential. (Would have caught this at 58.6 vs 76.4.)
3. **Cheap plasticity probe:** a 1-task, ~500-step sequential fit on the frozen prior — if cold-task MSE can't reach the control prior's level, the prior is too compact. Minutes, not 45h.

### Next experiments (priority order)

1. **Drop `negatives_only` → standard SupCon, weight 0.05, queue 512** (10k probe + audit + capacity gate). Standard SupCon keeps same-task samples in the denominator → retains intra-task uniformity (capacity) while still pushing inter-task apart (separation). Directly targets the diagnosed cause. Expect: query_intra_sim back toward ~0.85–0.9, footprints between control and the failed run, family IoU still below control.
2. **If still over-compact, sweep weight DOWN: 0.05 → 0.02 → 0.01** (negatives_only either way). We jumped straight to a strong setting; the sweet spot on the breadth axis is likely a *mild* separation nudge that trims family IoU 0.35→~0.25 while keeping core50 ≥ ~1,800.
3. **Re-anchor the target metric:** stop maximizing separation. The objective is max sequential success = f(capacity, interference); both have to stay in band. Track core50 and read IoU jointly; pick the prior that minimizes interference *subject to* core50 ≥ ~1,800.
4. Only after a capacity-preserving separated prior exists: re-run sequential at top_t≈768–1024 (now that footprints are mid-sized again). *(A sequential-side write-rule was mentioned here originally — struck; sequential-side schemes are off the table.)*

### Status

- Run killed; GPU free. Pretrain `...c0.05_negonly_q512_40k` checkpoints (10k/20k/30k/40k) and `audit_heldout_c005_40k` retained for reference. Partial sequential (`..._top_t_512`, 5 task JSONs + evals to step 15k) retained as the documented negative result.
- The Entry 20 conclusion ("SupCon package is the breakthrough") is **revised**: the package separates routing as advertised, but `negatives_only` makes it overshoot into capacity collapse. The *idea* (query-space separation transfers to held-out families) stands; the *dose/variant* was wrong.

### Decided next experiment — 2-knob isolation (with Josh, 14 Jun)

**Goal restated in light of Entry 21:** we are NOT trying to minimize interference anymore — that's solved and over-solved. The goal is to find the point on the routing-breadth axis that **minimizes interference *subject to* keeping per-task capacity in band.** We have both endpoints; we need the middle.

**Three points on the map (2 new 10k probes + the existing failed run):**

| cell | negatives_only | weight | status | what it isolates |
|---|---|---|---|---|
| (existing) | true | 0.05 | FAILED (Entry 21): over-compact | — |
| **probe 1** | **true** | **0.025** | new 10k | the **dose** knob (same structure, half strength) |
| **probe 2** | **false** | **0.05** | new 10k | the **structure** knob (keeps intra-task uniformity = capacity) |

Both: queue 512, sep 0.25, loc 0.25, layers [8,10,12,14], knn 36, rank 2 — identical to the failed run except the one varied knob. Same compressed 10k schedule as probe-C (steps=10000, decay auto-scales to 10k) so they compare apples-to-apples against `audit_heldout_probeC_10k`. Throwaway screens — no resume; the winner gets a FRESH 40k.

**Mechanistic prediction (pre-registered):**
- Probe 2 (drop negonly) is the favored fix. Standard SupCon keeps same-task samples in the denominator → retains the intra-task uniformity that is load-bearing for capacity, while still pushing tasks apart. Expect query_intra_sim to settle ~0.85–0.90 (not 0.99), core50 to land mid-range, family IoU still below control.
- Probe 1 (dose-down negonly) may only *delay* collapse — negonly has no anti-collapse term at any weight. Expect partial capacity recovery but a real risk it's still too narrow at 10k (query_intra_sim ~0.95+). If so, the verdict is "negonly is the problem, not the dose."

**What GOOD looks like (graduates to a fresh 40k + sequential):** lands strictly between the two anchors on BOTH axes —
- capacity retained: held-out L14 **core50 ≥ ~1,500** (control 2,643; failed 351–511), and in-run **query_intra_sim ≤ ~0.90** (control 0.83; failed 0.99), routing_intra_task_support_L14 ≥ ~5,000 (control 7,800; failed 3,400).
- separation retained: held-out L14 **family IoU ≤ ~0.28** (control 0.349; ≥20% below control), background IoU below control's 0.127.
- fit not wrecked: pretrain MSE within ~10% of control at matched step; if a 40k follows, held-in eval @20k ≥ ~0.9× control (69%+).

**What BAD looks like:**
- *Over-compact (same failure):* core50 < ~1,000 or query_intra_sim > 0.95 → capacity will collapse in sequential regardless of how good the IoU looks. This is the trap from Entry 21 — IoU will look *great* here; ignore it and read capacity.
- *No effect:* family IoU ≥ ~0.32 (≈ control) → the knob bought no separation; we gave up nothing but gained nothing.
- *Both in band but weak:* if both probes land mid-range, pick the better capacity/separation balance and note the cheaper knob.

**Decision rule after the probes:**
- If exactly one lands in band → it graduates to fresh 40k + sequential.
- If both → graduate probe 2 (mechanistically cleaner; capacity-preserving by construction), keep probe 1 in reserve.
- If neither → we've bracketed: next is negonly=false at a *higher* weight (0.1) to separate harder while keeping the anti-collapse term, OR negonly=true at 0.0125. No blind sequential runs until a prior clears the capacity gate.

**Reminder — the probes do NOT prove success.** They confirm the prior sits in the healthy routing band. Sequential success is still f(capacity, interference) and must be measured by a real sequential run on the graduating prior. Before that sequential, also run the cheap 1-task ~500-step plasticity probe (Entry 21 methodology fix #3) as a final capacity check.

Scripts: `job_scripts/nebius/libero_90/probes/{probe_10k_negonly_c0.025.sh, probe_10k_standard_c0.05.sh, run_probes2_seq.sh}`.

---

## Entry 22 - 15 Jun 26 (2-knob isolation results: negonly dead, standard SupCon under-separates — bracketed)

### Results (10k probes, held-out audit + in-run @10k)

| run | L14 core50 | L14 effnum | query_intra_sim | famIoU | bgIoU | verdict |
|---|---|---|---|---|---|---|
| control (broad) | 2,643 | 14,991 | 0.79 | 0.349 | 0.127 | interference-limited baseline (34% seq) |
| neg 0.05 (FAILED, E21) | 511 | 2,913 | 0.99 | 0.133 | 0.042 | capacity-dead (~12% seq) |
| **P1 neg 0.025** | **696** | 4,019 | **0.98** | 0.244 | 0.096 | **BAD: still capacity-dead** |
| **P2 std 0.05** | **1,465** | 8,499 | **0.91** | **0.338** | 0.126 | **BAD: ~no separation (≈control)** |

Gate was: GOOD = core50 ≥ ~1,500 AND query_intra_sim ≤ ~0.90 AND famIoU ≤ ~0.28. **Neither probe passes; they fail at opposite ends.**

### Pre-registered predictions vs outcome

- **P1 ("negonly dose-down may only delay collapse") — CONFIRMED, strongly.** Halving weight 0.05→0.025 moved capacity essentially zero: core50 696 vs failed 511, query_intra_sim 0.98 vs 0.99, effnum 4,019 vs 2,913, support_L14 4,312 vs 4,207. negonly has no anti-collapse term at any weight; dosing down only slightly softens a structural collapse. **The dose is not the lever; the negonly branch is dead.**
- **P2 ("favored fix: capacity preserved, separation milder but still below control") — HALF RIGHT.** Capacity preserved ✓ (core50 1,465 ≈ 3× failed, query_intra_sim 0.91, effnum 8,499). But separation did NOT come through ✗: famIoU 0.338 vs control 0.349 = 3% (nothing); bgIoU 0.126 vs 0.127 (nothing); in-run inter-task query sim 0.185 actually ABOVE control 0.140. Standard SupCon's same-task-in-denominator term protects capacity AND neutralizes the inter-task push at 0.05. std-0.05 ≈ control with slightly tighter intra-clusters.

### Conclusion — the contrastive frontier has a structural problem

- **negonly couples capacity and separation:** the same query collapse drives both low overlap and low capacity. Weakening it relaxes both toward control together (0.05→0.025: core50 511→696, famIoU 0.133→0.244). Its entire frontier is bad — there is no point on it with core50 ≥ 1,500 AND famIoU ≤ ~0.28.
- **standard SupCon decouples them** (capacity safe) **but is too weak to separate** at 0.05. Its dominant effect is intra-task clustering, not inter-task pushing.
- Note on training-amount: control routing is ~flat over training and failed-negonly relaxed slightly 10k→40k (core50 351→511), so P2's 1,465 at 10k would likely clear ~1,500 at 40k. Capacity is borderline-OK; **separation is the dealbreaker, and it won't improve with more steps** (property of the weak loss, not undertraining).

### Open question (sharp)

Can we get held-out family separation INTO the capacity-preserving regime? The thing we actually want is **footprint translation (disjoint but broad), not shrinkage.** The query-space contrastive has now failed at this twice — negonly separates only by collapsing; standard doesn't separate. Two untested mechanisms:

1. **Standard SupCon weight up (0.05 → 0.1):** does its own separation rise with weight before capacity collapses? Risk: dominant effect is clustering, so it may walk toward intra-collapse rather than inter-separation.
2. **Direct slot-space separation:** standard SupCon 0.05 + `routing_inter_task_separation` 0.25 → 0.5. Pushes task slot-distributions apart directly (translation, not shrinkage) — never swept in the pi05 regime. Mechanistically the most targeted at the decoupling; favored bet.

(Note: locality is NOT the compaction driver — control carries locality 0.25 and is broad, core50 2,643. Compaction came specifically from the strong negonly contrastive. So locality-off is not the lever; leave it unless a separation sweep shows it amplifying compaction.)

### Next experiment (LAUNCHED 15 Jun, tmux `probes3` — Josh: "do both") — 2-probe isolation, mirrors E21

- **Probe 3:** standard SupCon **0.1** (negatives_only=false, queue 512), all else = control. Isolates the contrastive-weight axis on the capacity-safe variant.
- **Probe 4:** standard SupCon **0.05** + `routing_inter_task_separation` **0.5** (negatives_only=false, queue 512), all else = control. Isolates direct slot-space separation.
- Same compressed 10k schedule + held-out audit + same gate (core50 ≥ ~1,500 AND query_intra_sim ≤ ~0.90 AND famIoU ≤ ~0.28). Throwaway screens; winner → fresh 40k + 1-task plasticity probe + sequential.
- GOOD: a point with famIoU ≤ ~0.28 while core50 ≥ ~1,500 (the decoupled frontier we haven't found yet). BAD: famIoU ≈ control (no separation) OR core50 collapse / query_intra_sim > 0.95.
- Reserve if both fail: separation 0.5 + locality 0 (test compaction amplification), or other pretraining-side separation formulations (e.g. similarity-weighted separation). Pretraining-side only.

### Status
Probes 1/2 completed (neither earned a 40k). Checkpoints + audits (`audit_heldout_{negonly_c0.025,standard_c0.05}_10k`) retained. Probes 3/4 LAUNCHED 15 Jun in tmux `probes3` (scripts `probe_10k_standard_c0.1.sh`, `probe_10k_standard_c0.05_sep0.5.sh`, runner `run_probes3_seq.sh`); audits `audit_heldout_{standard_c0.1,standard_c0.05_sep0.5}_10k`. ~23.5h. No 40k launched yet.

### Note: discussion after launching probes 3/4 (mechanism + capacity diagnostics)

Two clarifications worked out while probes 3/4 run. Both refine how to read the results; no config changed.

**(a) SupCon (negonly=false) vs `routing_inter_task_separation` are NOT the same loss.** Verified against code (`memory_lite.py`: `_compute_sample_contrastive_loss` L841 vs `_compute_routing_losses` L969). Three differences, two of which are exactly the bet:
1. **Space.** SupCon = cosine between per-sample query *vectors* `z=mean_{T,heads} q` in continuous k_dim, **before** the key lookup. Separation = cosine between per-task *slot-occupancy histograms* over the n_keys² space, **after** top-M→Cartesian→softmax retrieval. The query→slot map (through learned keys) is many-to-one and nonlinear, so query separation is only an indirect proxy for slot overlap — the Entry 5/7 finding. Live proof in our data: P2 (std SupCon 0.05) moved query geometry but left held-out famIoU at 0.338 ≈ control.
2. **Intra-task pull.** SupCon's numerator pulls same-task queries together (the force that collapsed query_intra_sim→0.99 and shrank footprints) — structurally inseparable from the contrastive form. Separation has **no** same-task term (pure `i<j` cross-task cosine on aggregated histograms); intra-task breadth is handed to the *decoupled* locality band. This is why separation can in principle reduce overlap by **translating** broad footprints to disjoint regions rather than shrinking them.
3. **Granularity.** SupCon is instance-level (every sample's query position matters); separation is distribution-level (only each task's aggregate histogram matters), so it tolerates internally-diverse within-task routing.
- **Caveat / failure mode to watch:** separation can still reduce overlap via the cheap **shrink-to-disjoint** shortcut (tiny private supports also have ~0 cosine). The intended guardrail is the locality band's *min-support floor* (`relu(min_entropy − task_H)`), but its current calibration ([128, 2048]) sits far below control's healthy support (~7,800), so the floor may not bind. Hence the gate is famIoU↓ **AND** core50≥~1,500 *together*; reserve fix is raising the min-support floor, not turning locality off.

**(b) Subkey-level capacity decomposition (held-out audit, L14).** "How many keys per task?" — decomposed each task's slot histogram into its two PQ half-subkey marginals (slot = i1·384 + i2):

| run | eff subkeys/half (of 384) | binary subkeys/half | eff slots |
|---|---|---|---|
| control (broad) | 192 | 384 | 14,991 |
| neg0.05 (FAILED) | 98 | 371 | 2,913 |
| P1 neg0.025 | 112 | 380 | 4,019 |
| P2 std0.05 | 152 | 384 | 8,499 |

- **Not a binary-restriction story:** nearly all 384 subkeys are touched in every run (coupon-collector over millions of retrievals). The collapse is in *effective* (mass-weighted) count: 192→98 per half.
- **Slot collapse is multiplicative.** eff_slots ≈ eff_half₁ × eff_half₂ × corr. control→failed: per-half (192/98)² ≈ 3.8×, plus the two halves become more correlated (joint/product ratio 0.41→0.30, ×1.37) ⇒ ≈5.1× slot drop from a 2× per-half drop. Query collapse makes (q1,q2) near-constant → the same (i1,i2) pairs co-win → halves lock. ~75% of slot collapse = per-half concentration, ~25% = half-correlation. Both trace upstream to Q.
- **Sharp statement of collapse:** per query, 4 heads × top-36 spans up to ~144 subkeys/half; the failed run's *whole-task* effective key count (98) is **below a single query's cross-head span** — within-task query variation adds ~nothing. Control (192) ≈ 2× a single query → genuinely diversifies.
- **Key-level target for probe 4:** eff subkeys/half toward ~190 **and** joint/product ratio toward ~0.41 while famIoU drops. New failure signature to watch: separation that cuts famIoU by *re-correlating* the halves (ratio falls) = shrinkage shortcut at the key level, even if per-half effnum looks ok.

**(c) Plain-English on eff_slots ≈ 2,900 (failed run).** It's the aggregate footprint over ALL of a task's observations (~500 slots carry the first half of the weight, tail brings effective total to ~2,900). But query_intra_sim 0.99 means every observation pulls nearly the *same* ~36-slot mixture, so that palette is addressed almost state-independently — aggregate footprint **overstates usable state-conditional capacity**. This is why ~2,900 slots still gave capacity-starved fits (per-task MSE 0.18–0.34): the memory acts closer to a per-task bias than a rich state→action map. Control's ~15k is not just 5× more slots but slots addressed far more state-distinctly.

**(d) Layerwise per-batch reads vs top_t — when is the write budget actually binding?** top_t only "reduces available adaptation params" if it is BELOW the per-batch effective read breadth at a layer; otherwise it is a no-op (you can update everything you read). Per-batch effnum (mean over training, by layer):

| layer | OLD seq (control prior, top_t=1536) | FAILED seq (supcon prior, top_t=512) |
|---|---|---|
| L8 | 1,631 | 331 |
| L10 | 2,296 | 356 |
| L12 | 3,587 | 487 |
| L14 | 5,520 | 784 |

Per-batch write coverage = min(top_t, reads)/reads. OLD L14: 1536/5520 = **28%** (heavily write-limited — the Entry 16/17 finding). FAILED L14: 512/784 = **65%**, and ≥100% at L8/L10/L12. So top_t=512 was matched-to-generous for the collapsed prior — the failed run updated a LARGER fraction of its reads than the 34% old run did. This rejects "we hit capacity twice (supcon + low top_t)": the two interact rather than add. Reads collapsed 5,520→784 at L14 (7×, the prior's doing); top_t=512 rode behind that tighter bottleneck and was a near-no-op. **Rule going forward: top_t is binding relative to read breadth, so set it from the winning prior's measured per-batch L14 effnum (target ~70–90% coverage), not a fixed carryover. A broad-but-separated probe-4 prior (L14 reads back toward ~5,000) will make top_t=512 binding again → use ~1536 for that sequential.**

---

## Entry 23 - 16 Jun 26 (Root-cause fix: cross-batch queue for the separation loss + rq512 rerun)

### Why probes 1–4 may have been unfair to separation (code finding)

Traced the two routing losses in code (`memory_lite.py`). The **contrastive** loss uses the cross-batch queue (`contrastive_query_queue`, =512 in our runs) — it concatenates 512 detached query vectors from prior batches. The **separation/locality** loss (`_compute_routing_losses`) does **not**: it operates on `_compute_subkey_scores(current query)` only, grouping the current micro-batch by task. With batch 32 over 90 tasks (random sampler), that means:
- **~1 sample/task** per step (≈27 distinct tasks/batch, mostly 1 sample each) → each task's slot-histogram in the separation loss is a single-observation estimate.
- **~9% pairwise coverage**: any two specific tasks co-occur in a batch only ~(27/90)² of steps, so the exact pairs we need to separate (e.g. basket family) get a gradient <1 step in 10, and no step ever sees the global 90-task structure.

So separation has been operating on a noisy, sparsely-covered estimator. The Entry-22 "separation can't decouple" conclusion is **likely premature** — separation may simply never have had a clean signal. (Josh caught this.)

### Fix implemented: routing-separation cross-batch queue

New config `routing_query_queue` (in SAMPLES; 0 = off, current behavior). When >0, `_routing_losses_queued` (new) runs a dense-histogram path:
- **Current batch** → differentiable per-task slot histograms over the full n_keys² space (dense == compact numerically; carries the gradient).
- **Queue** → FIFO of per-token detached queries (highest granularity, global FIFO, per Josh). Each step the queued queries are **recomputed against the CURRENT keys** (under no_grad) → detached per-task reference histograms covering all recently-seen tasks. Recompute-vs-current-keys (not frozen histograms) is deliberate: the separation loss *moves* the keys, so frozen references would lag.
- **Separation** = push each current (differentiable) task histogram away from all reference (detached) task histograms j≠i, vectorized as one `einsum('ihs,jhs->ij')/heads` with an i==j mask. Fixes both the 1-sample estimate and the 9% coverage (every present task pushed against ~all 90 references every step).
- Locality / global-balance stay on the current differentiable histograms (unchanged). Queue-off path is byte-for-byte the old compact code.

Wiring: per-token queries staged in `forward` (guarded by `_is_checkpoint_recompute()` to avoid double-enqueue under grad checkpointing), flushed after the optimizer step via the existing `flush_staged_contrastive_queries` hook (lerobot_train.py:184).

**Bug caught by smoke test (would have silently disabled the queue):** the existing flush had an early `return` when no *contrastive* entries were pending → the routing flush was skipped whenever contrastive was off. Restructured to flush both independently.

Smoke-tested in isolation (tiny module): queue populates to cap; **a single-task batch still gets a separation loss via references** (the coverage fix); gradients reach query_proj + keys; checkpoint-recompute guard holds; queue-off numerically identical to the old path; vectorized einsum identical to the per-pair loop. Files: `memory_config.py`, `memory_lite.py`.

### Rerun (LAUNCHED 16 Jun, tmux `probes5`)

Same configs as probes 3/4, only delta = `routing_query_queue=512`, so this isolates the queue's effect against the no-queue probes3 audits:
- **probe 3':** standard SupCon 0.1 + rq512 → `..._probe10k_standard_c0.1_rq512`
- **probe 4':** standard SupCon 0.05 + sep 0.5 + rq512 → `..._probe10k_standard_c0.05_sep0.5_rq512` (favored)
- 10k each, audits `audit_heldout_standard_{c0.1,c0.05_sep0.5}_rq512_10k`. Scripts under `probes/`; runner `run_probes5_seq.sh`.

What GOOD looks like (the test of the whole hypothesis): with the estimator fixed, separation should finally move **held-out** famIoU below control's ~0.349 **while** core50 stays ≥ ~1,500 — the decoupled point probes 1–4 couldn't reach. If rq512 still doesn't move held-out famIoU, that's much stronger evidence (now with a fair estimator) that the current separation formulation can't separate held-out near-duplicates under a frozen router → next moves stay PRETRAINING-side (similarity-weighted separation, stronger/longer pretrain, different routing-loss formulation). If it DOES move, the weight sweep (sep 2.0 / contrastive 0.2, possibly 20k) becomes worthwhile on top of the queue. [See Entry 24 for the actual result + decision.]

---

## Entry 24 - 17 Jun 26 (probes5/rq512 verdict: the queue fixes the ESTIMATOR, not the transfer — separation-metric artifact diagnosed, loss-magnitude audit, aggressive sep=2.0 probe launched)

### Headline

probes5 (rq512 reruns of P3/P4) is the fair test Entry 23 set up. Verdict: **the cross-batch routing queue works exactly as designed — it makes the separation signal honest — but at the tested doses (sep 0.25/0.5) prior-side separation STILL does not transfer to held-out lookalike families.** Held-out famIoU stays ≈ control. This is the §7 FAIL branch, but with a crucial caveat surfaced by a loss-magnitude audit: 0.25→0.5 was too small a lever to conclude, and separation is NOT drowned out by MSE. So before declaring separation dead we are running one aggressive, capacity-gated probe at **sep=2.0** (contrastive held at 0.05 to remove the confound). En route we diagnosed a metric artifact that nearly misled us.

### probes5 held-out audit (the §7 decision metric)

Audit pipeline validated against anchors (reproduces handover/Entry-22 numbers exactly: control core50 2643 / famIoU 0.349 / effnum 14991; failed-negonly 511 / 0.133 / 2913).

| run (held-out L14) | core50 | effnum | famIoU | bgIoU |
|---|---|---|---|---|
| control@40k (BROAD anchor) | 2643 | 14991 | 0.349 | 0.127 |
| c005@40k (COLLAPSED anchor) | 511 | 2913 | 0.133 | 0.042 |
| P3 no-queue (std c0.1) | 1162 | 6839 | 0.381 | 0.111 |
| **P3' rq512 (std c0.1)** | 979 | 5865 | 0.356 | 0.118 |
| P4 no-queue (c0.05 sep0.5) | 1752 | 10041 | 0.342 | 0.117 |
| **P4' rq512 (c0.05 sep0.5)** | 1501 | 8694 | 0.350 | 0.120 |

Gate = famIoU ≤ ~0.28 AND core50 ≥ ~1500 AND query_intra ≤ ~0.90. **Neither rq512 run clears it:** P4' famIoU 0.350 (fails), P3' famIoU 0.356 + core50 979 (fails both). The queue moved held-out famIoU by ≤0.01 vs its no-queue twin — and if anything made footprints marginally *more compact* (core50 1752→1501), the opposite of "more separated." So the queue did not unlock held-out separation at these weights.

### The separation-metric artifact (resolves a confusion that nearly misled us)

**The chart that confused us:** turning on the queue *raised* the logged `routing_inter_task_similarity_mean` (~+0.03–0.04; L14 0.121→0.160), i.e. `routing_inter_task_separation_mean` (=1−v) *dropped* 0.90→0.87. It looked like adding the queue + raising sep weight REDUCED separation — backwards from intent.

**Root cause (code):** the logged similarity is computed *inside* the routing-loss fn, and the queue swaps that fn for a different estimator. Same metric name, different quantity:
- no-queue `_compute_routing_losses` (L1189-98): mean pairwise cosine between **current-batch** per-task histograms → **sparse-vs-sparse**, only the ~27 in-batch tasks.
- rq512 `_routing_losses_queued` (L1334-44): cosine between each current task's **single-sample** histogram and the queue's **dense, aggregated, all-90-task reference** histograms → **sparse-vs-dense**, full coverage.

Over a 147,456-slot table, two single-sample histograms drawn from the *same* broad distribution (control effnum ~15K) mostly hit different slots → sparse-vs-sparse cosine is biased toward 0 → the old metric **systematically under-read overlap and flattered separation**. The dense all-task reference removes that bias (and the ~9%→100% pair-coverage bias). The queue made the metric **honest**; "separation got worse" is the flattery being stripped out.

**Three independent checks that it's measurement, not a real regression:**
1. **Query geometry identical** (computed in the contrastive path, untouched by `routing_query_queue`): query_inter_sim P4 0.186 vs P4' 0.183; query_intra 0.911 vs 0.911. The learned query map didn't change.
2. **Held-out audit identical** (offline, same method all runs): P4 famIoU 0.342 vs P4' 0.350.
3. **Trajectory shape:** the queue offset is present from step 200 and the curves run parallel down to 10k — a constant offset, not a divergence.

**Cleanest single statement:** no-queue P4 logs *lower* in-run similarity (0.095) than queue P4' (0.128), yet both have the *same* held-out famIoU (~0.345 ≈ control). The in-run delta is 100% estimator; the ground truth is identical. **Rule: never compare in-run `routing_inter_task_similarity` across queue on/off — only within a fixed estimator, or via the held-out audit.**

### What "near-duplicate" means here + a hidden-state correction

famIoU is measured strictly **held-out ↔ held-out** — among the libero_10 basket family (dataset task_index 4/5/7), NOT libero_90 ↔ libero_10:
- t4 "put both the alphabet soup and the cream cheese box in the basket"
- t5 "put both the alphabet soup and the tomato sauce in the basket"
- t7 "put both the cream cheese box and the butter in the basket"

The **cause** routes through libero_90: these are compositions of single-object basket primitives that libero_90 covers densely ("pick up the alphabet soup/cream cheese box/tomato sauce/… and put it in the basket"). Near-identical instructions → the frozen language-conditioned router drops them into the shared pretrain "…in the basket" basin. Per-pair overlap tracks shared content monotonically (control@40k L14): t4–t5 (share soup) **0.391** > t4–t7 (share cream cheese) **0.355** > t5–t7 (share only the "…basket" structure, disjoint objects) **0.302** ≫ background **0.127**. Even the no-shared-object pair sits at 2.4× background.

**Correction to Entry 19's "FiLM-on-language" framing (Josh's catch):** the query is `q = proj(x)·(1+γ(lang)) + β(lang)` (`memory_lite.py` L143-155) — a projection of the **action-expert hidden state** `x`, FiLM-modulated by language. γ/β are language-only (constant across a task's frames), but `proj(x)` varies per observation and carries the visual scene (different objects on the table). So there *is* discrimination signal beyond the instruction string; the "irreducible floor from identical language" claim was too strong. The floor is softer than stated — which is part of why pushing separation harder is worth a real test (and motivates future direction #2).

### Loss-magnitude audit (answering "what can we actually play with")

Loss assembly (`memory_lite.py` L1734-1755): `loss = MSE + Σ weightᵢ·rawᵢ`, every raw term logged. Reconstructed weighted contributions @10k:

| run | MSE | contrastive raw→**wtd** (%MSE) | sep: sim→**wtd** (%MSE) | locality wtd |
|---|---|---|---|---|
| P3 (c0.1, sep0.25) | 0.213 | 1.93→**0.193** (91%) | 0.098→**0.025** (12%) | 0.001 |
| P4 (c0.05, sep0.5) | 0.212 | 1.97→**0.098** (46%) | 0.095→**0.047** (22%) | 0.001 |
| P4' rq512 (c0.05, sep0.5) | 0.212 | 1.97→**0.098** (46%) | 0.128→**0.064** (30%) | 0.001 |

Findings:
1. **Contrastive is the core-compaction knob — it (not separation) collapsed P3.** core50 is monotone in *contrastive* weight: control 0.01→2643, P4 0.05→1752, P3 0.1→1162; query_intra_sim 0.79→0.91→0.93 in lockstep. SupCon's intra-task pull tightens each task's query cloud → shrinks footprints. At c=0.1 the weighted contrastive is 91% of MSE (co-dominant) — that's the compaction pressure. **This is why contrastive must stay fixed/low while we probe separation.**
2. **Locality is dead weight** — 0.001, ~0.6% of MSE (confirms Entry 19/20; not a usable lever).
3. **Separation is a real term, not drowned out** — 30% of MSE at sep=0.5 (rq512). So "too low on the scale" in the *negligible* sense is false.
4. **But the weight-response is weak:** doubling sep 0.25→0.5 moved the seen-task similarity it directly minimizes only 0.098→0.095 (~3%), and held-out famIoU 0.349→0.342 (nothing). Either saturation against MSE's pull or a seen-task sharing floor. A 2× change can't distinguish "scale-limited" from "can't transfer" — hence the aggressive jump.

### Decision + launched (probe 6)

One decisive, cheap probe: **sep=2.0** (8× P3, 4× P4), **contrastive=0.05 FIXED** (isolate separation, preserve capacity — Josh's call to avoid confounds), **rq512** (clean gradient), same compressed 10k schedule. Capacity-gated audit.

- Run: `libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep2.0_rq512`; scripts `probes/{probe_10k_standard_c0.05_sep2.0_rq512.sh, run_probe6_seq.sh}`; tmux `probe6`; wandb `r1sklapt`; ETA ~11h pretrain + ~35min audit.
- **Decisive gate (jointly):** held-out L14 **famIoU ≤ ~0.28 AND core50 ≥ ~1500**.
  - famIoU↓ **with** core50 held → real translation; separation was scale-limited → sweep upward, then 40k.
  - famIoU↓ **only** with core50 < ~1500 → shrink-to-disjoint shortcut (the locality min-support floor [128,2048] is too weak to block it — the capacity gate is the guard, not the loss). Reserve fix: raise min_support.
  - famIoU ≈ control (0.349) → separation conclusively cannot transfer to held-out lookalikes under the frozen router; the residual ~0.30 is the genuine-sharing floor → pivot off the interference axis to the capacity/co-host axis (realworld Entry 3: rank / collision-aware protection).

### Future directions (noted, non-prescriptive — Josh)

1. **20k re-test** of the winning recipe — the probes run a compressed 10k schedule; if separation is borderline, more router-training steps may be the genuine bottleneck (handover flagged 10k as possibly under-training the router for separation). Cheap to test before committing to 40k.
2. **Hidden-state vs language contribution to the query.** Quantify how much `proj(x)` (scene) vs `γ/β(lang)` drives routing for lookalike-language tasks, then reweight the fusion so the scene carries more signal — so basket tasks separate by *what's on the table* rather than collapsing on near-identical instructions. Pretraining-side, no new params at adaptation. Directly targets the held-out-transfer gap rather than fighting it with global separation weight.
3. **Per-layer aggressive separation at higher layers.** L14 is consistently the resistant, highest-overlap/highest-trust layer (L14 sim 0.16 vs mean 0.13 at rq512). A per-layer separation weight concentrated on L12/L14 may bite where a uniform weight can't.

### What we are not doing

- Not raising contrastive (it's the compaction confound; capacity-killer above ~0.05).
- Not reading in-run `routing_inter_task_similarity` across queue on/off as comparable (estimator artifact; use the held-out audit).
- Not touching locality (inert at 0.6% of MSE), idf_exponent, dropout, or sequential-side schemes (all measured ~no-ops / off the table).
- Not committing to a 40k or a sequential run until a prior clears the held-out capacity+separation gate jointly.

---

## Entry 25 - 17 Jun 26 (sep=2.0 DECOUPLES — the first separation win without capacity loss; 4-probe batch launched to isolate contrastive / sep-curve / locality)

### Headline

The aggressive probe from Entry 24 (P6: c0.05 / **sep2.0** / loc0.25 / rq512) is **the first prior in the entire project to reduce held-out interference WITHOUT collapsing capacity.** Held-out basket-family IoU fell 0.349→**0.311** (background 0.127→**0.099**) while per-task capacity went *up* (core50 1501→**2368**, ≈ control's 2643), at zero MSE cost and no query collapse — and with none of the shrink-to-disjoint signatures. This overturns the Entry-22 "separation can't decouple" conclusion: it was an artifact of the broken estimator (no queue) + too-low weight (≤0.5). With the clean estimator (rq512) **and** adequate weight (2.0), separation translates footprints apart instead of shrinking them. Josh's Entry-24 instinct ("we could just be too low on the scale") was right.

### The sep=2.0 result (P6), held-out audit + in-run

| (held-out L14) | core50 | effnum | famIoU | bgIoU | effK/h | j/p |
|---|---|---|---|---|---|---|
| control@40k (broad) | 2643 | 14991 | 0.349 | 0.127 | 193 | 0.40 |
| c005@40k (collapsed) | 511 | 2913 | 0.133 | 0.042 | 98 | 0.29 |
| P4' rq512 sep0.5 | 1501 | 8694 | 0.350 | 0.120 | 154 | 0.36 |
| **P6 rq512 sep2.0** | **2368** | **13248** | **0.311** | **0.099** | **183** | **0.39** |

Per-family-pair (L14): t4–t5 0.391→**0.321**, t4–t7 0.355→0.359, t5–t7 0.302→**0.254**.

In-run @10k (P6 vs P4', both rq512 → directly comparable): MSE **0.2121 vs 0.2118** (zero fit cost), query_intra **0.912 vs 0.911** / query_inter 0.182 vs 0.183 (query geometry unchanged — separation acts on slot histograms, not queries), but the seen-task similarity the loss minimizes dropped hard: **0.084 vs 0.128** mean (−34%), L14 **0.113 vs 0.160**. support broadened (L14 6084 vs 5714); effnum 10146 vs 8208; gate 0.946 vs 0.950.

### Why it's translation, not the shrink shortcut

Every collapse signature points *away* from shrinkage:
- effnum and core50 went **up** toward control (broader, not smaller footprints).
- subkeys/half **183 ≈ control 193** (not concentrated; failed-negonly was 98).
- joint/product ratio **0.39 ≈ control 0.40** (halves stayed decorrelated; failed was 0.29 — re-correlation is the key-level shrink signature).

So footprints moved apart *into distinct broad regions* — the mechanism Entry 23 hypothesized the queue would unlock once the estimator was honest. Proof that 0.5 was just too weak (not saturated): 0.25→0.5 moved the in-run target ~3%, but 0.5→2.0 moved it 34% while capacity *rose* — we're nowhere near a wall.

### Interpretation

1. **Separation decouples capacity from interference** when fairly estimated + adequately weighted. The breadth axis is no longer one-dimensional: we reduced overlap while *increasing* footprint breadth.
2. **It preferentially cleaned the incidental overlap.** Background dropped 22% vs family's 11% — separation kills the benign cross-family collateral harder than the genuine-sharing basket overlap (which shares real compositional primitives — Entry 24 discussion). That's the *right* selectivity (handover target property: overlap only at synergistic edges).
3. **Partial, not a clean gate clear.** famIoU 0.311 is below control but above the ~0.28 GOOD line; query_intra 0.912 is a hair over the 0.90 proxy but core50 (the real measure) is healthy. Clear headroom remains → hence the sep-curve sweep below.
4. **Interference only.** The dual-cycle plasticity ceiling (~40% diagonal, Entry 19) is untouched; the eventual sequential gain will show up as **retention** (no t5→t7 step-24000 cliff), not a higher peak.

### The 4-probe batch (LAUNCHED 17 Jun, tmux `probes7`)

All single-knob deltas from the P6 anchor; the question is whether **separation alone is the lever.** Runner `run_probes7_10_seq.sh` (interleaved pretrain→audit per probe); ~2 days total.

| probe | run tag | delta vs P6 | goal |
|---|---|---|---|
| P7 | `c0_sep2.0_rq512` | contrastive 0.05 → **0** | **A: is contrastive needed at all?** |
| P8 | `c0.05_sep3.0_rq512` | sep 2.0 → **3.0** | **B: sep curve** |
| P9 | `c0.05_sep5.0_rq512` | sep 2.0 → **5.0** | **B: sep curve, far end / turnover** |
| P10 | `c0.05_sep2.0_noloc_rq512` | locality 0.25 → **0** (support bands dropped) | **C: does locality do anything?** |

Hope (Josh): sep is all we need — contrastive redundant (A), locality inert (C), and the sep curve (B) locates the famIoU floor / capacity knee.

**What to look for (all vs the P6 audit 0.311/2368; gate = famIoU ≤ ~0.28 AND core50 ≥ ~1500):**
- **P7 (no contrastive):** if famIoU/core50 ≈ P6 → contrastive is redundant once sep carries the load (drop it; lose its compaction side-effect for free). If capacity *rises* and famIoU holds → even better (contrastive was only costing capacity). If separation degrades → contrastive's query-tightening was helping sep after all.
- **P8/P9 (sep curve):** expect famIoU to keep falling; watch for the shrink shortcut switching on at the far end (core50/effnum/j-p ratio dropping). P9 sep5.0 puts the weighted sep term (~0.4) above MSE (~0.21) — highest over-separation risk; capacity is the thing to watch.
- **P10 (no locality):** if ≈ P6 → locality is dead (drop permanently, as Entry 20/24 already suggest). If capacity drops materially → locality was a load-bearing anti-shrink floor after all.

### Caveats / notes

- **Contrastive read as 0.05, not 0.5.** The request said "Contrastive 0.5" for P8–P10; taken as a typo for 0.05 (the project's capacity-safe value, and the only reading under which these are clean single-knob deltas — 0.5 is 5× the 0.1 that already over-compacted, would collapse capacity, and would confound goals B/C). Proceeded rather than block since Josh stepped away; flagged for correction on return. If 0.5 was intended, P8–P10 need a rerun (P7 and the curve *shape* still stand).
- **P7 will not log `query_intra/inter_sim`** — those diagnostics live inside the contrastive block, gated by `contrastive_loss_weight > 0` (`memory_lite.py:621`). Read P7 capacity from the held-out audit (core50/effnum/subkey), which is the ground truth regardless.
- Disk cleaned to make room (see below): dead-end 10k probe **checkpoints** deleted (wandb + all audits retained).

### Next (after the batch)

Pick the winning prior on the joint axis (lowest famIoU with core50 ≥ ~2000, contrastive on/off per P7, locality on/off per P10) → fresh 40k (NOT resume — scheduler gotcha) → 40k held-out audit → 1-task plasticity probe → sequential, judged on the **retention matrix** (esp. early-task collapses), not just final average. The plasticity ceiling remains the separate, binding constraint on absolute performance.

**Update (18 Jun — P7 in, goal A answered: contrastive is LOAD-BEARING, "sep alone" disconfirmed).** P7 (c=0, sep2.0, rq512) held-out L14: core50 **7700**, effnum **37384**, famIoU **0.482**, bg **0.260** — the broadest, highest-overlap prior we've made, *worse than control* (0.349/0.127). In-run: MSE 0.201, routing_sim 0.136 (vs P6 0.084), support_L14 8175. Removing contrastive removed all intra-task compaction → footprints sprawl → separation still translates centroids apart but can't keep enormous clouds disjoint → overlap blows up. **Clean division of labor: contrastive = compaction (footprint *size*); separation = translation (footprint *position*); both required.** Breadth axis is Goldilocks: too much contrastive (negonly/0.1) → collapse; none → sprawl; **c0.05 + sep2.0 (P6, famIoU 0.311 / core50 2368) sits in the pocket and stays the frontier.** P8/P9 (sep curve) and P10 (locality) still running.

---

## Entry 26 - 19 Jun 26 (sep curve: sep=5 CLEARS the gate via translation; contrastive load-bearing, locality dead — sep=5 graduation 40k + full sequential LAUNCHED)

### Probes 7-10 complete — full results

Held-out audit (L14); all rq512, c0.05 unless noted; gate = famIoU ≤ ~0.28 AND core50 ≥ ~1500:

| run | core50 | effnum | famIoU | bgIoU | j/p | gate |
|---|---|---|---|---|---|---|
| control@40k (broad) | 2643 | 14991 | 0.349 | 0.127 | 0.40 | — |
| P4' sep0.5 | 1501 | 8694 | 0.350 | 0.120 | 0.36 | ✗ |
| P6 sep2.0 | 2368 | 13248 | 0.311 | 0.099 | 0.39 | ✗ |
| P8 sep3.0 | 2372 | 13305 | 0.309 | 0.092 | 0.39 | ✗ |
| **P9 sep5.0** | **2679** | 14881 | **0.264** | 0.087 | 0.39 | **✓ CLEARS** |
| P7 sep2.0 **c=0** | 7700 | 37384 | 0.482 | 0.260 | 0.44 | ✗ sprawl |
| P10 sep2.0 **noloc** | 2337 | 13081 | 0.309 | 0.103 | 0.38 | ✗ (≡P6) |

In-run @10k: MSE flat across the whole sep curve (0.2118 → 0.2121 → 0.2112 → **0.2102**), in-run sim falls monotonically 0.128 → 0.084 → 0.073 → 0.061; query_intra ~0.91 throughout (sep-independent — it's contrastive's knob).

### The three answers

**B — sep curve: monotone-improving on BOTH axes, no turnover, sep=5 clears the gate.** As sep 0.5→5.0, famIoU falls 0.350→**0.264** *while capacity rises* (core50 1501→**2679**, above control's 2643; effnum→14881≈control), at zero fit cost. Unambiguously **translation, not shrinkage**: j/p ratio rises 0.36→0.39 (halves *decorrelating*), effK/h 154→191, effnum climbing — every shrink signature points away from collapse. We feared sep5.0 would trip the shortcut (weighted term ~0.4 > MSE); it just kept improving. Curve hasn't turned over → headroom likely remains, but residual famIoU (0.264, ~3× bg) is increasingly genuine compositional sharing.

**A — contrastive is LOAD-BEARING.** P7 (c=0) sprawls: core50 7700, famIoU 0.482, bg 0.260 — *worse than control*, the broadest/highest-overlap prior we've made. Clean division of labor: **contrastive = intra-task compaction (footprint size); separation = inter-task translation (position).** Without compaction the footprints are too big to keep disjoint no matter how hard separation pushes their centroids apart.

**C — locality is DEAD.** P10 (loc 0) ≡ P6 (loc 0.25) on every metric (famIoU 0.309 vs 0.311, core50 2337 vs 2368, MSE 0.207 vs 0.212). Controlled confirmation of Entry 19/20/24 — drop it permanently.

Net: Josh's "sep is the lever" bet is largely vindicated — separation does the separating, contrastive just holds footprints compact, locality is removable dead weight. First gate-clearing prior in the project.

### Decision + what's RUNNING

Graduate **P9 (c0.05 + sep5.0 + locality-off + rq512)** to a full run. **LAUNCHED 19 Jun, tmux `sep5_full`** (log `outputs/sep5_full.log`):
- Script: `job_scripts/nebius/libero_90/combined/pi05_libero_10_4_layer_film_lora2_knn36_40k_c0.05_sep5_noloc_rq512_topt1536.sh` (two-stage, skip-if-exists guard).
- = the c0.01 combined 40k script with EXACTLY: contrastive 0.01→0.05, contrastive_query_queue 128→**512**, sep 0.25→**5.0**, locality 0.25→**0** (support bands dropped), **+ routing_query_queue=512** (the c0.01 reference predated it — the critical add). New run names. Pretrain arch/schedule + entire sequential stage (tfidf_top_t **1536**, 3000 steps/task ×10, value_lr 1e-3→1e-4, 50 eval eps) unchanged.
- Runs: pretrain `libero_90_pi05_..._contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k` → sequential `libero_10_sequential_..._top_t_1536`. ETA ~44h pretrain + ~45h sequential ≈ **3.7 days**.
- top_t=1536 kept deliberately: sep5 is broad-but-separated (core50 ~2679 ≈ control), so ~1536 is the right write budget per Entry 22(d), and safer than the Entry-19 cliff (lower overlap, famIoU 0.264 vs 0.349). Watch `task9-reads-task8-updates` early; re-derive from per-batch L14 effnum if overwrite climbs.

### What to check when it lands

1. **Mid-flight (after stage 1, ~44h):** held-out audit on the 40k checkpoint — did sep5 *hold* under full LR decay (famIoU/core50 ≈ 0.264/2679, not eroded toward control)? Runnable while stage 2 trains.
2. **Held-in eval** @20k/40k vs control 76.4/81.1 (fit guardrail).
3. **The real test — sequential retention matrix:** per-task init→final, the early-task collapses (esp. the t5→t7 cliff at step 24000 in Entry 19), `memory_iou/all_modules_mean`, read-through-overwrite. Diagonal expected ~unchanged (this cycle attacked interference, not the plasticity ceiling — which remains the binding constraint on absolute performance; levers = steps/task 3000→5000, value_lr floor 1e-4→2e-4 if needed).

---

## Entry 27 - 23 Jun 26 (sep=5 graduation result: interference HALVED, performance FLAT — lever is now plasticity; prior-usefulness write protection built + launched)

### Headline

The sep=5 prior (Entry 26) **delivered its technical target and bought nothing.** In the real sequential run the prior held under full LR decay and sequential read overlap **halved (0.107 → 0.052)** — yet final avg = **34.0%** vs the Entry-19 control's **34.4%** (within rollout noise). Every prediction held *except the score*. The interference axis is now exhausted: broad/interference-limited (34%), compact/capacity-limited (~12%, Entry 21), and now decoupled-middle/separated (34%) all land ≤34%. **The benchmark is plasticity-bound.** Separation reshaped forgetting from broad-and-mild to narrow-and-catastrophic, netting zero.

### Runs

- sep5 pretrain `libero_90_..._contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k` (wandb `ozb8vddy`)
- sep5 sequential `libero_10_sequential_..._sep_5.0_..._top_t_1536` (wandb `f64nunnx`) — 10×3000, top_t=1536, 50 eval eps
- control = Entry-19 sequential `..._contrastive_0.01_sep_0.25_loc_0.25_..._top_t_1536` (34.4%)
- (First sequential launch OOM'd in grad-checkpoint recompute; the analysed run is the relaunch with `gradient_checkpointing=false`. Slot pipeline reproduces the logged `memory_iou` exactly — 0.0520 / 0.1066 — so internals read correctly. Scripts: `scripts/slots.py`/`protect.py`/`beta_sweep.py` ported the Entry-18/19 env map to libero_10 `{0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}`.)

### Pretrain: the prior did its job (separation held, capacity preserved, fit slightly weaker)

| metric @40k | sep5 | control |
|---|---|---|
| held-in eval @20k / @40k | **68.1 / 78.9** | 76.4 / 81.1 |
| routing sim L14 (10k→20k→40k) | 0.085→0.081→**0.067** | 0.119 |
| query_intra / inter | 0.91 / 0.16 | 0.82 / 0.11 |
| effnum_L14 / supp_L14 | **13211** / 6307 | 12806 / 7651 |

So separation **held and improved** through decay (answers Entry-26 check #1 without the standalone 40k audit, which was never run; sequential read IoU 0.052 confirms transfer), capacity preserved (effnum ≈ control = translation not shrinkage), held-in only mildly weaker (−2.2pp @40k; −8.3pp @20k, right at the Entry-21 0.9×-control stop-gate but it recovered). Nothing wrong at pretrain.

### Sequential retention matrix (sep5)

Train order t0..t9 = env `4,6,9,2,7,0,8,1,3,5`. Final avg 34.0 (init 41.6, peak 45.0).

| ord | env | task | init | peak | final | ret% |
|----:|----:|------|-----:|-----:|------:|-----:|
| t0 | 4 | two mugs→plates | 32 | 48 | 34 | 106 |
| t1 | 6 | mug→plate+pudding | 54 | 54 | 26 | 48 |
| t2 | 9 | mugs→microwave+close | 18 | 28 | 12 | 67 |
| t3 | 2 | stove+moka | 68 | 68 | 56 | 82 |
| t4 | 7 | **soup+cheese basket** | 18 | 18 | **0** | **0** |
| t5 | 0 | soup+sauce basket | 18 | 22 | 22 | 122 |
| t6 | 8 | both mokas→stove | 44 | 44 | 34 | 77 |
| t7 | 1 | cheese+butter basket | 18 | 22 | 10 | 56 |
| t8 | 3 | bowl→drawer+close | 78 | 78 | 78 | 100 |
| t9 | 5 | book→caddy | 68 | 68 | 68 | 100 |

### Finding 1 — interference halved, score flat (the disconnect sharpened)

| | sep5 | control |
|---|---|---|
| seq read IoU (all mods / L14) | **0.052 / 0.090** | 0.107 / 0.155 |
| pairwise channels ≥12% | **9** | 26 |
| mean read-thru-overwrite (excl last) | **37.3%** | 55.4% |
| L14 effnum / core50 (capacity) | 5206 / 2526 | 7020 / 3325 |
| mean diagonal (init) / peak | 41.6 / 45.0 | 39.8 / 44.2 |
| **final** | **34.0** | **34.4** |

Interference fell hard and broadly (the env3 "bowl→drawer" task that overwrote *eight* others at 20–44% in control is defanged → mid-run seen-success was actually *higher*, e.g. @24k 27.3 vs 22.5). It just doesn't survive to the final number.

### Finding 2 — the cliff MOVED, it didn't close: forgetting reshaped spread → concentrated (the object-sharing autopsy)

Control's catastrophic collapse (env0, the step-24000 cliff, 28→4) is **gone** — env0 rescued (122% ret). A **new** collapse opened: **env7 (soup+cheese) 18→0 at step 18000.** The overwrite matrix tracks **object-sharing exactly**:

- env7 = soup **+ cheese** ; env0 = soup **+ sauce** ; env1 = cheese **+ butter** ; env0∩env1 = ∅ (basket frame only)

| channel (4-layer) | shared object | sep5 | control |
|---|---|---|---|
| env7 ← env0 | soup | **53.4%** | 61.9% |
| env7 ← env1 | cheese | **54.5%** | 58.0% |
| env0 ← env1 | *none* | **23.7%** | 45.5% |

Separation cleanly fixed the pair with **no genuine sharing** (env0←env1 45→24 → env0 survives) and **could not touch** the pairs with a shared object (env7←{env0,env1} ~53%). env7 is the **hub** of the family — soup∩cheese — so it can't be separated from both neighbours and eats 53%+54% = **81.5% read-through → 0**. Confirms Entry 24/26's "residual ≈0.26 is genuine compositional sharing," caught in a rollout. (This is the realworld-Entry-3 *concentration-vs-spread* pattern reproduced in sim: both runs lose one task catastrophically; separation just relocates which one + cleans the broad bleed, so the average is unchanged.)

### Finding 3 — the diagonal (plasticity) is the binding constraint, untouched by routing

Per-task block-min MSE is **identical** sep5 vs control: dual-cycle tasks plateau at 0.20–0.23, single-cycle at 0.08–0.17, set by task structure not prior. Six of ten tasks are two-full-pick-place compositions (OOD for the single-step libero_90 pretrain) that rank-2 LoRA on a frozen backbone won't fit past 18–54% in 3000 steps. Mean peak ≈ 45% caps everything; "even perfect retention caps this run at ~40%" (Entry 19/26). Reducing interference cannot lift a fit ceiling.

### Discussion — can IDF protect env7? Mechanism autopsy + the corrected rule (with Josh)

Q (Josh): isn't this what IDF is for — env7's writes should raise document-frequency and stop later tasks writing there? Traced the code (`lerobot_sequential_train.py`): online DF is `df_vec[used]+=1` (binary per batch, `used`=retrieved/READ indices), pooled over all tasks; `idf=log((B+1)/(DF+1))^e`; mask = top-t by `tf*idf`. **Structurally the wrong shape**, four reasons:
1. **Pooled & task-anonymous** — knows "popular," never "env7 relied on it."
2. **Read-frequency, not usefulness** — every task reads ~140K of 147K slots (coupon-collector), so "read by a prior" is near-universal → useless gate (measured: binary "any-prior-read" protection blocks 74–90% of every writer's demand).
3. **TF-overridable** — env0's soup-core TF ≈86× median vs IDF discount ≈3× (log crushes a 60:1 DF ratio) → the writer buys back the shared core. (Entry 19 already measured: e=1 keeps 100% of harmful writes, e=4 keeps 60% while blocking legit writes.)
4. **Self-pollution** — DF accumulates within the current task's own block → env0's reads inflate the DF of env0's private slots → the mask drifts against itself.

The behaviour Josh actually wants: *"if a slot was useful to any prior task, don't update it; probability of update decreases the more useful it was."* This is task-identity-aware + importance-weighted + graded — i.e. the realworld-Entry-3 "collision-aware write protection," deferred there, now reopened. It is EWC-flavoured (protect-important-params) but as a sparse write-gate on read-usefulness, not a Fisher loss.

**Protectability frontier (graded, per prior task's read core, L14):** binary is hopeless, but the *graded* form has real headroom, and — the key result — **the separated prior makes it ~1.5× cheaper**:

| protect env7 core | saved (env7 dmg) | cost (env0 blocked) | sep5 saved/cost | control |
|---|---|---|---|---|
| top-25% | 21.6% | 10.6% | **2.03×** | 1.23× |
| top-50% | 36.6% | 23.8% | **1.54×** | 0.98× (zero-sum) |

Damage is steep at the core (protect where the prior cared most), cost is gradual (the writer's demand is spread) — exactly the structure the "decrease with usefulness" rule exploits. Separation moved env0's demand partly off env7's core, so it's no longer the zero-sum Entry 19 correctly found *on the broad prior*. **Separation and protection compound** — which is why this lever is worth it *now*.

### Offline β-sweep (`scripts/beta_sweep.py`, first-order, static footprints)

Gate `π(s)=(1−u(s))^β`, `u(s)=max over prior tasks of peak-normalized read profile`; β=0 reproduces the measured read-through (37.3% sep5 / 55.4% ctrl — validated). Net = mean prior read-mass saved − mean writer demand blocked:

| β (sep5) | mean saving | mean cost | net | env7 RTO | env0 cost | env1 cost |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 81.5% | 0 | 0 |
| 4 | 12.4% | 6.9% | +5.5% | 52.7% | 13.2% | 16.4% |
| **8** | 17.2% | 11.0% | **+6.3%** | **41.8%** | 20.4% | 24.7% |
| 16 | 22.2% | 16.5% | +5.6% | 30.9% | 29.6% | 34.7% |

Net-positive at every β>0, peaks ~β=8 (env7 read-through roughly halved). Cost concentrates on env0/env1 (themselves basket tasks near the floor). To hit the same env7 outcome the control pays its writers ~36% vs sep5 ~20–25% (compounding, confirmed). **Caveats:** mass ≠ rollout success (likely *favourable* — saving sits on env7 at 0%, cost on env0/env1 at 22/10%, concave returns); static footprints (no closed-loop re-routing); these are why the real run is the arbiter.

### Mechanism implemented (opt-in, default OFF — legacy byte-identical)

`lerobot_sequential_train.py`:
- **Store** `_protect_usefulness_by_module[json_key]` = `u(s)`; `_accumulate_protect_counts_batch` (raw read counts/batch, all ranks, mirrors online-IDF) + `_finalize_protect_usefulness` at each task boundary (`u ← max(u, counts/counts.max())`, then reset). Folded *after* a task finishes so a task never protects against itself.
- **Gate** in `_compute_tfidf_top_indices_for_batch`: `tfidf *= (1−u).clamp(min=0)**β` before top-t → protected slots fall out, budget reallocates to the task's private reads.
- **CLI** `--protect_prior_slots` (bool, default `False`) + `--protect_beta` (float, default 4); threaded through `_update_policy_with_tfidf`. When off → `None` passed → old branch. Smoke-tested (`scripts/smoke.py`): peak-norm + max-agg + reset, gate reselection (vetoed slot drops, next-best pulled in), and **β=0 / store=None reproduce the legacy top-t exactly**.
- NB: implementation β acts on the top-t **ranking** (reselection), so its scale differs from the offline soft-suppression model — treat β=4 as "moderate."

### Launched (23 Jun, tmux `protect_b4`, wandb `11u7mdmj`)

`job_scripts/nebius/libero_90/sequential/pi05_libero_10_seq_sep5_prior_protect_beta4_topt1536.sh`: reuses the sep5 40k prior (no new pretrain), `--protect_prior_slots=true --protect_beta=4`, **eval 20 eps** (vs 50, faster). Otherwise byte-identical to the sep5 sequential. ETA **~17–18h** (eval-bound). Confirmed live: config dump shows `protect_prior_slots=True, protect_beta=4.0`; per-task boundary logs "Updated prior-usefulness protection store after task N".
- **Watch:** env7 (t4) — baseline 18→0; predicted partial rescue (RTO ~81→~53). env0/env1 (t5/t7) fit cost = the bill. `memory_iou` should drop below 0.052. Diagonal/init ~unchanged (interference lever, not the ceiling).
- **Decision rule:** env7 materially >0 at acceptable env0/env1 cost → β sweep {2,8}; env7 unmoved → β=8; env0/env1 crater → β=2. If even β=8 can't hold env7 and keep its writers, env7 is confirmed irreducible under write-masking → only rank/co-host (realworld Entry 3) or scene-vs-language query reweighting (Entry 24 #2) remain.

### Next steps

1. **Analyse the β=4 run on landing** — retention matrix + slot autopsy + read-IoU vs this baseline; does mass-saving convert to env7 success.
2. **Plasticity track (the actual binding constraint, pinned but primary for the *average*):** steps/task 3000→5000 (MSE still falling at block ends), value_lr floor 1e-4→2e-4, then lora_rank 2→4 / longer+stronger pretrain (held-in 78.9<81.1, 40k still undertrains libero_90). Protection protects the fit that exists; only this lifts the ~45% peak — and *then* retention gains convert to points.
3. **env7-specific (if protection insufficient):** scene-weighted query fusion so basket tasks separate by table contents not near-identical language (Entry 24 #2); higher rank for co-hosting the soup/cheese primitive.

### What we are not doing

- No more pretrain-side separation sweeps — the decoupled prior exists and its sequential payoff is flat; the interference axis is mapped end-to-end.
- No `idf_exponent`/weighted-DF (measured no-ops); no hard task-boundary veto (Entry 19 zero-sum); no router training / per-task params (off the table).
- Not raising `top_t` for env7 — the writer needs the shared core it damages (Entry 19 zero-sum); write-location masking can't fix a write-magnitude/co-host problem, only the *graded soft* gate trades it.

---
