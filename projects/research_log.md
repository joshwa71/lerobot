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
