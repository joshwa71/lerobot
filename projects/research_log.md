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

## Entry 28 - 24 Jun 26 (protection β=4 result: +6.5pp via the INCIDENTAL half, env7 irreducible; routing autopsy FLIPS — FiLM language is near-inert, routing is scene-driven)

### Part A — prior-usefulness protection (β=4) result

The Entry-27 mechanism ran (`..._top_t_1536_protect_beta4`, reused the sep5 prior, 20 eval eps, ~17h). **It works — on the protectable half.** Headline (caveat: 20 eval eps vs the 50-ep baselines, so ±3–4pp noise; slot metrics are eval-independent and corroborate):

| metric | control(50) | sep5(50) | **protect β4(20)** |
|---|---|---|---|
| final avg | 34.4 | 34.0 | **40.5** |
| within-run forgetting (init→final) | −5.4 | −7.6 | **−3.0** |
| mean read-through-overwrite | 55.4% | 37.3% | **29.8%** |
| pairwise channels ≥12% | 26 | 9 | **6** |
| read IoU | 0.107 | 0.052 | 0.051 |
| L14 effnum / core50 | 7020/3325 | 5206/2526 | 5257/2534 |

Forgetting roughly **halved** (−7.6→−3.0), and the gains are mechanistically attributable — the tasks whose read-through dropped are the ones that improved: env6 (37→28% / rollout 26→55), env8 (22→17% / 34→50), env1 (15→13% / 10→30), env9 (56→45% / 12→20). Read IoU unchanged because reads are frozen-router; protection is write-side (steers later writes off prior cores). Capacity untouched.

**env7 still collapses to 0** (the genuine hub). Its read-through dropped 81.5→**71.2%** and its killer channels fell (env7←env0 53→**40%**, env7←env1 54→**39%**) — the gate *did* steer env0/env1 off env7's core — but 71% + gate≈0.99 still = collapse. Its trajectory softened from a cliff (baseline 18→0 at the next task) to a bleed (25→10→5→5→0), but it ends at 0.

**Why env7 resists, and why that's expected — two mechanistic findings:**
1. **β=4 under-protected it specifically.** Offline predicted env7→52.7% at β=4; actual 71.2% ⇒ *implementation* β=4 ≈ *offline* β≈1 for env7. The top-t reselection is weaker than the soft-suppression sim.
2. **Top-t reselection is self-limiting on genuine contention.** env0/env1's soup/cheese-core slots have TF ~86× median; even ×(1−u)^4 they stay in the top-1536 → survive the gate. The slots it *can't* exclude are the high-TF shared ones = exactly the genuine-contention slots write-masking provably can't protect anyway (Entry 19 zero-sum). So the mechanism self-selects the **incidental** half (cheap, low-TF overlaps → the +6.5pp) and gracefully declines the **genuine** half (env7). This is the cleanest confirmation of the Entry-27 incidental-vs-genuine split.

**Verdict:** β=4 protection is a keeper — fold into the recipe (free, opt-in, +6.5pp via forgetting). Higher β won't rescue env7 (self-limiting) and would start costing env0/env1 — not worth it. env7 is now firmly the genuine-contention residual.

### Part B — env7 is "a routing issue": query-decomposition autopsy (Josh) — and it FLIPS the hypothesis

Hypothesis (Josh + me): the frozen FiLM-on-language router drops the lookalike basket tasks into a shared basin; the fix is to **up-weight `proj(x)` (scene) over language** in the query `q = proj(x)·(1+γ(lang)) + β(lang)`. First clarified (code): this is a *fusion* lever, not a routing-*loss* lever — the loss is downstream of `q` and can only separate what `q` already encodes (which is why sep=5 floored the basket famIoU at 0.26).

Then probed the actual query map from the **checkpoint alone** (no forward; `scripts/query_probe.py`): embed the 10 instructions (all-mpnet-base-v2), push through each layer's `film_mlp`, inspect γ/β. **The premise is wrong — the FiLM language pathway is near-inert:**
- **`γ ≈ 0`**: `‖1+γ‖ ≈ 45.3 ≈ √2048` at every layer → multiplicative modulation ≈ identity.
- **`β` is near-task-agnostic** despite distinguishable instructions:

| | instruction cosine (raw mpnet) | β cosine (post-`film_mlp`) |
|---|---|---|
| basket | 0.76 (env7~env0 0.86, env0~env1 0.61) | **0.98** |
| background | 0.43 | **0.945** |

`film_mlp` **compresses the instruction signal away** (cos 0.43–0.86 → 0.945–0.99; bias-dominated output). So the one signal that uniquely names the basket tasks ("which two objects") is present at the input and discarded. Routing is carried by **`proj(x)` — the scene**. (Recontextualizes Entry-18's "frozen language-router" → it's a frozen *scene* router; Entry-24 started softening this, now quantified to near-fully-scene. The whole sep/contrastive program has been separating tasks in *scene* space — works when scenes differ, floors when they don't.)

**Implication:** "up-weight scene over language" is **moot/backwards** — scene already dominates, language already ≈0 for discrimination. env7 collapses because the **basket scenes are genuinely similar** (same kitchen table / overlapping objects / basket) and the model has no working channel for the instruction that would separate them. The likely irreducible case.

**Revised options for env7 (all harder than a reweight):**
1. **Recover the language signal** (opposite direction): stop `film_mlp` collapsing to its bias (anti-collapse reg / align β with mpnet instruction *differences*) so "soup+cheese" routes differently from "soup+sauce". Caveats: re-opens held-out-generalization risk (Entry 18); basket instructions are themselves only cos 0.86 → bounded; new pretrain.
2. **Representation / co-host (rank)** for the shared region (realworld Entry 3 cautions rank doesn't allocate under a frozen router).
3. **Accept env7**, bank the protection win on the rest.

### Part C — forward probe PREPARED (pending; run next)

The checkpoint-only probe is decisive that language is near-inert, but the last open question — *is the basket residual driven by scene-similarity or by the small β-residual?* — needs `proj(x)` magnitudes. Prepared (NOT yet run):
- `scratchpad/query_forward.py`: monkeypatches `QueryMLPLite.forward`, runs ~25 batches/task on basket (task_idx 4/5/7) + controls (2/6) through the frozen sep5 checkpoint, recomputes routing under **full = proj·(1+γ)+β / scene = proj·(1+γ) / lang = β** and reports `‖scene‖` vs `‖β‖`.
- Trainer hook to add (env-var-guarded, after `task_index_to_name` ~line 1622): `if os.environ.get("QUERY_PROBE"): run_query_probe(...); sys.exit(0)` — reuses the validated policy/dataset/preprocessor setup, forward-only, exits before training.
- **Decisive test:** basket `IoU(scene)` vs `IoU(full)`. `scene << full` → β drives the collision → option 1 has teeth. `scene ≈ full` → pure scene similarity → only options 2/3.

**RESULT (ran via standalone `scratchpad/query_forward_standalone.py` — reuses `parser.wrap` + factories, NO trainer changes; 25 batches/task on basket 4/5/7 + controls 2/6):**
- **Scene dominates β by 17–21×**: `‖proj(x)·(1+γ)‖` ≈ 16 / 18 / 18 / 20 at L8/10/12/14 vs `‖β‖` ≈ 0.98. The language bias is a ~5% additive perturbation on the query.
- **Stripping β changes basket routing by ≈0%** (L14 basket weighted IoU): full **0.215** vs scene (β-stripped) **0.216** — e7~e0 0.221 vs 0.224, e7~e1 0.282 vs 0.283, e0~e1 0.142 vs 0.143. Language-only (q=β) would be 0.545, but β is 20× too small to bend the routing. (full-query basket 0.215 / background 0.058 reproduces the sequential JSON read IoU — pipeline validated.)
- **VERDICT: env7 is pure scene-similarity, NOT routing-fixable by reweighting.** Scene already dominates 20×; deleting language entirely leaves the basket collision unchanged. The three basket tasks have near-identical *initial scenes* (same kitchen, overlapping objects, a basket); the sole discriminator is the instruction, which the model routes ≈0% on. **Option 1 (down-weight language) is DEAD.** The only routing-based lever left is the *opposite and amplified* — recover AND scale the language pathway ~20× (anti-collapse + magnitude) so the instruction can steer routing — a large architecture change, with held-out-generalization risk (Entry 18) and bounded by instruction similarity (mpnet cos 0.86). Pragmatic read: env7 is the genuine *same-scene/different-instruction* residual (1 of 10); bank the β=4 protection win on the other 9 and pivot to the plasticity ceiling for the average; revisit env7 only if a language-amplification or co-host idea is worth a pretrain.

### Next steps
1. Run the forward probe (Part C) → decide whether env7 is routing-fixable at all.
2. Fold β=4 protection into the recipe regardless (it's a free retention win on the incidental channels).
3. Plasticity remains the binding constraint on the *average* (diagonal ~43; dual-cycle tasks low) — steps/task, value_lr, rank, longer pretrain. Independent of the env7 question.

---

## Entry 29 - 24 Jun 26 (LAUNCHED: autonomous protection+plasticity batch on the sep5 prior — 4 sequential runs, ~3 days)

Josh away a few days; lined up 4 **sequential-only** runs (all reuse the EXISTING sep5 40k prior — no new pretrain), single-knob deltas from the standing baseline **β=4 protection (Entry 28, 40.5% @20ep)**. Chained in tmux `plast_batch`; runner `job_scripts/nebius/libero_90/sequential/run_protect_plasticity_batch.sh`; runner log `outputs/protect_plasticity_batch.log`; per-run logs `outputs/batch_logs/<run>.log`; wandb project `vla-memory`. All 20-eval-eps for apples-to-apples with the β4 baseline. Robust runner (one failure doesn't abort; skip-if-final-ckpt-exists).

Order C → B → D → A (front-load the binding-constraint/plasticity tests; β8 last):

| # | run suffix (`…_top_t_1536_<suffix>`) | delta vs β4 | tests | prediction |
|---|---|---|---|---|
| C | `protect_beta4_steps5k` | steps 3k→5k | plasticity (safe) | diagonal/init ↑ (MSE still falling at 3k block-ends) |
| B | `protect_beta4_lr2x` | value_lr 1e-3/1e-4→2e-3/2e-4 | plasticity (LR) | diagonal ↑; **watch 2e-3 peak instability** (Entry 20: t8 late-block MSE; grad-clip 1.0 is the guard — if grad_norm/MSE blow up it's diverging, fallback floor-only 2e-4) |
| D | `protect_beta4_lr2x_steps5k` | both | plasticity ceiling / additivity | most likely new best; protection(write-location) ⟂ LR(write-magnitude) → should compose |
| A | `protect_beta8` | β 4→8 | protection curve | env7 ~unchanged (self-limiting); env0/env1 start paying; incidental channels protected harder |

**Standing conclusions feeding this batch (Entries 27–28):**
- β=4 protection is a confirmed keeper: forgetting halved (init→final −7.6→−3.0), mean read-through 37→30%, +6.5pp — entirely on the **incidental** channels; env7 (genuine hub) stays 0.
- **env7 is NOT routing-fixable** (forward probe, Entry 28 Part C): scene dominates the query 17–21×; stripping β changes basket routing ≈0% (full 0.215 vs scene 0.216 @L14). It's the genuine same-scene/different-instruction residual. Routing-side lever would be the *opposite+amplified* (recover & scale language ~20×) — deferred, risky.
- The **average is plasticity-bound** (diagonal ~43, dual-cycle tasks low) — hence this batch targets plasticity. The big uncovered lever is **rank-4** (per-slot expressivity) — needs a fresh 44h pretrain (blocks the GPU), so it's the first thing to queue on return, + a clean **50-ep re-eval** of whatever wins this batch (de-noise the 20-ep numbers).

**Analysis recipe (persisted: `scripts/vla_analysis/`):** env map `{0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}`; train order = task_index 0..9; basket family = task_idx 4/5/7 (env7/env0/env1, env7=soup+cheese hub).
- Retention matrix: `retention3.py` (add the 4 new run dirs to its `RUNS`). Baselines: control 34.4 / sep5 34.0 / β4 40.5 (final avg); mean init (diagonal) control 39.8 / sep5 41.6 / β4 43.5 — **for the plasticity runs the key read is whether mean init/peak rises above ~45**.
- Slot usage: `slots.py <run>` (add run to its `runs` dict) — capacity (L14 effnum/core50), read-through-overwrite (control 55/sep5 37/β4 30%; env7 86/81/71%), pairwise overwrite, read-IoU validation (must match logged `memory_iou`).
- wandb scalars: `wb.py` / `parse_wandb.py` — held-in/seen success, mse_loss, gate, query sims.
- Routing autopsy (done, reproducible): `query_probe.py` (checkpoint-only) + `query_forward_standalone.py` (forward; reuses `parser.wrap`+factories, NO trainer changes — run via conda env, `QUERY_PROBE_NB` batches/task).

**Mechanism in code (shipped, opt-in, default off):** `lerobot_sequential_train.py` — `--protect_prior_slots` / `--protect_beta`; store `_protect_usefulness_by_module` (u(s)=max over prior tasks of peak-normalized read profile, folded at task boundaries), gate `tfidf *= (1-u)^β` before top-t. β=0/off ⇒ legacy byte-identical.

**When results land:** retention matrix per run vs β4 baseline (diagonal first — did plasticity lift the ceiling?), B/D LR-stability check, A's env0/env1-vs-env7 tradeoff. Pick the winner → 50-ep re-eval → then rank-4 pretrain.

---

## Entry 30 - 29 Jun 26 (plasticity batch results: the diagonal MOVED for the first time — steps5k is the new best (44.5); LR-2× backfires, β8 over-protects; non-memory baselines LAUNCHED to locate the standard-finetuning ceiling)

### Headline

The Entry-29 batch (C/B/D/A, all reuse the sep5 prior + β-protection, 20 eval eps, all completed clean — no NaN/divergence/OOM) **broke the plasticity wall for the first time in the project.** Entry 27 found the per-task fit floor (block-min MSE) *identical* sep5 vs control — routing never touched the diagonal. Here, `steps=5000` and `lr=2e-3` BOTH lower it (−20% / −11%; both together −28%, the lowest in the project), and peak rollout climbed 45→**53-55**. **Run C (β4 + steps5k) is the new best at 44.5%** (+4pp over β4, +10pp over control/sep5). But the levers aren't free — they broaden the late-layer read footprint and re-couple interference — so LR-2× backfires on retention and the two don't compose at the rollout level. β=4 is confirmed the protection optimum; β=8 over-protects.

### Scoreboard (all protect-family 20-ep, mutually comparable + vs protectB4; control/sep5 50-ep; slot/MSE cols eval-independent)

| run | lever vs β4 | **final** | init(diag) | peak | forget | **MSE-floor** | L14 effnum | core50 | read-IoU | RTO | ch≥12% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control(50) | — | 34.4 | 39.8 | 44.2 | −5.4 | 0.149 | 7020 | 3325 | 0.107 | 55.4% | 26 |
| sep5(50) | — | 34.0 | 41.6 | 45.0 | −7.6 | 0.146 | 5206 | 2526 | 0.052 | 37.3% | 9 |
| **protectB4** | (baseline) | 40.5 | 43.5 | 49.5 | −3.0 | 0.150 | 5257 | 2534 | 0.051 | 29.8% | 6 |
| **C steps5k** | +steps | **44.5** | 43.5 | 53.5 | **+1.0** | 0.120 | 6630 | 3265 | 0.057 | 33.2% | 8 |
| **B lr2x** | +LR(2e-3) | 38.0 | **47.5** | **55.0** | **−9.5** | 0.133 | 8689 | 4443 | 0.071 | 33.2% | 7 |
| **D lr2x5k** | both | 42.5 | 46.5 | 53.5 | −4.0 | **0.108** | 11789 | 6032 | 0.085 | 37.9% | 14 |
| **A beta8** | β8 | 38.5 | 39.0 | 46.0 | −0.5 | 0.152 | 5143 | 2495 | 0.051 | **25.8%** | **3** |

Score order: **C(44.5) > D(42.5) > β4(40.5) > A(38.5) ≈ B(38.0) ≫ ctrl/sep5(34).** Slot read-IoU reproduces the logged `memory_iou` exactly for all 7 runs (pipeline validated). No instability: grad-norm max 0.037–0.045 everywhere (B/D the *lowest*, ~0.037, 25× below the clip) — the Entry-29 "watch 2e-3" worry was void; B/D's behaviour is a clean optimization effect, not divergence. LR confirmed (B/D peak ≈2e-3, others ≈1e-3).

### Finding 1 — the diagonal finally moved (block-min MSE), and it was partly optimization-limited

Per-task block-min `train/mse_loss`, mean over 10 tasks: β4 0.150 → **C 0.120 (−20%)**, B 0.133 (−11%), **D 0.108 (−28%)**, A 0.152 (≈β4). Across-the-board, including the hard dual-cycle tasks E27 said rank-2 LoRA "won't fit in 3000 steps" (soup+cheese 0.206→0.134, cheese+butter 0.210→0.162, mug+micro 0.238→0.181). **So the diagonal was partly optimization budget, not a pure rank-2 architecture wall** — more steps and higher LR both lower the floor, and they compose at the fit level (D lowest). A confirms the negative control: β is a pure write-protection knob, leaves fit untouched.

### Finding 2 — the levers broaden the read footprint → re-couple capacity↔interference

L14 effnum: β4 5257 → C 6630 → B 8689 → **D 11789** (broader than *control's* 7020). Memory sits at [8,10,12,14], so value updates at L8/10/12 perturb the residual stream → shift the **frozen** router's `proj(x)` query at L14 → route to a broader slot set. Training values harder broadens the late-layer footprint, which lifts capacity (fit) AND interference together: read-IoU 0.051→0.085, RTO 29.8→37.9%, channels 6→**14** (D). **Plasticity (broaden) and protection (compact-the-writes) push OPPOSITE directions on interference — they are NOT orthogonal** (the E29 "protection ⟂ LR → should compose" guess is disconfirmed at rollout: they compose on MSE, anti-compose on interference).

### Finding 3 — steps ≫ LR (the basin-depth mechanism, E18)

- **C `steps=5000` (WIN):** deepens every basin (−20% MSE) with only marginal broadening (effnum +26%, read-IoU +0.006). **Retention *improves* (+1.0, the only positive)** despite slightly higher exposure — exactly E18's *final ≈ f(basin depth, exposure)*: C deepens basins faster than it raises exposure, so the deeper basins absorb the overwrite. (Secondary: the 5k-step protection store accumulates more read-batches → sharper `u(s)` → better-targeted protection.) Fit gain shows up as **peak+retention**, not a higher fresh diagonal (init 43.5 = β4).
- **B `lr=2e-3` (BACKFIRES):** fits the *fresh* diagonal highest (init 47.5, peak 55.0) but the big-magnitude writes are the overwriters → **worst retention −9.5** → nets below β4. Clean plasticity/stability tradeoff, not instability.
- **D both (DOESN'T COMPOSE):** best fit floor in the project (0.108) + broadest footprint (11789) → most interference of the family (read-IoU 0.085, 14 channels, RTO 37.9%). The 5k steps rescue *some* of LR's retention damage (−4.0 vs B's −9.5, deeper basins) but it lands **below C**. Compose on fit, anti-compose on interference → C-alone wins.

### Finding 4 — β=4 is the protection optimum (β8 over-protects, as predicted E28)

A (β8) is the strongest interference suppressor on record — RTO 25.8%, only **3** channels≥12%, **env7 RTO 86→60.5%** (best ever; killers e7←e0/e7←e1 down to 40.6/35.4%), retention −0.5. **But** the heavier write-veto starves the basket writers' own fit (env0 craters, init 39.0 lowest) → excellent retention of a worse fit → 38.5, below β4. β=4 stays the sweet spot.

### Finding 5 — env7 still irreducible

env7 ends ~0–5 in every run. B's 2× LR gave it its best-ever *fresh* fit (init 30) and A's β8 its best-ever *protection* (RTO 60.5%) — neither saves it (60% read-through × gate≈0.98 still overwrites, and the protection that helps also starves its own writes). The genuine same-scene contention the E28 forward probe nailed. Parked.

### Caveats
- All 20-ep → ±3–4pp/cell. C's env4 "two mugs" reads 40→5 final while β4 *and* D held env4=40 → almost certainly a noisy single eval (if env4≈40, C's true mean ≈48). **The winner needs a clean 50-ep re-eval before locking in.**

### Decision — run the missing non-memory baselines (Josh, 29 Jun)

The whole memory program is judged against an *implicit* ceiling we never measured: **plain pi05 multi-task finetuning, no memory layers.** With the diagonal now shown movable and C at 44.5/peak 53-55, the live question is whether ~45–55% is a *base-model* ceiling on this OOD dual-cycle suite or a memory-specific limit. Two baselines (base pi05, no memory, **same base train args** — same scheduler/warmup, pi05's default base LR, bs32, grad-ckpt, bf16, empty_cameras, rename_map, normalization):
1. **`libero_90_and_long_pi05_base_50k`** — 50k steps on libero_90 **+** libero_10(Long) merged (the natural multi-task ceiling *with* the pretrain data).
2. **`libero_10_pi05_base_50k`** — 50k steps on just libero_10(Long) (the 10-task-only finetune ceiling).
- Both: `lerobot-train` (NOT sequential — standard joint finetune), eval **libero_10 @ 50 eps/task** at the end only (`save_freq=eval_freq=50000`, code stable → no storage blow-up / eval drag). `scheduler_decay_steps=50000` (honor the schedule, E20 gotcha). Base = pinned `pi05_base` snapshot `9e55186`.
- **Dataset built:** `outputs/libero_90_and_long` via `merge_datasets.py` (libero_90 3959 eps + the 10 Long tasks 379 eps; the libero_10 dataset's task_index 0–9 are the only ones with episodes — 10–39 are empty vocab). NB: merge needed a lenient features check (libero_10 carries a redundant per-feature `fps` key; bypassed via monkeypatch, utility unchanged; libero_90 placed first so the target inherits its clean schema).
- Scripts: `job_scripts/nebius/baselines/`. **What this tells us:** if base-pi05 on 90+10 reaches ~50–55% per-task, the memory method is roughly at the joint-finetune ceiling and the remaining gap is the OOD dual-cycle structure (→ rank/longer-pretrain, not continual-learning machinery); if base-pi05 is well *above* us, the memory constraints (frozen backbone + rank-2 values) are costing real performance and that's the thing to attack. The libero_10-only run isolates how much the 90-task pretrain data actually helps the Long suite.

### Cleanup
Freed **640G**: deleted the 9 intermediate per-task checkpoints from each of the 4 batch runs (kept each run's final + `last` symlink + the small eval/`memory_by_task` JSONs that the analysis reads). Killed the now-redundant `training_state` reaper (batch done; ts already at 0). Disk 1.1T free (55%).

### Next (after baselines land)
1. Read the two baselines' per-task libero_10 success → place our 44.5 / peak-55 against the standard-finetune ceiling.
2. **50-ep re-eval of C** (`protect_beta4_steps5k`) to de-noise 44.5 and resolve the env4 cell. *(Winner final checkpoint retained for this.)*
3. Then the orthogonal plasticity swing: `lora_rank=4` (fresh 44h pretrain — now best-justified, attacks the per-slot expressivity the steps/LR levers can't), and/or push `steps→7000` (steps is the clean Pareto lever; MSE was still the cleanest mover).

---

## Entry 31 - 1 Jul 26 (non-memory baseline: joint-finetune ceiling = 72.6%; the memory gap is ~28pp, concentrated on the forgetting/dual-cycle tasks)

Plain pi05, **NO memory**, same base train args (bf16, bs32, grad-ckpt, warmup 4k / decay 50k, pi05 default LR), 50k steps, eval libero_10 @ 50 eps/task. Ran **B1 = libero_90 + libero_10 joint finetune**; killed B2 (libero_10-only) — B1 answered the question. Run artifacts deleted (results here; wandb `zv5k7a6m`).

**B1 → 72.6% on libero_10** (500 eps). Per-task final % vs our best memory runs (by env id; C = `protect_beta4_steps5k`, the Entry-30 winner):

| env | task | BASE (90+10) | C steps5k | β4 | gap base−C |
|--:|--|--:|--:|--:|--:|
| 0 | soup+sauce | 74 | 40 | 15 | +34 |
| 1 | cheese+butter | 66 | 50 | 30 | +16 |
| 2 | stove+moka | 96 | 70 | 55 | +26 |
| 3 | bowl+drawer | 86 | 90 | 75 | −4 |
| 4 | two mugs | 52 | 5 | 40 | +47 |
| 5 | book | 68 | 70 | 65 | −2 |
| 6 | mug+pud | 86 | 55 | 55 | +31 |
| 7 | soup+cheese | 80 | 0 | 0 | **+80** |
| 8 | both mokas | 58 | 40 | 50 | +18 |
| 9 | mug+micro | 60 | 25 | 20 | +35 |
| | **overall** | **72.6** | **44.5** | **40.5** | **+28.1** |

- Ceiling is **72.6%, +28pp over our best** → we are NOT near the standard multi-task finetune ceiling; the frozen-backbone + rank-2-memory continual setup costs real performance (disconfirms the "maybe we're already at the ceiling" hypothesis).
- **env7 (soup+cheese) is not irreducible task difficulty** — base solves it at **80%** vs 0% under memory. Its collapse is pure continual-learning forgetting (shared-slot overwrite); joint finetune has no forgetting so the "genuine same-scene contention" only bites under sequential adaptation.
- The gap concentrates on the forgetting/dual-cycle tasks (env7 +80, two-mugs +47, mug+micro +35, soup+sauce +34); the easy single-cycle tasks are already at ceiling (bowl+drawer −4, book −2). → case for the plasticity/capacity + forgetting track (rank-4, longer pretrain), not more routing separation.
- Caveat: 72.6% is a *joint* finetune (all tasks at once, no continual constraint) — an upper bound for the setup, not a target a frozen-backbone continual method can fully reach. The value is the **size + location** of the headroom.

### Update (1 Jul) — layer-wise LoRA rank probes launched (capacity where it matters, without rank-4-on-all-4)

That 28 pp is mostly forgetting + plasticity, and rank is the one untried per-slot-capacity lever. Rank-4 on all four layers is 4.8 B values → OOM (~148 GB > H200; confirmed by the VRAM ladder below), and rank-3 has matmul issues — so the question is *where* to spend a smaller increase. Implemented **per-layer rank**: `--policy.memory_layer.layer_ranks=[...]`, matched to `layers` by order, asserts length; empty ⇒ scalar `lora_rank` (legacy byte-identical). Smoke-tested (per-layer shapes, length assert, backward compat, mixed-rank forward). Two 10k probes, **C's noloc-sep5 recipe verbatim except `layer_ranks`** (compressed 10k schedule = every prior probe → the existing rank-2 sep5/P9 audit is the `[2,2,2,2]` baseline):

- **P1 `[2,2,2,4]` on `[8,10,12,14]`** — keep all four, boost the action-proximal L14 (+25%, 3.0 B values). *Invest in the highest-value / most-output-proximal layer.*
- **P2 `[4,4,4]` on `[8,10,12]`** — drop L14, rank-4 on the rest (+50%, 3.6 B). *Josh's original: shed the messy high-interference L14, beef up the cleaner earlier layers.*

Decisive read at the held-out audit: does **P2's L12 inherit-and-worsen** the family IoU / core50 — the high-trust/high-forgetting role is *positional* (last memory layer), so dropping L14 likely migrates it to L12, now at rank 4 = more destructive overwrites (Entry 4) — or does P1's proximal boost sit better with routing intact? Plus the 1-task plasticity probe for fit. Caveat: rank is a **fit** lever, not a forgetting one under the frozen router (realworld Entry 3) → expect diagonal gains; the average is plasticity-bound anyway.

Status: launched (tmux `layerrank_probes`, P1→P2, ~10 h each). P1 confirmed stepping — attach applied `L8=r2/L10=r2/L12=r2/L14=r4`, 119 GB (fits). **VRAM ladder:** P1 3.0 B = 119 GB, P2 3.6 B ≈ 129 GB (fits), all-r4 4.8 B ≈ 148 GB (OOM) — why 4-on-4 is off the table. Code: `memory_config.py` (`layer_ranks`), `memory_lite.py` (threaded `lora_rank_override` through attach → `MLPPlusMemory` → `HashingMemoryLite`). Script: `probes/run_layerrank_probes.sh`.

---

## Entry 32 - 2 Jul 26 (layer-rank probe verdict: drop-L14 REJECTED — the last-layer role is positional and migrates+amplifies; keep-4-boost-L14 clean. Rank mental model: capacity AND interference both scale with rank → [2,2,4,4] probes with a contrastive re-look LAUNCHED)

### Probe results (10k + held-out audit; anchors reproduce exactly — control 2643/0.349/0.127, P9 2679/0.264/0.087)

| run | last layer | effnum | core50 | famIoU | bgIoU | worst pair |
|---|---|---:|---:|---:|---:|---|
| control@40k | L14 | 5928 | 2643 | 0.349 | 0.127 | e7~e0 0.391 |
| P9 `[2,2,2,2]` (baseline) | L14 | 6085 | 2679 | 0.264 | 0.087 | e7~e1 0.302 |
| **P1 `[2,2,2,4]`** | L14 | 5693 | 2553 | **0.290** | 0.084 | e7~e0 0.328 |
| **P2 `[4,4,4]`** (drop L14) | **L12** | 4099 | 1894 | **0.301** | 0.066 | **e7~e1 0.388** |

**P2 REJECTED — the positional hypothesis confirmed, and it overshot.** P9's L12 (same position, with L14 above): famIoU 0.210 / effnum 3014 / core50 1362. P2's L12, promoted to last memory layer: famIoU **0.301 (+43%)**, effnum 4099, core50 1894 — *more* family-overlapped than the old last layer ever was (P9 L14 0.264), worst pair e7~e1 **0.388** > even broad control's L14 pairs, elevated at every P2 layer (L10 0.386). Mean famIoU over the layers sequential would actually touch: **P2 0.248 vs P9 0.211** — removing the worst layer *raised* the average interference surface, and parked it on rank-4 slots (2× destructive per overwrite). In-run corroboration: P2's L12 gate tracked P9's **L14** trajectory point-for-point from step 2k (0.69→0.961 vs L14's 0.81→0.969) — the high-trust last-layer role migrated immediately and structurally. Fit also paid: block-MSE min 0.215 vs P9 0.196 (+10%; losing the most action-proximal memory site costs leverage). Fails on every axis.

**P1 PASSES — r4-on-L14 leaves routing intact.** Capacity preserved (core50 2553 ≈ 2679, effK/h 120 = P9, j/p 0.39 — no shrink signatures), famIoU 0.290 within the sep5 probe band (P6/P8/P9 = 0.311/0.309/0.264), background 0.084 ≈ 0.087, lower layers *better* separated (L8 fam 0.132 vs 0.172), fit ≈ baseline (min MSE 0.202 vs 0.196), gate ladder intact with L14 slightly up (0.977). +25% value params (3.02 B, verified from checkpoint shapes) at zero routing damage.

### The rank mental model (the clean way to think about rank vs fit vs interference)

> **capacity ≈ footprint × per-slot rank;  interference-damage ≈ overlap × per-slot destructiveness. Rank multiplies BOTH sides.**

Consequences:
1. The (c=0.05, sep=5.0) routing pocket was selected under a **rank-2 capacity gate** (core50 ≥ ~1500 ⇒ ~3000 rank-units). Rank-4 relaxes exactly that constraint: compaction that was fatal at rank 2 (c=0.1 → core50 1162 = dead) is ~affordable at rank 4 (1162 × r4 ≈ 4650 rank-units ≈ 87% of P9's healthy 5360). **The rank-2-tuned contrastive ceiling need not carry to rank 4.**
2. Symmetrically, overlap costs more at rank 4 (bigger behavioral transform per shared slot) → same famIoU, more forgetting damage → argues for at-least-as-much separation.
3. Both effects push the same direction: higher rank supports smaller-better-separated footprints.
4. **Bounds on the re-look:** (i) the contrastive ceiling is not only capacity — it's **query collapse** (state-independent addressing, Entry 21/22c), which rank does NOT fix; guard query_intra ≤ ~0.93. (ii) **Compaction alone doesn't separate** (P3: c=0.1 shrank footprints, famIoU stayed ≈ control) — c-up only pays through sep's translation, so test it with sep=5 held. (iii) The sep benefit-cap is the genuine scene-sharing floor (famIoU ~0.26, forward-probe-proven scene routing) — rank-insensitive; **sep stays 5.0**. (iv) Under mixed rank the contrastive weight is global across layers → c-up also compacts the still-rank-2 L8/L10, which can't afford it; if the c-up cell fails, expect it to fail there first → fail-route = per-layer contrastive (sibling of the per-layer-sep idea, Entry 24 #3).

### Next: `[2,2,4,4]` A/B probes (LAUNCHED)

`[2,2,4,4]` = the max-affordable all-layers config: 3.62 B values (= P2's footprint, fits), boosts the two layers carrying ~70% of read mass/trust (L12+L14, per Entry 22d ladder), keeps L14 in place so no positional migration. Two cells, single-knob delta, interleaved run→audit→run→audit:

| cell | config | question |
|---|---|---|
| A | `[2,2,4,4]` c=0.05 sep=5.0 | does r4-at-L12 perturb L14's routing from below? (P1 only tested the last layer — nothing downstream of it) |
| B | `[2,2,4,4]` **c=0.1** sep=5.0 | does rank-4 make compaction affordable AND convert to lower famIoU via sep? (the model-driven contrastive re-look; untested combination) |

**Rank-adjusted gate:** famIoU ≤ ~0.28 AND query_intra ≤ ~0.93 AND capacity-in-rank-units ≥ baseline at the r4 layers (core50 ≥ ~1300 at L12/L14) AND r2 layers not cratering (L8/L10 core50 ≥ ~50% of P9's 534/1047). B graduates if it holds capacity and lands famIoU < A; A graduates if B fails the P3 pattern (famIoU ≈ A) or craters L8/L10; `[2,2,2,4]` (P1, checkpoint retained) is the validated fallback. Blast-radius contingency for the eventual sequential: per-layer sep bump on L12/L14 only.

Known risk either way: rank is a **fit** lever, not a retention lever (frozen router fills all ranks of a shared slot; realworld Entry 3) — the diagonal should move; env7-class genuine contention won't, and r4 overwrites hit 2× harder → β4 protection stays mandatory in the sequential stage.

Script: `probes/run_layerrank_probes2.sh` (run A → audit A → run B → audit B, ~23.5 h total). Audits: `audit_heldout_ranks2244_c005_10k`, `audit_heldout_ranks2244_c01_10k`. Analysis: `scratchpad/audit_ranks.py` pattern (extend RUNS).

### Cleanup
Deleted P2 `ranks_444` checkpoints (~64 G, rejected branch — wandb + audit JSONs retained) and `training_state` from P1 + P9 probe dirs (never resumed). **Kept**: P1 `ranks_2224` and P9 10k `pretrained_model`s — P1 is the fallback graduation candidate; both are the comparators for the pending 1-task plasticity probe (rank→fit conversion check, still to run).

### Appendix (2 Jul, discussion w/ Josh) — the rank model CORRECTED: fractional damage is ~rank-invariant to first order; net sign on forgetting is ambiguous and empirically decidable

Josh's 2-task critique of "rank multiplies destructiveness": with 1% (read-mass) overlap, doubling rank uniformly doubles capacity in BOTH contested and uncontested slots, so the *fraction* of a task's learned function that is corruptible stays ~1% — the "2×-destructive" claim double-counts (scales the per-slot numerator, forgets the denominator scaled too). **Accepted.** Damage ≈ Σ_{s∈overlap} reliance(s) × distance-moved(s); under uniform rank scaling the reliance *fraction* on the overlap is invariant, provided three assumptions — each with a specific leak:
1. **Reliance concentration** — with 2× per-slot capacity, SGD may achieve the fit relying on *fewer* slots more intricately. Invisible to routing metrics (retrieval is query/key-driven, rank-blind); lives purely on the value side. If reliance concentrates, the same slot-overlap carries a larger reliance fraction.
2. **Specialization drift** — rank-2's limited capacity is an implicit regularizer keeping contested slots at blunt shared primitives (which part-serve both tasks = the measured forward transfer). Rank-4 lets the overwriter specialize contested slots further from the shared solution → larger per-slot delta seen by the frozen reader.
3. **Threshold nonlinearity** — forgetting is cliff-y and concentrated (realworld E3, env7); 1+2 both push toward concentrated corruption at fixed average corrupted mass.
Counter-channel previously uncredited: **basin deepening** — rank-2 was *underfitting*, so rank-4 stores MORE information (deeper basins), and `final ≈ f(depth, exposure)` + the C result (steps deepened basins → retention +1.0 despite ↑exposure) says deeper basins absorb corruption. Retention-POSITIVE channel. And the surviving (modified) form of the original claim: **mixed rank `[2,2,4,4]` migrates capacity share into exactly the highest-famIoU layers** (L12/L14) — the honest "blast radius" is capacity-share migration, not per-slot arithmetic. **Net sign of rank on forgetting: ambiguous a priori.**

Josh's (B): "1% slot overlap" carries no importance info — right, and one level deeper than our weighted metrics: **read-mass is a proxy for importance** (true importance = ∂success/∂slot-corruption, never measured). Proxy has been well-calibrated at the tail so far (RTO called stack +60%, red-bowl +209%, env7→0), but a rank change is exactly the intervention that could de-calibrate it (rank-4 fits plausibly less redundant → higher importance per unit read-mass).

**Three empirical tests (all ~free):**
1. **Forgetting-per-unit-RTO calibration curve** — scatter x=per-task RTO, y=retention (final/init) over the ~70 task-points from the 7 rank-2 runs; the `[2,2,4,4]` sequential adds 10 points. On the curve → rank-invariant per unit overlap (Josh's A). Above → drift/concentration dominate. Below → basin-deepening dominates (C pattern). Task-matched version (each task vs its own 7-run history) kills task-identity noise. *(RTO = read-through-overwrite: fraction of a task's read WEIGHT on slots later tasks updated, mean over layers.)*
2. **Within-run rank contrast (diff-in-diff)** — `[2,2,4,4]` contains both rank classes under identical conditions (L8/L10=r2, L12/L14=r4). Measure contested-slot value-drift per layer (e.g. env7's core slots: ‖Δslot values across env0's block‖/‖before‖, from per-task checkpoints) → ρ = d(L14)/d(L8). Compare ρ_mixed vs ρ of an all-r2 baseline to cancel the layer-position confound. ρ_mixed ≈ ρ_r2 → no per-slot amplification; ρ_mixed ≫ → specialization drift confirmed. Same DiD for update-mass concentration.
3. **Slot-ablation importance probe** (new instrument, Josh-approved idea): zero the top-k contested slots on a frozen adapted checkpoint, measure the `--eval.type=loss` delta → calibrates read-mass against actual importance. ~minutes.

**Data availability (checked):** all 7 rank-2 runs' `memory_by_task` JSONs intact → test 1 buildable today. C's per-task checkpoints were deleted (E30 cleanup; final only) → test 2's rank-2 baseline = **protectB4** (all 10 per-task ckpts retained, same β4/prior; only mismatch 3k-vs-5k steps, ~cancels in the layer ratio). sep5 also retains 10/10. **⚠ RETENTION FLAG: the eventual `[2,2,4,4]` sequential's per-task checkpoints must NOT be cleaned until the drift analysis (test 2) is done.**

**Rank-2 calibration curve BUILT (2 Jul):** 70 task-points → `outputs/analysis/rank2_rto_retention.json`; generator `scripts/vla_analysis/rto_curve.py` (run the same extractor on the rank-4 sequential and overlay). Shape: RTO<20% → ret ~104% (net transfer), 0 collapses; 20–40% → 95%; 40–60% → 85%; **>60% → 7/10 tasks collapse to ≤5** (bin means mask bimodality — deep basins hold, shallow ones cliff, exactly the two-factor model). corr(RTO, drop): pearson −0.36 / spearman −0.42 — moderate, confirming exposure is only half the story (basin depth is the other half; blockmin_mse + init stored per point for exactly that regression).

### Update (3 Jul) — `[2,2,4,4]` A/B verdict: **A (c=0.05) GRADUATES — healthiest audit in the project; B (c=0.1) repeats the P3 pattern (compaction ≠ family separation)**

Both cells + audits completed clean (chain 19:02 → 16:42 UTC). Held-out audit (last layer L14; gate famIoU≤0.28 ∧ q_intra≤0.93 ∧ r4-core50≥1300 ∧ L8/L10 ≥ 50% of P9):

| cell | L14 core50 (rank-units) | L12 core50 (r-u) | L8/L10 core50 | famIoU L14 | bgIoU | e7~e1 | q_intra | verdict |
|---|---|---|---|---|---|---|---|---|
| P9 r2 baseline | 2679 (5.4k) | 1362 (2.7k) | 534/1047 | 0.264 | 0.087 | 0.302 | 0.912 | — |
| **A c=0.05** | **2981 (11.9k)** | 1687 (6.7k) | 923/1599 | **0.265** | 0.092 | 0.288 | 0.910 | **✓ GRADUATES** |
| B c=0.1 | 2156 (8.6k) | 1227 (4.9k) | 682/1055 | **0.279** | 0.073 | **0.367** | 0.931 | ✗ P3 pattern |

- **A answers the arch question: r4-at-L12 does NOT perturb L14 from below.** L14 famIoU 0.265 = P9's 0.264; core50 *up at every layer* (923/1599/1687/2981 vs 534/1047/1362/2679 — above even control's, before counting rank-units: 2.2–2.5× at the r4 layers); 4-layer mean famIoU 0.199 vs P9 0.211; fit fine (min MSE 0.201 vs 0.196). One structural shift: the gate ladder flattens (in-run gate L12 0.964 ≥ L14 0.952 — trust spreads over the two beefy layers), showing as mildly higher L12 famIoU (0.245 vs 0.210) — watch L12 channels in the sequential.
- **B disconfirms the contrastive re-look at this dose.** The rank-relaxed capacity gate ~held (rank-units ≥ ~92% of baseline; L8/L10 did NOT crater — the global-c bluntness worry was wrong), and background improved (0.073). But **famIoU went UP not down** (0.279 > A's 0.265) and the hub's worst channel degraded (e7~e1 0.367, approaching control's 0.355) — compaction pulls each basket task's queries tighter around the *same scene representation*, so sep=5 still can't translate the family apart. Same failure shape as P3, now demonstrated with capacity held by rank — the family overlap is scene-genuine, not a capacity artifact. q_intra 0.931 right at the guard. B's only win (background) is the half β4 protection already handles at write time. **c=0.05 confirmed; the contrastive question is closed at both ranks.**
- **Next:** graduate A → fresh 40k (`layer_ranks=[2,2,4,4]`, noloc sep5 recipe, clean schedule) → 40k audit → sequential with C's config (β4 + 5k steps/task + top_t 1536), **keeping per-task checkpoints** (⚠ retention flag) → overlay the 10 rank-4 points on the rank-2 RTO curve (test 1) + the within-run r2-vs-r4 DiD (test 2, baseline protectB4). Optional pre-flight: 1-task plasticity probe A-10k vs P9-10k (~30 min) to confirm rank→fit conversion before the ~4-day GPU commit.

---
## Entry 33 - 8 Jul 26 ([2,2,4,4] verdict: rank exonerated AND exhausted (+6 diagonal, on-curve retention, no drift amplification) — bottleneck = 14pp frozen-backbone fit + 12pp retention tax → STAGED PRETRAINING protocol launched (competence into the frozen backbone, memory as pure residual substrate))

### The [2,2,4,4] graduation result

Chain landed clean (40k pretrain held-in **81.9** best-of-family → sequential, C's config, 20 eps). Final **46.5%** (+2.0 over C, within noise), init **49.5** / peak **58.5** (both project bests), forgetting −3.0 (C: +1.0). Retention matrix (init→final): e4 35→20, e6 55→**75**, e9 40→20, e2 85→80, e7 25→**0**, e0 45→45, e8 20→**45**, e1 30→20, e3 80→80, e5 80→80.

| run | final | init | peak | forget | MSE-floor | L14 effnum/core50 | read-IoU | RTO | ch≥12% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protectB4 (r2,3k) | 40.5 | 43.5 | 49.5 | −3.0 | 0.150 | 5257/2534 | 0.051 | 29.8% | 6 |
| C steps5k (r2,5k) | 44.5 | 43.5 | 53.5 | +1.0 | 0.120 | 6630/3265 | 0.057 | 33.2% | 8 |
| **r2244 (r4@L12/14)** | **46.5** | **49.5** | **58.5** | −3.0 | **0.105** | 6394/3152 | 0.059 | 34.7% | 9 |

Slot pipeline validates (computed IoU 0.0592 == logged). Exposure ≈ C on every axis — rank-4 did NOT broaden footprints at matched steps/LR.

### Both pre-registered rank tests (E32): rank-invariant — Josh's fractional-invariance model confirmed

1. **RTO-curve overlay (test 1): ON the curve.** Task-matched residual vs each env's 7-point rank-2 history = **+1.3pp ≈ 0** (e6 +31/e8 +18 net-transfer absorbers; e7 −21 vs a floor-capped prediction; rest −7…+2). Points: `outputs/analysis/rank4_2244_rto_points.json`.
2. **Contested-slot drift DiD (test 2): r4 slots drift LESS ≈ pure gradient dilution.** ρ = d(L14)/d(L8) on env7's core across env0's block: protectB4 (all-r2) **1.26** (updated-slots) vs r2244 (L14=r4) **0.74** → DiD ≈ 0.59, ≈ the expected 2×-params dilution (0.63). No specialization-drift excess. (`scripts/vla_analysis/drift_did.py`; outputs in `outputs/analysis/`.)

**Why retention still looked worse: the fit gains landed on high-RTO tasks whose collapse floors are unchanged** — e7 init 10→25 then →0 regardless; e0 +20 init kept +5. More to lose, same losers. Run-specific aggravation: the 40k audit's e7~e0=0.445 watch-item materialized (e7←e0 overwrite **67.8%** vs C's 41.6%; env0 hits 74.5% of e7's L8 core mass vs protB4's 44.3%) — this pretrain converged the soup pair (prior-wide), concentrating e7's death into one channel. e3-as-heavy-writer appears in C too (5k-steps effect, not rank). Gate-ladder flattening (L12 0.977 ≥ L14 0.973) persisted with no visible extra damage channel — L12 watch-item closed.

### Why the fit gain was small

MSE floor −12% vs C but concentrated on already-easy tasks (soup+sauce −30%, book −21%); the hard dual-cycle set barely moved (**soup+cheese −2%**, mug+micro −6%, cheese+butter −9%). +50% params bought less than steps 3k→5k (−20%). ~Half the blocks stop improving by mid-block (identical pattern in C). **Per-slot capacity was never the binding constraint on the low tasks — the frozen backbone is** (base pi05 joint finetune = 72.6 on the same data).

### Bottleneck, measured (gap to 72.6 = 26.1pp)

**14.1pp fit-at-peak** (e7 +55, e1 +36 — basket dual-cycle) **+ 12.0pp retention-from-peak** (e4 35 noisy-cell, e7 25, e9 20, e0 15). Both halves rank-insensitive. Lever ledger now measured-and-spent: separation (flat), protection (β4 opt), steps (half-saturated at 5k), LR (backfires), rank (+2 net). Structural constraints: frozen backbone (fit half) + frozen router & gate≈0.97 (retention half).

### Discussion (Josh) → decision: STAGED PRETRAINING

Josh's 2-prong framing: (1) some tasks don't convert capacity into fit; (2) some convert but get overwritten. Options weighed: gate-desaturation penalty (first-order analysis: gate scale CANCELS out of seq-task-on-seq-task interference since ΔV ∝ 1/g — the real effect is raising the post-collapse floor by shifting pretrain competence into the frozen part ⇒ same bet as stronger pretrain, enforced structurally); more layers (extend DOWN, [4,6,8,10,12,14] — last-layer hub role is positional, earlier layers are the clean ones; E9/E11 precedent); β8×rank-4 (new cell: r4 private capacity may absorb the veto that starved writers at r2); **magnitude-based protection** (move (1−u)^β from top-t *ranking* to per-slot *gradient/LR scaling* — the E27 offline soft-suppression frontier that measured net-positive at all β, env7 RTO 81.5→41.8 at β=8, which the ranking implementation only delivered at ~β≈1 strength; strongest un-tried retention lever, PARKED not dead).

**Chosen: Josh's staged protocol** — (1) finetune base pi05 (no memory) on libero_90; (2) attach memory, train ONLY memory (values+keys+query/FiLM+gate) against the frozen backbone with the sep5 recipe; (3) sequential as usual. Memory becomes an additive residual/adaptation substrate; core competence lives in frozen weights by construction. What it buys: post-collapse floor = stage-1 zero-shot (e7 currently falls to literal 0 because corrupted memory < no memory); ~blank value table at stage 3 (no pretrain content to read through and lose); big iteration-speed win (stage 2 has no backbone grads/Adam → every future prior experiment is a cheap stage-2 rerun). What it does NOT fix: delta-on-delta overwrite of sequentially-learned fit. Open question the probe answers: does the sep5 routing pocket reproduce on a frozen backbone (standard audit gates: famIoU≤0.28 ∧ core50≥1500 ∧ q_intra≤0.93)? Pre-registered success criterion for the full chain: **no task ends below its stage-1 zero-shot floor, and final ≥ 50**.

### Code + launch

- New flag `--policy.train_memory_only` (pi05): `configuration_pi05.py` field; `PI05Policy._apply_train_memory_only()` (freezes everything without `.mlp.mem.` in the name — 60 memory tensors trainable / 2.45B params, 813 frozen / 4.14B) called from `__init__` (strict only when no pretrained_path — attach happens later on the load path), `post_load_setup`, AND after the try/except in `from_pretrained` (the hook is exception-swallowed there); `get_optim_params` now filters `others` by requires_grad. Smoke-tested fresh-init + load-path (pi05_base = structurally a stage-1 ckpt): only mem params trainable, groups clean (52 router tensors @ base LR + 8 value tensors @ memory_lr), backward reaches memory, flag-off byte-identical, strict-raise when memory absent.
- Scripts `job_scripts/nebius/libero_90/staged/`: `stage1_base50k_stage2_probe10k.sh` (stage 1 = E31-baseline recipe on libero_90, 50k, eval **libero_10 @ 50 eps at 50k = the zero-shot floor table**; stage 2 probe 10k compressed, frozen-base, sep5 recipe verbatim; audit `audit_heldout_frozenbase_10k`) — **LAUNCHED 8 Jul, tmux `staged`, log `outputs/staged1.log`**, ETA ~2 days stage 1 + ~8h probe + 35min audit. `stage2_full40k_stage3_sequential.sh` (40k clean schedule → `audit_heldout_frozenbase_40k` → C's sequential verbatim, run `libero_10_sequential_..._frozenbase_..._protect_beta4_steps5k`) — READY, launch after the probe audit clears.
- Grad-ckpt: stage 1 TRUE (measured requirement — plain pi05 bs32 OOMs without it, 29 Jun test); stages 2/3 FALSE (frozen backbone; r2 values-only no-ckpt bs32 = measured sequential precedent).

### Bookkeeping
- ⚠ E32 retention flag DISCHARGED (tests 1+2 done): r2244's 10 per-task checkpoints (~450G) deletable on Josh's word (keep final/`last`; JSONs+evals carry the analyses).
- Analysis artifacts: `outputs/analysis/{rank4_2244_rto_points.json, slots_2244.out, slots_C.out, drift_did.out}`.

---
## Entry 34 - 10 Jul 26 (STAGED protocol first results: stage-1 zero-shot floor ≈ 0 on all collapse tasks; frozen-base probe = a NEW ROUTING REGIME — gates naturally moderate at 0.63-0.68 (first time ever) but the routing losses lose their actuator and the audit FAILS separation (famIoU 0.390, inverted layer ladder) → nonlinear query head (query_proj_layers=2) implemented + probe LAUNCHED; script 2 held)

### Stage 1 (libero_90 base finetune, no memory, 50k) — healthy, and the floor table kills the floor-raising hope

Train MSE 0.09-0.13 at 50k (≈ joint pretrains' fit). **Zero-shot libero_10 @ 50 eps = 10.6% mean**; per env: e3 bowl+drawer **60**, e5 book **32**, e2 stove+moka 8, e0/e6/e7 **2**, e1/e4/e8/e9 **0**. So the post-collapse floor the protocol was meant to raise is ~0 exactly on the collapse-prone tasks (env7 floor = 2). Corollary worth recording: every sequential init we've ever measured (25-85%) was ~entirely memory adaptation, not backbone transfer — the diagonal IS the memory system. (Floor table = the pre-registered stage-3 comparison row; per-task successes recovered from the eval runner's per_task dump in `outputs/staged1.log` — lerobot-train wandb only logs the aggregate.)

### Stage-2 frozen-base probe (10k, sep5 recipe verbatim, train_memory_only) — Josh: "curves look very different." Confirmed; qualitatively new regime

| @10k (vs P9 joint probe) | P9 joint | frozenbase |
|---|---:|---:|
| mse_loss | 0.210 | **0.097** (0.130→0.097, real residual learning) |
| gate mean / L14 | 0.935 / 0.969 | **0.682 / 0.633** |
| query_intra / inter | 0.915 / 0.178 | **0.715** / 0.313 |
| routing sim in-run | 0.061 | **0.179** (flat from step 2.5k) |
| support_L14 | 6009 falling | **9160 rising** |
| grad_norm | ~1.2 | ~0.1 |

- **The gate finding is real and novel: first natural gate moderation in the project** (every joint config saturated ≥0.93; the E18 "overwrites pass at full strength" amplifier). With the backbone carrying the function, the network simply doesn't over-trust memory. This was the gate-desaturation lever (E33 option) achieved structurally, for free.
- q_intra 0.71 = strongly state-conditional addressing (healthy opposite of the E21 collapse) — the frozen backbone acts as an anti-collapse regularizer.
- Loss composition: MSE 0.097 vs aux ≈ 1.06 — **the routing losses already dominate the objective ~10:1** (stage 2 is organically a router-training phase).

### Held-out audit (`audit_heldout_frozenbase_10k`) — FAILS separation, smashes capacity

L14: famIoU **0.390** (control 0.349, P9 0.264; gate ≤0.28 FAIL), bg **0.188** (2.2× P9), core50 **6456** / effnum 14663 (2.4× control; capacity gate passed ×4.3). **Layer ladder INVERTED**: L8 is now the worst layer (famIoU 0.538, bg 0.272) instead of the cleanest — raw frozen features are generically similar early and differentiate with depth, the opposite profile of a routing-loss-shaped backbone. Family pairs uniform (e7~e0 0.385 ≈ e7~e1 0.399 ≈ e0~e1 0.386) — generic broad-footprint collision, not object-graded semantics.

### Mechanism (the diagnosis)

**The routing losses lost their main actuator: backbone co-adaptation.** In every joint pretrain, contrastive/sep gradients flowed through proj(x) into the backbone and reshaped the hidden states themselves — that's what actually bought the E20-26 compaction/translation. On a frozen backbone the only routing surface is one shared linear proj + FiLM, which cannot fold 90 interleaved task clouds into compact separated footprints. Weight-limited is disconfirmed (aux already 10:1 over MSE); step-limited unlikely (in-run sep flat 0.18 from 2.5k). Representation-limited.

### Why this is ambiguous rather than a clean fail (discussed with Josh)

Three of four damage-model terms moved favorably — gate 0.63-0.68 (attenuation at read time), near-blank values (no pretrain content to lose through shared reads), 2.4× slot capacity for β4 to steer writes into — and one badly (exposure: broad overlapping reads → high stage-3 RTO). The audit gate thresholds were calibrated on joint priors where high famIoU meant load-bearing shared content at gate 0.98; none of those side conditions hold here, so 0.28 may not transfer. The floor-raising benefit, though, is measured dead (floors ≈ 0).

### Decision (Josh): fix the router for frozen-x FIRST — nonlinear query head. Script 2 HELD.

**Code (shipped + smoke-tested):** `--policy.memory_layer.query_proj_layers` (default 1 = original single linear, byte-identical) + `query_proj_hidden_dim` (0→input_dim). `memory_lite.py`: `_build_query_proj` (Linear→SiLU→…→Linear), threaded via cfg into `QueryMLPLite`; param tagging now iterates all proj params (`pk_query_proj_param` — sequential-trainer `train_query_proj` plumbing unchanged); **`reset_parameters` fixed** (xavier looped over Linears — the old line assumed a bare `.proj.weight` and would crash on Sequential). Smokes: depth-1 state-dict keys unchanged (legacy checkpoints load); depth-2 forward/backward through the head; policy-level with train_memory_only = 68 trainable tensors (60 + 4×4 qproj), optimizer groups clean. ~2M params/layer at depth 2.

**Probe LAUNCHED (10 Jul, tmux `qproj2`, log `outputs/qproj2_probe.log`):** `stage2_probe10k_qproj2.sh` — single-knob delta from the failed frozen-base probe (`query_proj_layers=2`), run `libero_90_pi05_8_10_12_14_frozenbase_probe10k_qproj2_c0.05_sep5.0_noloc_rq512`, audit `audit_heldout_frozenbase_qproj2_10k`. ~8h + 35min.

**Read + decision rule:** primary = famIoU materially down from 0.390 toward ≤~0.28 with core50 ≥1500 (it starts at 6456 — huge room) and q_intra ≤0.93 (some RISE from 0.71 is expected and fine — the head can now tighten clusters); secondary = does the inverted ladder flatten, in-run sep < 0.18. If it clears/materially improves → stage-2-full graduates WITH `query_proj_layers=2` (**NB script 2 `stage2_full40k_stage3_sequential.sh` must gain the flag before launch**). If ≈ depth-1 → the frozen-feature separability ceiling is confirmed → run script 2 on depth-1 anyway as the end-to-end baseline (the changed damage model deserves one full measurement); depth-3/wider-hidden only if something suggests headroom.

### Cleanup (~360G freed; disk 43%)
- Failed depth-1 frozenbase probe: `checkpoints/` deleted (wandb + `audit_heldout_frozenbase_10k` retained — they're the comparison baseline for qproj2).
- sep5 + protectB4 sequentials: 9 intermediate per-task checkpoints each + final `training_state` deleted (their retention reason — the rank-2 DiD baseline for E33 test 2 — is discharged; finals + `last` + memory_by_task JSONs + evals + wandb retained).
- **`libero_90_pi05_base_nomem_50k` (the finetuned pi05) untouched** (weights + training_state), per Josh — it's the base for every stage-2 variant.
- Flag for later (not touched, different track): `outputs/train/realworld_v2` is 267G of packaged real-world test models from realworld-E1.

---
## Entry 35 - 12 Jul 26 (STAGED PROTOCOL VERDICT: DEAD as a frontier route — killed at 6/10 blocks. Held-in PARITY (81.7) but held-out adaptation collapses (inits −18pp, MSE floors 1.5-2×, e7 init 0): compact routing was a JOINT-TRAINING product; diffuse routing dilutes both reads AND writes. Pivot back to the joint track)

### Chain status + kill decision (Josh)

Stage-2 full (40k, frozen base, depth-1 head per E34) completed clean; 40k audit ran; stage 3 (C's config verbatim) auto-started and was **killed at 6/10 blocks** (~30k steps) — mid-flight results were clearly non-competitive and the pre-registered criterion (final ≥ 50) was already unreachable. Partial artifacts retained (memory_by_task tasks 0-5, evals through 25k, wandb); the 6 per-task checkpoints deleted.

### Stage-2 full: the dissociation that defines the postmortem

- **Held-in libero_90 eval 82.8 @20k / 81.7 @40k — PARITY with the best joint pretrains** (control 81.1, [2,2,4,4] 81.9), at MSE 0.087, gates drifting 0.68→0.75 (still ≪ joint 0.97), routing sim stuck at 0.18.
- So: memory-as-residual on a frozen backbone serves SEEN tasks perfectly well. The failure is specific to being an **adaptation substrate for new tasks**.

### Stage-3 partial (6 blocks): the diagonal collapses

| ord | env | task | FB init | r2244 init | C init | FB block-min MSE | vs r2244 |
|---|---|---|---:|---:|---:|---:|---:|
| t0 | e4 | two mugs | 25 | 35 | 35 | 0.141 | 1.49× |
| t1 | e6 | mug+pud | 30 | 55 | 45 | 0.142 | 1.80× |
| t2 | e9 | mug+micro | 30 | 40 | 35 | 0.267 | 1.49× |
| t3 | e2 | stove+moka | 65 | 85 | 75 | 0.182 | 1.57× |
| t4 | e7 | soup+cheese | **0** | 25 | 10 | 0.244 | 1.86× |
| t5 | e0 | soup+sauce | (killed mid-eval) | 45 | 25 | 0.098 | 2.03× |

Mean init over matched 5 = **30.0 vs r2244's 48.0 / C's 40.0**. MSE floors 1.5-2× worse on EVERY block. Retention no better either (e9 30→5 collapse-shaped). Even perfect retention caps this run ~30-35.

### Mechanism (from the partial JSONs — and it kills the simple write-starvation story)

Reads are ~2× broader (L14 effnum 11-15k vs r2244's 5-7k, per the audit) — but writes are NOT starved, they're **diffused**: 4L unique updates 95k/190k/255k (t0/t1/t4) vs r2244's stable 55-62k. Broad, low-concentration TF → the per-batch top-1536 churns constantly → updates smear over enormous slot sets with tiny per-slot totals; the retrieval mixture spreads output over many weakly-specialized slots. Compounding: gates frozen at ~0.75 attenuate the new task's memory contribution (ΔV must grow to compensate, within a 5000-step budget), the backbone never co-adapted to integrate large memory corrections, and β4's (1−u) gate over famIoU-0.39-broad prior cores scatters later tasks' writes further (update counts GROW with block index).

**The central lesson: the compact, separated, high-trust routing that made joint-pretrained memory a good few-shot substrate is a product of BACKBONE CO-ADAPTATION, not of the memory module or its losses.** Freezing the backbone keeps the competence but forfeits the substrate quality; E34's qproj2 result already showed no query-head capacity buys it back.

### Banked findings (the protocol wasn't free to run, so keep what it taught)
1. **Zero-shot floor table** (stage-1, 50 eps): mean 10.6; ~0-2% on every collapse-prone task → all sequential inits ever measured are ~pure memory adaptation.
2. **Gate moderation happens naturally on a frozen base** (0.63→0.75 vs joint 0.97) — but it is NOT a free anti-overwrite knob: trust and plasticity are coupled (this run is the demonstration).
3. **Routing separation/compaction is a joint-training technology** (frozen-feature equilibrium famIoU ≈ 0.39 regardless of query-head capacity).
4. Stage-2-style memory-only training: ~1.18s/step, no ckpt, huge VRAM headroom — reusable machinery (`train_memory_only`, `query_proj_layers`) if a frozen-base variant is ever wanted.
5. Held-in eval CANNOT gate substrate quality (81.7 parity here) — only held-out adaptation probes can.

### Pivot (back to the joint track; the E33 lever ranking resumes)
1. **Magnitude-based protection** (per-slot gradient/LR scaling by (1−u)^β instead of top-t reselection) on the [2,2,4,4] prior with C's config — the E27 offline frontier (net-positive at all β, env7 RTO 81.5→41.8 at β=8) that the ranking implementation never delivered. Sequential-only, no pretrain.
2. **50-ep re-eval of r2244's final** (~7h) — de-noise the 46.5 frontier number before building on it.
3. **6-layer depth** ([4,6,8,10,12,14] r2 joint pretrain) — the remaining capacity shape (E33).

### Cleanup
- Stage-3 partial: checkpoints deleted; JSONs/evals/wandb retained. Stage-2 40k prior (23G) + its audits retained for reference (E21 precedent for documented negative results); stage-1 base (`libero_90_pi05_base_nomem_50k`) retained — it's the floor-table reference and reusable for any future frozen-base work.

---
### Entry 35 addendum (12 Jul, deep-dive at Josh's request) — the full mechanistic chain, with code findings

**Code finding 1 (the actuator, confirmed):** `memory_lite.py` forward passes live `x` into `query_proj` — NO detach. In joint pretrains the contrastive/sep gradients therefore flow THROUGH the query into the backbone (and at E24-measured magnitudes: aux terms 30-91% of MSE). Joint routing quality was bought by sculpting the hidden states; in stage 2 that path dies at `requires_grad=False`. qproj2 (E34) already showed head capacity is not the binding constraint.

**Correction to this entry: the substrate is NOT blank.** Per-slot ‖θ‖ (down+up, L8/L14): frozenbase-40k mean 2.50/3.09 vs sep5-joint-40k 1.93/2.75 (untouched-slot init ≈0.9). Stage 2 built joint-magnitude residual content — slightly LARGER (the ΔV ∝ 1/gate compensation, gates 0.68-0.78). So stage-3 reads on un-adapted slots deliver stage-2 libero_90 residuals into libero_10 behavior, not zeros.

**The plasticity chain (Q: why is plasticity down when per-task support is UP?):** support was never the capacity metric — concentration is. Five measured links:
1. Frozen features → no compaction/separation possible (code finding 1) → diffuse routing: per-batch L14 effnum during the sequential 10-12k (FB) vs 3-7k (r2244); audit footprints 14.6k vs 6.1k; effK/h ~190 vs ~120 (near-flat retrieval scores).
2. Diffuse reads → low per-slot TF → the per-batch top-1536 churns → writes SMEAR: unique updated L14 slots 27.5k/46k/66k (t0/t1/t4, growing with block) vs r2244's stable 20-24k; update events/slot p90 204-716 vs 838-1193 — the would-be specialist slots get ~4-5× fewer repeated updates.
3. Coverage collapse: fraction of the task's read MASS on slots it adapted = 43-53% (t1/t4) vs r2244's 64-76%; core50-adapted 43-54% vs 72-94%. Half the retrieval mixture is stage-2 residual content, not task adaptation.
4. Transmission loss: gates frozen (W_g not trainable in sequential) at 0.68 (L8) - 0.89 (L14), vs 0.97 joint — the adapted contribution is attenuated on top.
5. (Unquantified residual) the frozen downstream layers never co-adapted to consume large memory corrections.
Net: block-min MSE 1.5-2.0× worse on all six blocks; inits −18pp. grad_norm ≈ equal (~0.02) — not an optimizer-signal problem.

**Q: why do layers behave differently (inverted famIoU ladder)?** Two different causes for two different ladders. Frozen-base famIoU tracks RAW FEATURE GEOMETRY: early-layer expert hiddens are generically similar across tasks (L8 0.538) and task-differentiate with depth (L14 0.390) — while breadth stays flat (12-15k at every layer). Joint famIoU tracks the TRUST/CONTENT EQUILIBRIUM: both gate and read-mass ladder UP with depth, MSE parks shared load-bearing content at the action-proximal layer, and separation pressure loses exactly there (L14 worst) while winning at the low-traffic early layers (L8 best) — i.e. the joint ladder was never "deep features are less separable" (the frozen run proves the opposite); it's where the MSE-vs-separation tug lands. Corollary: the frozen run still ladders TRUST upward (gate L8 0.68 → L14 0.89) even as famIoU improves with depth — the two ladders are separable phenomena.

**Q: is "state+lang not enough to separate routing"?** No — the information is sufficient (joint runs separate the same nominal inputs; E28: scene carries ~20× the language signal). What's insufficient is a FIXED representation with only a head on top: stage-1 features are organized for action prediction, not task discrimination; separability was an emergent property of routing-loss gradient reaching the backbone. Secondary (plausible, circumstantial): value-content anchoring — in-run sep flatlines by step 2.5k, exactly when value content becomes useful at the initial routing locations; moving routing away from built content costs MSE, pinning the equilibrium early.

**Implication:** the staged protocol cannot be patched router-side (qproj2 = the direct test). Any fix must let routing gradients touch the backbone at some phase — which is the joint protocol. Closed.

---
## Entry 36 - 12 Jul 26 (The debate resolved by measurement: frozen features ARE separable (linear probe ~98%) but CROWDED (inter-task cos ~0.9); joint training was blowing the geometry apart at exactly the memory layers (0.93→0.46). Router failure = anchored optimization, not information. → ROUTER WARM-UP v1 (geometry-before-content, Josh's protocol) LAUNCHED; contribution reframed around the staged setting)

### The discussion (Josh's challenges, and where each landed)

1. **"Why isn't the router learning to separate, given a nonzero input difference?"** — the sharpened form of the E35 question, and it exposed my "representation-limited" claim as imprecise. The mechanistic answer, assembled this session:
   - **Value anchoring at 40× LR** (Josh's reading, confirmed): values (1e-3) specialize onto the *initial random* routing within ~2k steps; the router (keys/proj in the base group, 2.5e-5) then faces first-order MSE cost for any move. The sep-flatline-at-2.5k in both frozen probes is the fingerprint. In joint runs the same LRs exist but backbone drift keeps re-shuffling the geometry — no anchor ever sets.
   - **No exploration across key cells**: sep's gradient flows only through the top-M candidates (hard selection); moving a task to genuinely different slots means crossing key-Voronoi boundaries — a plateau. Joint x-drift supplied the hops; frozen x + 2.5e-5 supplies none. (Why qproj2 failed: capacity was never the constraint.)
   - The FiLM/language channel that carries the cleanest per-task signal atrophies under training (E28) and nothing in the objective prevents it.
2. **Contribution reframe (Josh)**: the staged setting IS the thesis — "post-train a VLA for its domain (stage 1 = sim stand-in, skipped in reality), then bolt on a sparse LoRA-lookup that adds capabilities without erasing old ones." The joint protocol is the methodologically weaker story (bespoke pretrain). This reverses E35's "closed" verdict on staging: the goal isn't to beat 46.5 via staging, it's to make staging work.
3. **"Separation never converted to points" corrected (Josh)**: E8 measured 16.5→34.5 from introducing separation — it converts when interference is binding; it stopped converting in the joint pi05 era once protection/basins absorbed exposure and fit became binding (E27). In the STAGED regime routing quality is a **fit lever** (the E35 dilution chain: compaction → concentrated writes → slot specialization → self-adapted read mass), so separation should convert here again.
4. **Language-only routing (Josh's idea) — evaluated, parked**: E28's forward probe already measured it (q=β): basket family IoU **0.545 vs 0.215** scene-based — doubles the binding collision (near-duplicate unseen instructions co-locate under any smooth map; mpnet cos 0.86), and constant per-task queries = the E21 state-independent regime = per-task-adapter baseline rebuilt minus state-conditional composition. Keepable kernel: a hybrid language-only HEAD (private sections) — post-deadline queue.
5. **Move memory layers up (Josh's idea) — measured, negative** (probe below).

### The feature-separability probe (new instrument: `scripts/vla_analysis/feat_probe.py`)

Hooks on each expert layer's `.mlp` input (= the router input), mean-pooled per sample; 4 sweeps (stage-1 & joint sep5 ckpts × libero_90 & libero_10), layers [4,8,10,12,14,15,16,17]; linear task probes + centroid cosine geometry. Data: `outputs/analysis/featprobe/*.npz`. (Gotcha fixed: task vocabs contain empty entries — skip ValueError from the per-task dataloader.)

| linear probe acc | L4 | L8 | L12 | L14 | L16 | L17 |
|---|---|---|---|---|---|---|
| stage-1 libero_90 (90-way) | 89.5 | 98.1 | 98.1 | 97.9 | 97.5 | 96.6 |
| stage-1 libero_10 held-out | 98.0 | 98.6 | 98.9 | 98.3 | 98.0 | 97.4 |

| inter-task centroid cos | L8 | L12 | L14 | L15 | L16 | L17 |
|---|---|---|---|---|---|---|
| stage-1 (frozen) | 0.93 | 0.89 | 0.87 | 0.86 | 0.86 | 0.93 |
| joint (co-trained) | **0.46** | **0.48** | **0.49** | 0.61 | 0.69 | 0.86 |

- **Josh right: the information was never missing** — linearly separable at ~98%, held-out included (caveat: frame-level splits share episodes → accuracies optimistic; geometry numbers unaffected).
- **The features are crowded**: all tasks in a ~0.9-cos cone, inter≈intra — differences live in low-variance directions that dot-product retrieval cannot resolve unless the query proj amplifies them. **Joint training was doing exactly that to the features themselves** (0.93→0.46, precisely at the memory layers, decaying above L14) — the co-adaptation actuator, now quantified at the geometry level.
- **Move-up dead**: separability flat L8→L16, L17 worse; crowding improves marginally with depth. [8,10,12,14] stays (also preserves all baselines).
- **The needle**: basket family at cos **0.978–0.994** frozen (joint: 0.89–0.93) — still linearly discriminable; the warm-up's hardest test.

### Code shipped: `--policy.train_router_only` (+ smoke)

Freeze everything except keys + query_proj/FiLM (28 tensors); values pinned at init (slot_up zero). Smoke verified the two load-bearing mechanics: **zero values ⇒ MSE gradient on the router is exactly ~0** (warm-up purity) and **contrastive+sep gradients DO reach keys/proj** (the learning signal); `get_optim_params` drops the empty values group; `train_memory_only` regression intact (60/2-group).

### LAUNCHED: router warm-up v1 (12 Jul, tmux `rwarmup`, ~2.9h + 35min audit)

`staged/stage2_router_warmup10k.sh` → run `libero_90_pi05_8_10_12_14_frozenbase_rwarmup10k_lr1e-4_c0.05_sep5.0_noloc_rq512`, audit `audit_heldout_frozenbase_rwarmup_10k`. Config: router LR 2.5e-5→**1e-4**, c=0.05 kept (Josh: comparability; recalibrate later if needed), sep5/noloc/rq512, [8,10,12,14], 10k compressed schedule. **Decision tree**: famIoU ≤~0.28 ∧ core50 ≥~1500 ∧ q_intra ≤~0.93 → anchoring hypothesis confirmed, geometry-before-content becomes the method → discuss A-then-sequential. Collapse signature (core50 crater / q_intra >0.95) → unopposed contrastive won → c down, ~3h loop. Still ~0.39 → the sep-through-top-M formulation itself can't find the probe's directions → fallback: **seed the query proj from the linear-probe directions** (we have the probe weights).

### Design decisions from the discussion
- **A vs B after warm-up** (A = short values-on-90 phase, B = straight to sequential): Josh prioritizes iterating the geometry to near-perfect first (the warm-up↔audit loop is ~4h/iteration — the staged protocol's iteration dividend), expects A to help downstream. Note: anchoring is a FEATURE once geometry is good (content cements the map); A also calibrates gates (untrained ≈0.5 after warm-up) and re-audit after A is warranted (values perturb downstream x → routing shifts, E30 mechanism).
- **Per-task LoRA-FT baseline** (vla-memory.md §5, never run; PEFT is wired — `use_peft`): three roles — (i) the fit CEILING of adapter-adaptation on this exact frozen backbone (turns the "LoRA-FT is enough" premise into a number; separates "what the lookup costs" from "what freezing costs"), (ii) the reviewer-mandatory comparator (answer: task-identity-free routing, fixed params, measured forward transfer — e6 30→75, e8 20→50), (iii) effort compass (LoRA-FT ~55 → substrate near ceiling, work on retention; ~70 → memory design leaves fit on the table). ~8–17h; slot after the warm-up chain.

---
## Entry 37 - 12 Jul 26 (ROUTER WARM-UP v1: ALL GATES CLEARED — famIoU 0.145, the best routing prior in project history, on a FROZEN backbone, in 3h. Anchoring hypothesis confirmed end-to-end; the "scene-genuine 0.26 floor" was substantially an anchored-equilibrium artifact. → A-phase (values-only, ROUTER FROZEN) + sequential LAUNCHED)

### Warm-up result (10k router-only, lr 1e-4, c0.05/sep5/noloc/rq512, frozen stage-1 backbone)

In-run @10k: routing sim **0.050 mean AND L14** (P9 joint: 0.061/0.085 — beat the joint run outright, and L14 matches the mean for the first time ever: no MSE-trust equilibrium → no last-layer pathology), q_inter **0.016**, q_intra 0.883 (inside P9's healthy band — the unopposed-contrastive/E21 collapse never came at c=0.05), support_L14 5191, effnum 10.4k.

Held-out audit (`audit_heldout_frozenbase_rwarmup_10k`), all gates cleared:

| held-out L14 | famIoU | core50 | bgIoU | e7~e0 | e7~e1 | e0~e1 |
|---|---:|---:|---:|---:|---:|---:|
| control@40k | 0.349 | 2643 | 0.127 | 0.391 | 0.355 | 0.302 |
| P9 joint (prev best) | 0.264 | 2679 | 0.087 | 0.255 | 0.302 | 0.235 |
| fb-depth1 (anchored) | 0.390 | 6456 | 0.188 | 0.385 | 0.399 | 0.386 |
| **RWARMUP** | **0.145** | **2955** | **0.066** | **0.234** | **0.138** | **0.062** |

- famIoU **45% below the best joint prior**; capacity ABOVE it (core50 2955; no shrink shortcut — effnum 6505 ≈ P9's 6085); q_intra 0.883 ≤ 0.93. Generalization gap in-run→held-out ≈ 2.9× (P9's was 4.3×).
- **Flat layer ladder** (0.155/0.144/0.129/0.145) — first ever; more evidence the joint famIoU ladder was the MSE-trust equilibrium, not geometry.
- **Basket family cracked**: e0~e1 0.062 = background level (P9 0.235); e7~e1 halved (0.138); even the shared-soup pair e7~e0 (0.234) is below every prior built. The E24-28 "scene-genuine ~0.26 floor" was substantially the anchored equilibrium.

Mechanism chain now closed end-to-end: features separable-but-crowded (E36 probe) → joint training separated by reshaping features (inter-cos 0.93→0.46) → on frozen features the router was blocked by value-anchoring at 40× LR + no exploration (E36) → remove the anchor (values pinned at zero) + 4× router LR → the router finds the probe's discriminative directions on its own. **Geometry-before-content works.** (Full credit: Josh's protocol, over my initial "representation-limited" skepticism.)

### A-phase design decision (Josh's Q: values-only with frozen router, or keys/queries too at small LR?)

**Chose: router FROZEN (option 1).** (i) MSE's router-gradient points back at the joint attractor (famIoU ~0.26, trust ladder) — small LR only slows the walk back, and it erodes a certified asset; (ii) routing drifts anyway via upstream-value→x perturbation (E30 Finding 2) — minimize the controllable part and measure the rest with a re-audit; (iii) values-on-frozen-router demonstrably fits (every sequential block ever); (iv) crisp protocol: aux-only geometry → frozen router → MSE-only content, consistent with the sequential's frozen-router constraint. **Reserve trigger for option 2** (router unfrozen at ~2.5e-6 ≡ train_memory_only + tiny optimizer_lr, zero new code): A-phase MSE plateauing meaningfully above the joint pretrains' libero_90 level (~0.13-0.16) = separation-vs-usefulness mismatch.

### Code + chain (LAUNCHED 12 Jul, tmux `stageA`, log `outputs/stageA.log`)

- New modifier flag `--policy.freeze_memory_router` (composes with train_memory_only): trainable = memory module MINUS keys/query_proj = **32 tensors** (values 8 @ memory_lr + gate/value_proj/swilu 24 @ base LR). Smoked: freeze pattern, optimizer groups, both prior modes regression-clean.
- `staged/stageA_values10k_seq.sh`: **A phase** (10k values-only on libero_90 from the warmed router ckpt, aux losses kept ON as pure telemetry — grads dead-end on the frozen router but the in-run routing-sim log becomes the live drift monitor; held-in eval @10k = seen-task plasticity check) → **re-audit** `audit_heldout_frozenbase_rwarmupA_10k` (deployed geometry after value training; informational) → **sequential** (C's config verbatim: β4 + 5000 steps/task + top_t 1536, 20 eps, per-task ckpts) → run `libero_10_sequential_pi05_8_10_12_14_frozenbase_rwarmupA_..._protect_beta4_steps5k`. ETA ≈ A 3.2h + eval ~1h + audit 0.6h + sequential ~40h.
- Pre-registered sequential reads: inits vs fb-depth1's mean-30 disaster and r2244's 48 (the dilution chain predicts routing quality converts to FIT here); self-adapted read mass (fb 43-53% → want 70%+); retention matrix vs floor table; RTO overlay (damage-per-exposure).

### Artifacts
- **PRESERVED (Josh): the warmed-router checkpoint** `libero_90_..._rwarmup10k_lr1e-4_.../checkpoints/` — the backtrack point if we later want to un-freeze and let the router keep learning (e.g. option-2 arm, different A designs, different value schedules all restart from here).
- Deleted: qproj2 probe checkpoints (~37G; negative result — audit + wandb retained). Kept: stage-2-full 40k prior (23G, documented negative result / E35 postmortem subject), stage-1 base (untouchable), featprobe npz (analysis data), all audits.

---
### Entry 37 addendum — launch gotcha caught at first contact (12 Jul)

The A-phase's first launch silently ran in **router-only mode**: the warmed checkpoint's saved `config.json` carries `train_router_only: True`, which supersedes the CLI's `train_memory_only` (by the flag-precedence rule). Killed at step ~400, fixed by passing `--policy.train_router_only=false` explicitly in the A stage, relaunched; correct mode confirmed in-log (`train_memory_only+freeze_memory_router: 32 param tensors trainable`). **Rule for all future staged chains: any freeze-mode flag set in an upstream stage must be explicitly disabled downstream — checkpoint configs carry them forward.** Cleanup: qproj2 probe checkpoints deleted (~37G; audit+wandb retained); warmed-router checkpoint preserved in full (18G) as the backtrack point.

---
## Entry 38 - 13 Jul 26 (rwarmupA sequential postmortem: the killer is ROUTING DRIFT — value training re-points the frozen router above L8 (self-IoU 0.21-0.33 vs L8's exact 1.000); exposure/writes exonerated; the certificate was void by task 0's own block. Fix: FROZEN-BASE ROUTING (dual-path) implemented + smoked; stageB chain (A rerun -> audit -> 5-task sequential) LAUNCHED)

### The run (killed at 7/10 blocks, Josh's call)

`libero_10_sequential_..._frozenbase_rwarmupA_..._protect_beta4_steps5k` — C's config verbatim from the A checkpoint (warmed router + 10k values-only fill). Ran 16:08 Jul 12 → killed 12:15 Jul 13 at 7/10 blocks.

Eval trajectory (env: init → @35k): e4 15→20, e6 55→35, e9 5→0, e2 **80→25**, e7 **40→5**, e0 30→35, e8 50 (just trained). Seen-avg @35k = **24.3** (r2244 @35k: 44.3). Matched-5 inits = **39.0** vs fb-depth1 30.0 / C 40.0 / r2244 48.0.

### Fit side (Josh's MSE/gate observation): real, quantified, SECONDARY

- Block-min MSE mean 0.150 = **+20% vs C** (0.120; rank-matched) and +40% vs r2244 (0.105; r4 at L12/14) — but 15-25% BETTER than fb-depth1 on every matched block. Rollout inits ≈ C despite the MSE gap.
- Gates frozen (W_g untrainable in sequential) at **0.74/0.88/0.90/0.89** per layer vs 0.97+ in every joint run — consistent with the +20% (ΔV must be ~1/g larger per unit correction; plus no downstream co-adaptation). The A-phase substrate itself is healthy: held-in **83.9%** @10k (parity with the best joint pretrains), MSE 0.079-0.092, no reserve trigger.
- The E35 dilution chain is otherwise FIXED by the warm-up: per-batch L14 effnum 5,624 ≈ joint (r2244 4,989 / C 5,060; fb 9,317), self-adapted read mass 61-86% ≈ joint, writes 82-143k unique slots (fb: 95-255k), e7 got its best-ever cold start (init 40; joint era 8-25). **Routing quality converts to fit in the staged regime — the run did not die of fit.**

### Retention side: exposure and writes exonerated, then the anomaly

- Cleanest static exposure ever: **3** pairwise channels ≥12% (e7←e0 45%, e2←e8 29%, e0←e8 12%) vs r2244's 9 / control's 26.
- Write magnitudes identical to joint: e7-core drift across e0's block (same channel, same metric as the E33 DiD) = 0.49/0.43 core&updated at L8/L14 vs r2244's 0.51/0.38, at LOWER mass exposure. End-state contested-slot ‖θ‖ 5.59 vs 5.76 (L8, r2 both) — no 1/g write blowup.
- Against the 70-point rank-2 calibration curve (RTO 20-40% → ret 95%; 40-60% → 85%): e2 ret **31%** @ RTO 37, e6 **64%** @ 24, e7 **12%** @ 49 — **~3× the historical damage per unit exposure** — while low-exposure tasks sit ABOVE the curve (e4 133% @ 27, e0 117% @ 12). e6 lost 15pp across a block that perturbed **0.6%** of its core mass. Static overwrite cannot explain this run.

### The discovery — READ-SIDE ROUTING DRIFT (new instrument: pre/post deployment audits)

Ran the held-out audit on the seq-35k checkpoint and compared per-task footprints to the post-A audit (same instrument, same demos, seed-matched; `audit_heldout_rwarmupA_seq35k`):

| self-IoU (seq start → 35k) | L8 | L10 | L12 | L14 |
|---|---|---|---|---|
| weighted footprint (trained t0-t6) | **1.000 exactly** | 0.24-0.32 | 0.17-0.28 | 0.17-0.25 |
| binary core-50 set | **1.000** | 0.16-0.23 | 0.13-0.23 | 0.14-0.16 |

- **L8 is the built-in control**: its input is the immutable stage-1 features (no memory below it) → frozen queries → IoU exactly 1.0 (also validates the instrument end-to-end). Everything above L8 re-routes almost completely: value updates at L8-12 perturb the residual stream; the frozen router's queries at L10-14 move; retrieval hops key-cells.
- **Untrained tasks drift too, in a scene-proximity gradient** (t7=e1 basket 0.50 < t8=e3 0.60 < t9=e5 0.73): the perturbation δx = g·Δmem(x) is state-dependent — routing moved most where content was written.
- **Timeline**: IoU(post-A audit, own-block reads) ≈ 0.21-0.39 already at t0 — **the certified famIoU-0.145 geometry was void within the first block**; each task fits whatever transient routing exists during its block (own-block→35k ≈ 0.66-0.78 vs the ~0.93 block-JSON measurement ceiling), and the anchor keeps sliding after.
- **Still-mine mass** (fraction of eval-time reads on slots the task adapted AND that weren't later overwritten): e2 35-46% (the biggest collapse, lowest mass), e6 41-47%, vs e4 59-69% (the task that held). Substituting effective for static exposure puts the collapses back ON the calibration curve (>60% → historical collapse band). The "3× anomaly" was RTO under-measuring exposure once reads move.
- **Separation survived; anchors didn't.** Deployed geometry @35k: famIoU 0.190 / bg 0.107 / core50 4188 — eroded from 0.145/0.066/2955 but still better than ANY joint prior audited (P9 0.264/0.087). A still-clean map that has MOVED is as fatal as a collapsed one → more separation would not have saved this run; **stationarity, not geometry, is the missing property.**
- Corollary: β4 protection references stale maps in both directions under drift (steers writes off where early tasks USED to read, while they now read elsewhere).
- Gate note (Josh's hypothesis, retention half): mid-sigmoid gates (0.74-0.89) sit on the steep part — g·(1−g) 4-6× the input-sensitivity of joint's saturated 0.97 — a secondary channel from the same root (the stream the frozen gate reads is non-stationary). The primary channel is the query re-routing.

### Mechanism synthesis

The staged substrate lacks **routing stationarity**. E37 proved the geometry is achievable on frozen features; nothing holds it — the router input is the live stream, value training moves the stream, and the warmed router separates tasks along low-variance directions of the crowded frozen-feature cone (E36: inter-task cos 0.86-0.99) so there is no margin to absorb the motion. Joint priors tolerate the same nominal channel because (1) co-trained features are spread (cos ~0.46 → wide margins), (2) heavy pretrain content (‖θ‖ 1.83-2.6 vs A's 1.06) makes sequential writes a smaller relative field change (+35% vs +75%), (3) saturated gates are insensitive. Refines E35's verdict: joint training doesn't (only) create the geometry — it creates the margins and ballast that keep it still. (r2244 post-sequential audit for the cross-regime drift number was started and killed on Josh's call — the L8-vs-L10/14 within-run control is decisive without it; partial audit deleted.)

### Fix implemented: FROZEN-BASE ROUTING (dual-path), `--policy.memory_layer.use_frozen_base_input_features` (default false = byte-identical)

Memory ROUTING (query projection + gate) reads the backbone features as they would be WITHOUT any memory contribution; the value/output path (LoRA transform, swilu) stays on the live stream. Addressing becomes stationary by construction — L8's IoU=1.000 extended to every memory layer. The warmed router was trained with values pinned to zero, i.e. on EXACTLY the features the frozen branch serves → it drops in unchanged.

- **Training/joint path** (`modeling_pi05.py`): lazy fork — streams are identical below the first memory layer, so a memory-free suffix stream (`compute_frozen_suffix_layer`, mirrors the expert side of `compute_layer_complete`: shares the per-layer prefix KV since the prefix never attends to the suffix) runs only from fork_lo to fork_hi under no_grad, its per-layer mlp inputs passed as `router_x` into each memory layer. At fork_lo the live mlp-input IS the routing feature (router_x=None). Dropped after fork_hi.
- **Inference path** (suffix-only denoise): pass A = expert forward with memory bypassed (per-wrapper capture state on `MLPPlusMemory`; own deepcopy of the prefix KV cache — the attention appends to it), stashing each memory layer's mlp input; pass B pops the stash as router_x. Strict stash hygiene (exactly-1 per layer, consumed-after-pass asserts, exception-safe).
- `HashingMemoryLite.forward(..., router_x=None)`: query + gate read router_x; value path unchanged. Incompatible with `memory_only` (guarded).
- **Smokes (all pass, float32 tiny model, real code path):** T1 flag-on @ zero values == flag-off BITWISE (live pass untouched); T2 frozen-stream fidelity 0.00e+00 (any mask/rotary/residual bug would fail); T3 router_x stationary under value bumps while live x moves; T4 inference dual-pass bitwise clean + cache uncorrupted + stationary; T5 grads flow (values + query_proj) with the no_grad stream present; T6 gradient-checkpointing parity 0.00e+00; T7 single-memory-layer edge case. Policy-level probe on the real warmed checkpoint: flag parses, "Frozen-base routing ENABLED" fires, 32/841 freeze pattern, **1.11 s/step** (old A: ~1.18 — overhead within noise; the fork costs ~40% of one expert-suffix pass, expert ≈ 300M of 6.6B).

### LAUNCHED: stageB chain (13 Jul, tmux `stageB`, log `outputs/stageB.log`)

`staged/stageB_frozenroute_A10k_audit_seq5.sh`: **A rerun** (10k values-only, router frozen, frozen-route ON — gates must train against the frozen-branch input they'll read at deployment; old-A content was measured compatible but 4h buys exact train/deploy consistency) → **audit** `audit_heldout_frozenroute_rwarmupB_10k` (expect ≈ the warm-up certificate 0.145/2955 — the frozen branch serves exactly the warm-up features) → **sequential, FIRST 5 TASKS** `[0-4]` (deadline scope; contains all three collapse cases e2/e6/e7 + the e7←e0 genuine channel), C's config verbatim + the flag. Runs: `libero_90_..._frozenroute_rwarmupB_values10k_...` → `libero_10_sequential_..._frozenroute_rwarmupB_..._steps5k_tasks5`.

**Tripwire (zero GPU):** when task 0's block JSON lands, IoU(t0 block reads, post-A audit) must sit near the ~0.93 measurement ceiling (old wiring: 0.28). If low → the fix isn't holding in the real trainer → kill.

**Pre-registered reads on landing:** (1) per-task self-IoU pre→post ≈ 1.0 at ALL layers (the property, in production); (2) still-mine mass ≈ 1 − static-RTO; (3) retention back on the calibration curve at the cleanest-ever exposure — e2/e6 should hold near init; e7←e0 (famIoU 0.234, genuine soup channel) remains the residual risk with β4; (4) fit ≈ unchanged or slightly better (inits ~39-40; tasks no longer chase their own routing within a block).

### Cleanup
- Deleted (analysis retained — JSONs/evals/wandb/audits): killed-sequential checkpoints (251G), old A-phase checkpoints (36G). Disk 47%.
- Untouched: `libero_90_pi05_base_nomem_50k` (stage-1 base), warmed-router checkpoint (stageB source + backtrack point), r2244/C/sep5/protectB4 finals, stage-2-full 40k (documented negative), all audits, `realworld_v2`.
- New analysis artifacts: `audit_heldout_rwarmupA_seq35k` (the drift measurement), scratchpad `rerouting.py`/`drift_rwA.py`/`norms_cmp.py` patterns.

### Next steps
1. stageB lands (~16-18h): tripwire after block 0, then the pre-registered reads above. If retention holds → extend to 10 tasks (the script's sequential stage is the same command with `online_task_ids=[0..9]`) and/or 50-ep re-eval.
2. If e7 still collapses via the genuine soup channel → that's the honest residual: magnitude-based protection (E27 offline frontier) is the parked lever.
3. Post-deadline queue unchanged: per-task LoRA-FT baseline on the frozen stage-1 (fit ceiling / reviewer comparator), hybrid language-only head, joint warm-up variant.

---
### Entry 38 addendum (13 Jul, discussion) — the drift tax is retroactive: this channel was live in EVERY sequential run ever

Josh's observation, confirmed on existing data: routing drift (values mutating the stream the frozen router reads) was not introduced by the staged protocol — it has been present in every sequential run in the project; the regimes differ only in the damage coefficient.

- **Joint-era evidence, reread with today's lens:** E30 Finding 2's footprint broadening (L14 effnum 5257→11789 as write magnitude/duration grew) IS this mechanism — we measured it and named it "broadening" because we never compared a task's own footprint before/after. **B (lr2x) is the joint-era drift casualty**: damage-per-exposure ~3× β4's (ret −9.5 @ RTO 33% vs −3.0 @ 30%) — the same anomaly signature that cracked the rwarmupA run. Its "backfire" was attributed entirely to bigger overwrites; doubled ΔV also doubles stream perturbation.
- **Why joint survived it at standard LR:** margins (feature cos 0.46 vs the frozen cone's 0.86-0.99), ballast (relative field change +35% vs +75%), gate saturation. The static-exposure machinery (calibration curve, timing-matched cliffs, β4's +6.5pp) worked because drift-per-write was small there — but not zero.
- **The deep consequence: the rank-2 calibration curve's ABSOLUTE level is contaminated.** "85% retention at 40-60% RTO" was measured with drift on. No drift-free sequential run has ever been observed; **stageB is the first stationary-addressing run in project history.** Pre-registered read upgraded accordingly: retention ON the historical curve at matched exposure ⇒ drift was staged-only; retention ABOVE the curve ⇒ the gap is the historical drift tax, quantified behaviorally — and it applies to the whole family, joint runs included.
- **Follow-ons if confirmed:** (1) the flag for the JOINT track — with a variant: a joint router was trained on live features *including pretrain memory content*, so its frozen routing input should be features computed with **values snapshotted at sequential start** (same dual-path code, frozen branch carries the snapshot; ~10GB extra at r2), not memory-free. "r2244 + snapshot-frozen routing" is the candidate frontier run. (2) The direct retrospective number: the r2244 post-sequential audit (pre→post self-IoU for the joint regime, ~1.1h GPU) once stageB frees the card.
- Relative conclusions of the project (write-budget, separation-as-translation, incidental/genuine protection split, capacity/interference axis) stand — all A/B'd within-regime with drift on both arms. The absolute retention levels, and specifically the lr2x reading, carry the unquantified tax.

Launch status at write time: A-phase stepping cleanly (step 200, 1.12s/step, loss 0.111, frozen routing confirmed in-log); tripwire ~21:15, chain ETA ~06:30-08:00 Jul 14.

---
## Entry 39 - 14 Jul 26 (stageB verdict: FORGETTING SOLVED — first flat retention matrix in project history (MSE forgetting +0.0-1.7%, routing self-IoU exactly 1.0000); the low score is NOT retention, it is the staged substrate's rollout-fit conversion (inits 35 vs r2244's 48), decomposed per-task: e9 = warm-up footprint dilution (measured), e4 = backbone-integration gap (footprint-controlled), e7 = staged WINS. The "apparent forgetting" (e6 60→25) is 20-ep eval noise on a +0.5%-drifted function)

### The run

`libero_10_sequential_..._frozenroute_rwarmupB_..._protect_beta4_steps5k_tasks5` — chain completed clean 07:57 Jul 14 (A-phase 10k frozen-route -> audit -> 5-task sequential, C's config + `use_frozen_base_input_features=true` everywhere). Chain hygiene verified in-log: "Frozen-base routing ENABLED" in all stages, 32/841 tensors trainable in A, values-only 2.42B/6.6B in sequential, protection store folded after every task. (Bookkeeping note: the overnight chain monitor was killed right after launch, so the ~21:15 tripwire never ran live; the chain completed on its own and the tripwire was run retroactively this morning — it PASSES, see below.)

Eval (20 eps/cell):

| step | e4 | e6 | e9 | e2 | e7 | seen-avg |
|---|---|---|---|---|---|---|
| 5k | 10 | | | | | 10.0 |
| 10k | 10 | 60 | | | | 35.0 |
| 15k | 10 | 25 | 5 | | | 13.3 |
| 20k | 25 | 25 | 10 | 70 | | 32.5 |
| 25k | 20 | 35 | 5 | 70 | 30 | **32.0** |

First-5 comparison @25k (all 20 eps except C/sep-era 50): stageB **32.0** / rwarmupA 35.0 / C 39.0 / r2244 42.0. Inits: stageB **35.0** (10/60/5/70/30) / rwarmupA 39.0 / C 40.0 / r2244 48.0. Block-min MSE mean: stageB **0.127** / rwarmupA 0.156 / C 0.131 / r2244 0.120.

### 1. The fix works EXACTLY as designed — stationarity is total (code exonerated)

- **Tripwire (retroactive, all 5 tasks × all layers):** IoU(block reads, post-A audit) = **0.90-0.93 at L8/L10/L12/L14 uniformly** — at the measurement ceiling (rwarmupA's immutable-L8 control: 0.92-0.93). rwarmupA's same numbers above L8 were 0.25-0.39. The training path routes stationarily.
- **The definitive instrument** (new post-sequential audit `audit_heldout_frozenroute_rwarmupB_seq25k` vs post-A): per-task footprint self-IoU = **1.0000 exactly, every task, every layer** (rwarmupA: 0.21-0.32 above L8). Deployed geometry after 5 blocks is byte-identical to post-A: famIoU 0.144/0.144, bg 0.066/0.064, core50 2958/2958, effnum 6507/6507. **The E38 read-side drift channel is dead.**
- **Parameter-level proof** (checkpoint diff 010000→015000, e9's block): the ONLY tensors that changed are the 8 slot_up/slot_down — keys, query_proj/FiLM, gate, value_proj, swilu, backbone all bitwise-zero delta. Combined with self-IoU=1.0: the sole interference channel left in the whole system is *values on shared slots*.
- Certificate transfer: warmup-audit ↔ postA(B)-audit per-task IoU = L8 1.000, L10-14 0.980-0.988 (live-route A-phase: 0.926-0.945). The 0.98 residual is the value_proj-bias delta (the warmup audit ran on the live path, whose stream at values=0 still carries g×bias; the frozen branch is strictly memory-free) — expected, not drift. A-phase itself no longer moves routing.

### 2. FORGETTING: SOLVED. The E3-style MSE matrix is flat — first time ever

Ran the full 5×5 forgetting matrix offline (paired-noise `_eval_loss_on_seen_tasks`, 16 batches/task, all 5 per-task checkpoints):

| ckpt | e4 | e6 | e9 | e2 | e7 |
|---|---|---|---|---|---|
| 5k | **0.1140** | 0.6065 | 1.6084 | 0.8788 | 0.9069 |
| 10k | 0.1140 | **0.1049** | 1.6067 | 0.8794 | 0.9039 |
| 15k | 0.1144 | 0.1050 | **0.1891** | 0.8597 | 0.8882 |
| 20k | 0.1145 | 0.1050 | 0.1900 | **0.1326** | 0.8863 |
| 25k | 0.1159 | 0.1054 | 0.1910 | 0.1329 | **0.1682** |

Just-trained → final: e4 **+1.7%**, e6 **+0.5%**, e9 **+1.0%**, e2 **+0.3%**, e7 **+0.0%**. For calibration: realworld-E3's matrix had red-bowl **+209%**; rwarmupA lost e2 80→25 in rollouts. Nothing in project history is close to this flat. Untrained-task cells move ≤0.3% between checkpoints (paired noise works); small negative transfer visible pre-training (e7 0.907→0.886 before its block = mild positive transfer).

Supporting numbers, same story: cleanest exposure ever (RTO t0-t2 10-12%, t3 3.7%, **zero** pairwise channels ≥12%, read IoU L14 0.055-0.113); still-mine 56-77%; e6's core-50 hit by later writers at only 0.2-1.4% of read mass per layer; and the direct field measurement — across e9's block, e6's mass-weighted value field changed **1.2-2.1%** (core-50: 0.1-0.6%), e4's 0.8-1.3%.

**So the e6 "collapse" (60→25 at e9's block) is NOT forgetting.** The function e6 reads moved +0.1% across that block. Verdict: 20-ep binomial noise around a marginal policy (p≈0.35-0.45 → ±11pp/cell; the 60 was the outlier draw) plus possibly a sliver of closed-loop brittleness (the 25k video shows a wrong-mug grab — object-binding wobble — while 15k failures are second-step timeouts). A 50-ep re-eval of the retained per-task checkpoints would fully settle it; the MSE matrix already settles the mechanism question.

Protection was inert here — nothing to protect: t1-t4's read mass on high-usefulness (u≥0.25) prior slots was 0.1-0.8%, realized writes there 0.0%. Note discovered en route (applies to ALL runs, incl. r2244): u(s) is peak-normalized and access distributions are so sharp (max/p99 ≈ 10-16×) that u at the core-50 boundary is only ~0.035 → (1-u)^β≈4 ≈ 0.87 — β4 only ever vetoes the top ~1% mega-hot slots. It paid +6.5pp in the joint era because damage concentrated exactly there; worth remembering if protection is ever re-tuned.

### 3. What actually went wrong: rollout-fit conversion, not retention

The 10pp gap to r2244 @25k is entirely on the diagonal (inits 35 vs 48) — and MSE does NOT explain it (stageB MSE floor 0.127 ≈ r2244's 0.120, better than C's 0.131; stationary targets fit EASIER than rwarmupA's moving ones, −19% MSE). The staged substrate converts flow-matching fit into rollout success much worse than joint substrates. Per-task decomposition (the useful part):

- **e9 (mugs→microwave, init 5 in BOTH staged runs vs 35-40 joint) = warm-up footprint dilution, measured.** The warmed router assigns e9 a footprint 2.1-2.4× everyone else's (audit L14 core50 6980 vs 2100-3800; block core50 7081 vs r2244's 3759; per-batch L14 effnum 8973 vs r2244's 5896). top_t=1536 covers only 17% of its per-batch reads (r2244: 26%); updates smear (ev/slot p50 17, p90 549 vs r2244's 837) → no slot specializes → mushy mixture. This was visible in the E37 audit table (e9 = 14993/6973, 2.4× the others) and unremarked — the capacity gates were min-only. **Audit-methodology fix: add a footprint-size dispersion/max gate (flag any task >2× median core50).** The E38 claim "per-batch effnum 5624 ≈ joint, dilution fixed" was a mean over blocks that masked e9's 8973.
- **e4 (two-mugs dual-cycle, init 10-20 vs 35 joint) = the backbone-integration gap, now footprint-CONTROLLED.** e4's staged footprint is essentially identical to joint (L14 effnum 6254 vs 6274, writes 17.5k vs 20.1k, same ev/slot scale) and its MSE lands 0.114 — yet rollouts 10-20 vs 35. With routing/dilution equalized, what remains is E35's link-5 residual, isolated cleanly for the first time: a backbone that never co-trained with memory in the loop does not integrate memory corrections into long-horizon composition (videos: first pick-place cycle completes, second hovers/fails). One-step MSE cannot see compounding.
- **e7 (soup+cheese, init 30; rwarmupA 40) = the staged geometry WINS where routing binds.** Joint-era e7 inits were 10 (C) / 25 (r2244) with famIoU-collision routing; the warmed geometry's clean basket separation converts to the best e7 fits on record. The staged substrate trades: fixes routing-bound tasks, loses integration-bound ones.
- Amplifier on everything: gates are frozen mid-sigmoid constants under frozen-route (L8/L10/L12/L14 = 0.74/0.80/0.72-0.76/0.81-0.85; rwarmupA's live gates self-amplified to 0.88-0.92 during blocks; joint 0.97) — every correction delivered at 75-85% strength on a substrate where memory IS the task competence (stage-1 floors ≈ 0-2).

### 4. The pre-registered drift-tax read is VOID at this exposure — the window can't discriminate

All five tasks sit at RTO ≤ 12%, i.e. in the calibration curve's <20% bin where the rank-2 history already predicts ~zero forgetting. Observed zero forgetting is consistent with the curve AND with the drift-tax hypothesis — the run proves stationarity holds in production and that clean-exposure forgetting is nil, but it cannot yet price the historical drift tax. The 5-task window also ends exactly where rwarmupA's catastrophe began (its e2 80→25, e6 −15, e9→0 all happened at blocks 5-7). The A/B that shows the fix's retention payoff requires blocks 5-10.

### 5. Next steps (proposed, in value order)

1. **Port stationary addressing to the JOINT track — "r2244 + snapshot-frozen routing"** (E38-addendum follow-on, now maximally supported): joint substrate has the fit (init 48), stageB proves stationarity eliminates function-level forgetting, and r2244's within-window losses (e4 35→20, e9 40→20, later e7→0) are the drift-suspect retention tax. Code delta: the frozen suffix branch must run memory WITH slot values snapshotted at sequential start (the joint router/gate were trained on live-with-content features; memory-free features would be OOD for them) — snapshot slot_up/slot_down at seq start (~7GB bf16 at r2244 sizes), thread a use-snapshot flag through the frozen-branch memory call. Expected: init ≈ 48 with a flat matrix → new frontier >46.5, and the drift tax finally quantified within the joint regime.
2. **Extend stageB to 10 tasks** (same command, `online_task_ids=[0..9]`, ~40h): the matrix predicts the seen-avg holds ≈ its init mean through blocks 5-10 where rwarmupA fell to 24.3. That is the controlled catastrophe-elimination demonstration (mechanism E38 → fix → matched-config rescue) — the cleanest causal story the project owns.
3. **50-ep re-eval of the 5 retained per-task checkpoints** (~cheap): every conclusion above rides on ±11pp cells; de-noise the trajectory, resolve e6's wobble empirically.
4. **Staged-track plasticity levers unlocked by zero forgetting** (if the staged track stays a thesis pillar): with drift dead and exposure clean, the levers that backfired via interference are now safe by construction — memory_value_lr 2e-3 (B's fit was best-in-family, its −9.5 retention was the drift tax; pre-registered: under frozen-route the tax should vanish), top_t 1536→~3072 (e9's coverage 17%→34%), steps 5k→7k. Cheap sequential-only reruns from the same A checkpoint.
5. **e4-class integration gap: do not chase in-protocol** — E35/E36 stand (needs backbone co-adaptation at some phase = joint track). The staged track's honest scope: routing-bound tasks won, integration-bound tasks conceded, zero forgetting.

### Artifacts
- New: `audit_heldout_frozenroute_rwarmupB_seq25k` (10 JSONs, the 1.0000 measurement), MSE matrix (`scratchpad/mse_matrix.{py,jsonl}` — reusable instrument: paired-noise per-checkpoint loss eval via the trainer's own `_eval_loss_on_seen_tasks`), `field_change.py` (mass-weighted value-field perturbation from checkpoint pairs), `stageB_analysis.py` (full battery), `ckpt_diff.py` (safetensors group diff).
- Retained: stageB A checkpoint + all 5 per-task sequential checkpoints (needed for #3/#4 and any drift-tax follow-ups), both post-A audits, block JSONs, evals/videos, wandb.
- GPU free.

---
## Entry 40 - 14 Jul 26 (E39 follow-through: joint track PARKED (off-thesis, Josh); the conversion-gap attack goes 4-way parallel — (1) affine slots + no gate [LAUNCHED], (2) lr 2x, (3) steps 7k, (4) top_t 3072. Code shipped: lora_slot_bias (affine "lora + value") + eval_final_episodes; 20/20 smokes)

### Decisions from the E39 discussions (recorded so the reasoning survives)

1. **Joint track parked as off-thesis (Josh).** The thesis is "take an off-the-shelf VLA and add capabilities without erasing old ones" (with router-abstention as the future-work arc), so "r2244 + snapshot-frozen routing" is at most a *pricing ablation* (what does co-training buy at matched stationarity), not the mainline. Josh's counter to the E39 integration-gap story — "base + adapters is how everyone finetunes; the backbone shouldn't have to adapt" — is a testable claim, NOT settled by the E39 exclusion inference: our memory differs from standard LoRA-FT in four taxed ways (sparse routed mixture of 144 rank-2 fragments vs dense unconditional adapter; 4 expert-MLP sites vs attention+MLP everywhere; a trust gate in front; softmax averaging). **The per-task LoRA-FT baseline (E36, reviewer-mandatory, PEFT wired) is the arbiter**: LoRA-FT at 35-50 on e4/e9 => backbone sufficient, the deficit is OUR machinery's conversion tax (closable in-protocol); LoRA-FT ~10 => frozen-backbone adapters can't do those tasks at all, scope the claim. PROPOSED, not yet scheduled (not in the 4-way batch). Supporting fact: libero-90 has no dual-cycle tasks, so no stage-1 finetune can "embed composition subspaces" — the joint runs' e4/e9 rollouts (35-40) were installed by VALUES on a frozen backbone at sequential time, so composition-through-values is demonstrably possible; the joint edge is conversion machinery, not backbone knowledge.
2. **Gate REMOVED rather than saturated (Josh's call, agreed).** The E39 candidate mechanism for the conversion gap: the A-phase gate is calibrated on libero-90 where the backbone is competent and memory auxiliary (settles 0.72-0.85 mid-sigmoid), then deployed where memory carries ~everything (stage-1 floors ~0-2) through a valve whose delivered magnitude is state-dependent (g(1-g) sensitivity ~5x saturated) and never saw libero-10. Saturation-pressure would patch this; removal deletes it structurally, with no aux weight to tune. Every healthy joint run effectively ran ungated (0.97-0.99, "no modulation" — E19). We keep per-channel state-dependent modulation anyway via swilu (`value_proj(out * silu(swilu_proj(x)))`). `mem_gated=false` is an existing flag; must apply from the A phase onward (values calibrate ||dV|| ~ target/g; stripping at deploy overshoots ~1.3x). Not kept for the abstention roadmap either: an MSE-trained trust valve is not an OOD-abstention detector — build that purpose-built later. (Gate-variance probe — is the 0.77 a constant or a modulator? — noted, not run.)
3. **"Lora + value" = per-slot AFFINE, not a layer split (from Josh's hybrid idea).** value_i(x) = U_i V_i x + b_i. The homogeneous LoRA (up @ SiLU(down @ x)) has no constant term — the DC component of a correction must ride x's stable cone direction, spending rank. The per-slot bias learns the DC directly (no U-V product coupling => robust to e9-style diluted writes), frees rank for the state-dependent residual, and restores the additive-constant class of value_type="vector" WITHOUT handing state-conditionality to the router (which the warm-up deliberately de-resolves within-task: q_intra 0.883 — vectors-by-routing would rebuild the E21/E22c per-task-bias pathology). Layer-split variants (lora early / vectors late or vice versa) rejected: L12/L14 carry the read mass, trust, and action proximity (r2244's winning allocation), and constants there would be router-addressed precisely where within-task resolution is weakest.
4. **e6's 60->25 formally attributed to eval noise + marginality** (Josh's challenge "would noise explain +/-30pp?"): a single -35pp cell at 20 eps under constant p~0.36 is a ~2-5% draw; with ~12 adjacent-cell pairs per run, >=1 such swing per run is near a coin flip — and every baseline shows them (C's e9 35->50 "negative forgetting"; r2244's e4 55->20 in-window). The claim rests on the paired-noise MSE matrix (+0.1% across e9's block), not on noise arguments. All 4 new arms carry a 50-ep final eval (below) to shrink the headline cell to +/-7pp.

### Code shipped (smoked 20/20, `scratchpad/smoke_affine.py`)

- `--policy.memory_layer.lora_slot_bias` (default false): per-slot bias b_i (v_dim) on LoRA slots, added to the slot output before corruption/weighted-sum. Zero-init => flag-on is BITWISE identical to flag-off until trained (smoked); legacy checkpoints load with `missing={slot_bias}` and keep the zero init (smoked on the module load path). Tagged `pk_value_param` + `fixed_lr` => value optimizer groups (both trainers), TF-IDF top-t row mask (`_get_value_params` extended; dim-0 slot indexing uniform — smoked), protection, and the offload gather (smoked) all apply unchanged. Files: `memory_config.py`, `memory_lite.py`, `lerobot_sequential_train.py`.
- `--eval_final_episodes` (sequential trainer, default 0 = unchanged): intermediate evals keep `eval.n_episodes`; the eval after the FINAL task uses this count. Helper `_eval_n_episodes_for_task` (+ fixed an `idx` shadowing hazard at the eval call site by capturing `task_pos` at the loop head). Selection logic smoked.

### The 4-way parallel batch (Josh: filesystem imaged -> 4 GPU VMs, one arm each)

All arms: 5 tasks [0-4], C's sequential config, frozen-base routing, beta4, 20-ep evals + **50-ep final**, idempotent skip-guards. Arms 2-4 are sequential-only from the EXISTING stageB A checkpoint (`...frozenroute_rwarmupB_values10k.../checkpoints/last` — already on disk, baked into the image). Arm 1 runs its own A phase (value-path changes require it; the warmed router drops in unchanged — frozen-branch routing is a function of backbone+keys+query only, which also makes the post-A audit provably identical to `audit_heldout_frozenroute_rwarmupB_10k`, so it is skipped).

| # | script (`job_scripts/nebius/libero_90/staged/`) | delta vs stageB | tests | key read |
|---|---|---|---|---|
| 1 | `stageC_affine_nogate_A10k_seq5.sh` | `lora_slot_bias=true` + `mem_gated=false` (A rerun + seq) | affine capacity + gate-miscalibration hypothesis jointly | e4/e9 inits vs 10/5; e6-class stability at 50-ep final; A-phase MSE vs 0.092 |
| 2 | `stageB_seq5_lr2x.sh` | value_lr 1e-3->2e-3 (end 2e-4) | drift-tax-free lr (joint-era B: best fresh diagonal +4, -9.5 retention = the tax, now dead by construction) | inits up with retention flat => bank it; retention drop => the lr tax was never (all) drift |
| 3 | `stageB_seq5_steps7k.sh` | steps/task 5000->7000 | optimization budget on stationary targets | MSE floors + inits; forgetting must stay flat (matrix if needed) |
| 4 | `stageB_seq5_topt3072.sh` | top_t 1536->3072 | the e9 DILUTION hypothesis (E39): per-batch L14 write coverage 17%->34% | e9 init off 5 => dilution real; e9 still ~5 at sharper writes => conversion/substrate, not budget |

Expected magnitudes (honest): arms 2-4 are worth ~+3-7pp average and ~0 on e4 (its MSE is already joint-level at joint-identical footprints); arm 4 is a hypothesis test more than a fix. Arm 1 is the mechanism bet. The LoRA-FT baseline (above) remains the queued arbiter for whatever gap survives the batch.

### Status

- **Arm 1 LAUNCHED** 14 Jul 14:18 (tmux `stageC`, log `outputs/stageC.log`): flags confirmed in-config (`lora_slot_bias: True, mem_gated: False`), **28 param tensors trainable** (= 32 - 8 gating + 4 slot_bias, as predicted), frozen-base routing enabled, step 200 loss 0.111 @ 1.13 s/step (stageB A: 0.111 @ 1.12 — biases start at zero and values are still small, so early parity is expected). ETA: A ~3.1h + held-in eval ~1h, sequential ~12h, chain lands ~07:30 Jul 15.
- Arms 2-4: scripts ready + syntax-checked; launch on the cloned VMs (`bash job_scripts/nebius/libero_90/staged/<script>` in tmux). If this box's stageC is mid-A when the image is taken, the clone's arm-1 script re-runs A from scratch (guards key on completed checkpoints only).
- Cross-arm eval note: arms are mutually comparable (same seeds/eval protocol); all differ from stageB itself only by their single delta (stageB @25k baseline row: 20/35/5/70/30 = 32.0 @ 20 eps).

### Update (14 Jul, VM 2) — arm scripts recreated in git; ARM 3 (steps7k) LAUNCHED

- **Gotcha found on the cloned VMs: the four arm scripts never propagated.** `job_scripts/` is gitignored (`.gitignore:27`; the 224 tracked scripts were force-added historically), and the E40 scripts were written without `git add -f` — so commit `a8ac9ff` carried the code + log but not the scripts, and the clones came up without them. All four recreated verbatim from the stageB template + the E40 spec table and force-added: `stageB_seq5_{lr2x,steps7k,topt3072}.sh` + `stageC_affine_nogate_A10k_seq5.sh` (the stageC one is a for-the-record reconstruction — arm 1 is already live on the source box and its run names may differ; check its `outputs/stageC.log` before reuse). All sequential stages carry `--eval.n_episodes=20` + `--eval_final_episodes=50` (50 eps only for the eval after the LAST task's training; flag verified at `lerobot_sequential_train.py:399-404`). **Rule reaffirmed: new files under `job_scripts/` require `git add -f`.**
- **Arm 3 LAUNCHED** on VM 2, 14 Jul 17:43 (tmux `steps7k`, log `outputs/steps7k.log`, wandb `x269apkt`): run `libero_10_sequential_..._frozenroute_rwarmupB_..._top_t_1536_protect_beta4_steps7k_tasks5`, sequential-only from the existing stageB A checkpoint. Config dump verified: `online_steps_per_task=7000`, frozen-base routing, train_memory_only+freeze_memory_router, β4 protection, top_t 1536, value_lr 1e-3→1e-4, tasks [0-4], 20-ep evals + 50-ep final. Stepping cleanly: step 200 loss 0.399, grdn 0.010, 1.10 s/step. Final checkpoint = 035000 (5×7k); ETA ~19-20h (7k-step blocks + 20-ep evals).
- VM-2 disk cleanup (clone-local; all `pretrained_model` weights kept): deleted never-resumed `training_state` from the 5 stageB per-task checkpoints (5×19G — run complete, `reinit_optimizer_each_task` makes them dead weight; the weights stay for the MSE-matrix/drift instruments), the stage2-full 40k prior's intermediate `020000` ckpt + `040000/training_state` (final 40k weights kept per E35), and the r2244 pretrain's intermediate `020000` (graduated 40k prior weights kept). Untouched: stage-1 base (in full), warmed-router backtrack ckpt, stageB A ckpt (in full), all finals/`last`, audits, memory_by_task JSONs, evals, `realworld_v2`.

### Update (14 Jul, VM 3) — ARM 2 (lr2x) LAUNCHED; stepping cleanly like the others

- **Arm 2 LAUNCHED** on VM 3 (`computeinstance-e00xwgqsddb43xnsz3`), 14 Jul 17:58 (tmux `lr2x`, log `outputs/lr2x.log`, wandb `i6zojqts`): run `libero_10_sequential_..._frozenroute_rwarmupB_..._top_t_1536_protect_beta4_lr2x_steps5k_tasks5`, sequential-only from the existing stageB A checkpoint. Config dump verified: `memory_value_lr=2e-3 → 2e-4` linear (the single delta vs stageB), frozen-base routing ENABLED, train_memory_only+freeze_memory_router (32/841 tensors), β4 protection, top_t 1536, tasks [0-4] × 5000 steps, 20-ep evals + 50-ep final.
- **Stepping cleanly, in-family with arm 3:** step 200 loss 0.343, grdn 0.010, `lr:2.0e-03` in-log (the 2× peak confirmed live), 1.08 s/step (arm 3: 0.399 / 0.010 / 1.10). Final checkpoint = 025000 (5×5k); ETA ~14-16h.
- VM-3 disk cleanup (clone-local, ~200G freed, disk 56%→48%; all `pretrained_model` weights kept): same recipe as VM 2 — deleted never-resumed `training_state` from the 5 stageB per-task checkpoints (5×19G) and the stageB A checkpoint (19G; arms consume its `pretrained_model` only), the stage2-full 40k prior's intermediate `020000` + `040000/training_state` (~56G; final 40k weights kept per E35), and the r2244 pretrain's intermediate `020000` (23G; graduated 40k prior weights kept). Untouched: stage-1 base (in full, per Josh), warmed-router backtrack ckpt, all finals/`last` symlink targets (verified before deletion), audits, memory_by_task JSONs, evals, `realworld_v2` (267G, per Josh).

---
## Entry 41 - 15 Jul 26 (E40 3-arm postmortem + the conversion-gap probe program: five instruments, two falsified mechanisms, one corrected over-claim -> a 3-layer failure model. The e6-after-e9 drop is REAL (4/4 replication; E39/E40 noise attribution REVERSED). Soft (grad_scale) protection shipped + calibrated; denoised-chunk error adopted as the gate metric; 3 arms in flight (topt3072 / softprotect / bs64))

### The three completed E40 arms (rsynced to one box; env order e4/e6/e9/e2/e7; finals 50-ep, stageB final 20-ep)

| arm | inits | mean init | final | block-min MSE | block-END MSE | eval-time paired MSE (diag mean) |
|---|---|---|---:|---:|---:|---:|
| stageB (baseline) | 10/60/5/70/30 | 35.0 | 32.0 | 0.1274 | 0.1410 | 0.144 (E39) |
| affine+nogate | 20/55/10/75/38 | 39.6 | 32.8 | 0.1190 | 0.1274 | 0.129 |
| lr2x | 35/65/25/80/32 | **47.4** | 35.6 | 0.1132 | 0.1239 | 0.125 |
| steps7k | 10/55/10/70/32 | 35.4 | **36.0** | 0.0984 | 0.1235 | 0.126 |

Config integrity verified from checkpoints + wandb for all arms (each ran exactly its single delta). Josh's framing question: losses lower, perf flat — why?

### Arm verdicts

- **Arm 1 (affine+nogate): both hypotheses dead, cleanly.** A-phase healthy (held-in 80.8 vs stageB-A 81.7; MSE comparable). The biases LEARNED (nonzero on ~all 147k slots, ||b|| ~0.6x the UV-path norm at every layer, stable A->seq) — the parameterization works; the DC term was not the binding constraint. Gate removal delivered full-strength corrections — also not binding (values recalibrate to the gate during training either way). +4.6pp init is at the noise edge; final 32.8 == baseline. DECISIONS: `mem_gated=false` retired (gate stays — also operationally simpler: gated arms reuse the rwarmupB A checkpoint); `lora_slot_bias` retired from the recipe, flag kept in code.
- **Arm 2 (lr2x): the only init mover (47.4 = r2244's 48)** — and it gave ~all of it back (final 35.6). Grad norms 0.022 max (no instability). See the correction below for how much of the init story survives.
- **Arm 3 (steps7k): dead lever at the endpoint.** The -23% block-min MSE is partly a min-over-more-windows selection artifact; the within-block trajectory converges by ~2.5-3k steps then OSCILLATES (e4: 0.087<->0.132 band, no trend — NOT instability, batch-window variance; my earlier "+54% late-block degradation" claim retracted on the full trajectory, and the TF-IDF self-pollution mechanism I proposed for it is disconfirmed in code: DF accumulates per batch but IDF is frozen within a block, recomputed only at boundaries). Endpoint function: -12% vs stageB == the other arms; inits unchanged. Retro-explains E30: 3k->5k helped joint because joint blocks hadn't converged at 3k; staged blocks converge by ~3k, so 5k is already past the knee. 5k->7k retired at the current write rule.

### FORGETTING: still solved in every arm (MSE matrices, paired-noise instrument rebuilt as scripts/vla_analysis/mse_matrix2.py — partial-load slot-swap, 12x less IO)

Just-trained->final drift: stageB +0.0-1.7% (E39), affine <=+0.6%, lr2x +0.4-2.2%, steps7k +1.2-1.7%. Stationarity holds at 2x LR, with biases, at 7k steps. Untrained-task cells reproduce across arms to 3 decimals (instrument validated). lr2x diagonals uniformly -13% vs stageB.

### The e6-after-e9-block drop is REAL — E39/E40's eval-noise attribution REVERSED

e6 drops across e9's block in **4/4 runs** (stageB -35, affine -10, lr2x -20, steps7k -30; pooled ~3sigma), e4 in 3/4. No other block shows systematic cross-task deltas. Mechanistic chain, each link measured:
1. **Exposure topology is arm-invariant** (frozen router): all arms share footprints, RTO 10-13%, bleed channels e6<-e9 = 4-6%/layer, e4<-e9 = 3-4% (vs e2's 2-3%) — e9 is the one anomalous writer (35.8k L14 slots written, 2x anyone; the warmed router gave it a 2.4x footprint; E39).
2. **The perturbation is tiny in function space**: e6-perceived field change across e9's block 1.36-1.62% (core50 0.13-0.16%); paired-MSE drift on e6's cell +0.5-1.3%.
3. **The rollout cost is 10-30pp** — a ~10-20x function->success amplification on MARGINAL tasks only (e2 at 80% absorbed identical drift free). Policies sit near the success boundary; 50-action open-loop chunks compound small field changes.
4. **Protection can't gate it (structurally)**: rank-mode (1-u)^beta only flips top-t membership — high-TF slots never rank out, and the diffuse tail sits at u~0.01-0.05 under peak normalization (u at core-50 boundary ~0.035; max/p99 ~10-16x) => beta4 vetoes only the top ~1% mega-hot slots (E39 note, now the load-bearing fact).
Displacement analysis (checkpoint diffs, L14): lr2x moves written slots ~1.6x (median ||d|| 1.33-1.51 vs affine 1.03-1.21, steps7k 0.87-1.08 — steps7k's median slot moved LESS: same mask budget spread over more batches + decayed-LR tail). The giveback (lr2x -11.8pp init->final) rides on this channel: e6 -31, e9's own decay -17, e4 -15 — all post-e9-block or late-block micro-drift.

### The conversion-gap probe program (5 instruments, chronological; all persisted to scripts/vla_analysis/, results in outputs/analysis/e41/)

The sharpest puzzle: steps7k and lr2x reach the SAME eval-time paired MSE (within 2%) with 20-ep inits 10-vs-35 (e4) and 10-vs-25 (e9). Working hypothesis #0 (displacement/amplitude -> "commitment") was formed from this; the program below tested it and its rivals.

**Probe A — downstream-gain (Josh's layer-position hypothesis: "layers 15-17 fix the corrections back toward pretrained behavior").** Inject at L14/L8: the learned memory delta vs matched-norm random vs matched-norm feature-direction perturbations; measure velocity-readout movement. RESULT: **disconfirmed in its strong form** — learned deltas transmit 2-3x BETTER than random (T_rand 1.9-3.2) and 1.2-1.8x better than feature directions, at both layers, every arm. Downstream is an amplifier tuned to the memory's directions, not a fixer. Surviving nuance: amplitude response saturates at L14 (2x delta -> +60-70% output; L8 +81-94%; layer-norm renormalization is the likely mechanism) — a headroom warning for LR pushes, not a case for moving layers. Also: total velocity-space throw of the memory is ARM-INVARIANT (~25 at L14) — lr2x does not deliver a bigger total correction, it delivers a better-placed one. Layer repositioning PARKED with evidence.

**Probe B — denoised-chunk (the integrated field).** Run the real 10-step denoise on demo obs; compare executed chunk to demo chunk. The training loss only ever queries the field ON the noise-demo interpolation bridge (x_t anchored to the true action — teacher-forcing one level down); integration queries the model's OWN trajectory, which leaves the bridge after step 1. RESULT: chunk error ranks arms **exactly as rollouts do, 9/9** (e4: lr2x 0.156 < steps7k 0.172 < stageB 0.191; e9: 0.366/0.401/0.459; e2: 0.224/0.272/0.304) where velocity-MSE couldn't separate lr2x from steps7k. Arm gaps grow on the last 10 chunk steps. **ADOPTED as the standard gate metric** (~10 min/checkpoint, no simulator).

**Probe C — bias decomposition (pre-registered: weak arms carry a systematic pull toward A-phase content; amplitude de-biases).** Signed velocity errors, K=6 paired draws/state, finite-K-corrected bias fraction + cosine to the pre-sequential (A-state) bias field. RESULT: **FALSIFIED.** Bias fractions uniform across arms (0.38-0.43); task-level bias ~0.03 everywhere; every arm shrinks the A-bias field to ~25-35% equally; and r2244 — the best converter — has the HIGHEST bias fraction (0.41-0.47) and STRONGEST A-pull (cos 0.49-0.55 vs staged 0.34-0.42).

**Probe D — off-bridge generalization (pre-registered: weak arms generalize worse off the training manifold).** Noise-shift trick: passing noise+sigma*xi queries the field at x_t + t*sigma*xi with the EXACT analytic target (u+sigma*xi), so L(sigma) measures off-bridge field quality with zero approximation. RESULT: **FALSIFIED.** Relative degradation at sigma=0.6 is 23-27% for all staged arms, no rollout-matching order (lr2x slightly worse if anything); r2244 modestly better (18-21%).

**Probe E — trajectory-error coherence (the remaining suspect after C+D: error ORGANIZATION along the model's own denoise path).** Step the real 10-step denoise manually; at the model's own x_t the demo-consistent velocity is analytic (v* = (x_t - a)/t); record e_k per step. RESULT — two findings:
1. **Family-wide: coherence 0.94-0.97 for EVERY model including joint r2244** (adjacent-step cos 0.91-0.95). The field's error along a trajectory is essentially one constant vector per (obs, seed); integration accumulates it ~1:1 — **no model in this family self-corrects during denoising.** Endpoint offsets ~0.25-0.3 normalized units/dim for everyone.
2. Arms are consistently but MODESTLY ordered (coherence stageB 0.946/0.958/0.949 > steps7k 0.940/0.950/0.940 > lr2x 0.937/0.947/0.937 on e4/e9/e2; endpoints stageB ~10% worse; per-step RMS tracks the on-bridge MSE gap).

### THE CORRECTION (the program's most important output)

Assembling all instruments: **within the staged family, function quality is consistent everywhere** — stageB < steps7k ~= lr2x by ~10% on every metric (on-bridge MSE, chunk error, coherence, endpoint, per-step RMS) — and this matches the only well-measured rollout numbers, the **50-ep finals (32.0 < 36.0 ~= 35.6)**. The dramatic "same loss, 3x rollouts" init contrasts were **20-ep cells (+/-11pp) on marginal policies**, and hypothesis #0 over-fit a mechanism to them. Amplitude is hereby DEMOTED from "the conversion mechanism" to "a real ~10% fit improvement plus favorable draws." Session tally: two falsified mechanisms (C, D), one demoted over-claim (#0), one disconfirmed architecture hypothesis (A). METHODOLOGY ADOPTED: 20-ep init cells retired from decision-making (50-ep or chunk-metric only); every future arm gets the probe battery before narrative.

### The 3-layer failure model (current best understanding)

- **Layer 1 — rollout-level interference (nailed, fix shipped):** later tasks' diffuse writes -> tiny value-field drift -> 10-30pp drops on marginal tasks via open-loop amplification; carried by the one diluted writer (e9); provably ungateable by rank-mode protection.
- **Layer 2 — the family fit ceiling (characterized, not closed):** every frozen-backbone-adapter model produces a velocity field whose error is COHERENT along its own denoise trajectory (~0.95); the endpoint inherits the per-step field error ~1:1 (~0.25-0.3 units). The one-step loss structurally under-weights exactly this component (it samples pointwise against noisy targets; the systematic part hides under the sampling floor — bias_frac ~0.40 of a small number). This is WHY loss and rollout success decouple. All tested levers (rank r4, +600M bias params, gate, steps, 2xLR) move it <=10% — they all act on value-path capacity, which is not where the offset mainly lives.
- **Layer 3 — staged-family vs base ceiling (open):** base joint finetune = 74.8% on these 5 tasks vs our 32-36; the offset is not task-intrinsic. Rank-2-mixture expressivity vs adaptation budget vs frozen backbone — undecomposed; the base-finetune control was deleted (E31), which elevates the **LoRA-FT baseline to the decisive experiment** (same frozen backbone, dense adapter, same budget, read through the probe battery: if its chunk error ~0.15 -> the tax is OUR sparse-mixture path; if ~ours -> the ceiling is frozen-backbone adaptation generally and the thesis claim scopes to "joint-adapter-level fit with zero forgetting").

### Soft (grad_scale) protection: shipped, smoked, calibrated (the Layer-1 fix)

- **Design**: TF-IDF top-1536 mask unchanged (locality); protection moves from the ranking score to the UPDATE: each surviving slot's applied update is multiplied by (1-u_q(s))^beta. **Implementation catch that would have silently no-op'd: Adam's step is invariant to a time-constant per-row gradient scale** (m-hat and sqrt(v-hat) scale together; smoke-measured movement ratio 0.995 under naive grad scaling). Correct mechanism = **post-optimizer-step blend** theta <- theta_pre + scale*(theta_post - theta_pre) on protected rows — exact per-slot LR scaling under any optimizer (smoke: measured 0.00389 vs theoretical 0.25^4 = 0.00391).
- **u-normalization fix**: `protect_u_norm=corefrac` — u = counts / count-at-core50-boundary, clipped to 1 (peak-norm's u~0.035-at-boundary degeneracy fixed; the whole prior core now protects).
- New config: `protect_mode` (rank=legacy default, byte-identical) / `protect_u_norm` (peak default). Smokes 33/33 (scripts/vla_analysis/smoke_softprotect.py) + affine regression 20/20.
- **beta calibration** (offline, real lr2x per-slot deltas x read-mass distributions, L14+L8; scripts/vla_analysis/calibrate_beta.py): smooth trade, no knee — e9-block bleed onto e4/e6 kept 70/60/48/35/24% at beta 1/2/4/8/16 for e9 static write-mass cost 7/12/18/26/35%. **beta=4 chosen**: halves the bleed; static cost overstates real cost (mask reallocates suppressed magnitude to prior-free slots; e9 — the writer that matters — is the CHEAPEST to constrain); midpoint is information-maximizing under unknown damage-response shape. Decision tree: e6 still drops at beta4 -> response is threshold-y -> beta 8-16; e7's init craters (later writers pay cumulatively more, ~28-38% static at beta4) -> beta 2.

### Batch-size smokes (Josh's hypothesis: bs32 under-covers the trajectory)

bs128 native: OOM (~146GB demand vs 139.8 usable). bs64 native: OOM (~140GB). bs32 training steady-state is ~125-131GB (the "37GB" VM3 reading was a mid-eval allocator-released snapshot). => effective-64 via `gradient_accumulation_steps=2` (the committed script; the trainer counts OPTIMIZER steps and merges retrieval indices across micro-batches, so the TF-IDF mask sees a true 64-frame TF). **The strongest mechanism for bs64 is mask stability, not gradient noise**: at bs32 the top-1536 is ranked from 32 frames' retrievals -> mask churns -> write budget rotates across slot subsets -> updates smear (the dilution pathology; steps7k made it WORSE — more draws, 37.7k unique slots). 64-frame TF -> stabler selection -> fewer slots, more events each -> counters dilution AND shrinks the bleed surface. This is the anti-dilution lever from the opposite direction to topt3072.

### In flight (all 5-task, stageB-verbatim + single delta, 50-ep finals)

| VM | run | delta | pre-registered reads |
|---|---|---|---|
| VM3 | topt3072 | tfidf_top_t 3072 | e9 init + chunk error (dilution-as-budget?); e6@20k cell (2x write breadth should WORSEN the bleed if the model is right — falsification opportunity) |
| VM2 | lr2x+softprotect (grad_scale/corefrac/beta4) | protection mechanism + 2e-3 | inits >=~45; e6-across-e9-block drop <=10pp (vs -10..-35 in 4/4); e9 init may pay a few pp; final >=42 = new frontier |
| VM1 | bs64accum2 | effective batch 64 | e9 L14 written-slot count (expect down from 35.8k) + ev/slot p50 up; e9 init + chunk error; e4/e6 bleed shrinkage |

All three land ~16 Jul; read via retention matrices + slot JSONs + the probe battery (chunk metric first), NOT 20-ep init cells.

### Next steps (after the 3 arms)
1. **LoRA-FT per-task baseline** (e4/e9/e2, ~8-17h) — now the decisive Layer-3 experiment; read through the probe battery, not just success.
2. **Seed-averaging micro-experiment** (offline, ~1h): endpoint error = shared bias + per-seed component (probe-B spread 0.09-0.25 is substantial); averaging denoised chunks over a few noise seeds may cancel the seed part at pure inference cost. Test with probe B before touching evals. Risk: mode-averaging on multimodal segments.
3. If softprotect works: 10-task extension of the winning config (the catastrophe-elimination demonstration, E39 #2).
4. Parked with evidence: layer repositioning (probe A), gate/bias variants (arm 1), steps>5k (arm 3), global top-t for retention (E19 + bleed mechanism), joint track (off-thesis, E40).

### Code / artifacts / bookkeeping
- Trainer: `protect_mode` / `protect_u_norm` / `_core50_boundary_count` / `_snapshot_protected_rows` / `_blend_protected_rows` in lerobot_sequential_train.py (defaults byte-identical; committed ba388ad1 with the E41 instruments + both arm scripts).
- Instruments PERSISTED to scripts/vla_analysis/ (mse_matrix2, probe_conversion [gain+chunk], probe_bias, probe_offbridge, probe_coherence, calibrate_beta, arms_{slots,displacement,wandb}, smoke_{softprotect,affine}, run_*.sh) — the E39 instruments died in a scratchpad; not again. Results: outputs/analysis/e41/*.jsonl.
- Job scripts (git add -f, dir gitignored): stageB_seq5_lr2x_softprotect.sh, stageB_seq5_bs64.sh (+ the E40 four from a8ac9fff/reconstructions).
- Eval-comparability note: stageB's final row is 20-ep; all E41-era finals are 50-ep.

---
## Entry 42 - 16 Jul 26 (E41 3-arm verdicts: bs64 EXONERATED-AND-RETIRED (no accum bug; chunk parity at 2x cost); softprotect = new 50-ep frontier 41.2 BUT the blend mechanism is DEFECTIVE (Adam momentum leak, ~90% passthrough) — the number is the lr2x family + eval fortune, not protection; topt3072 FALSIFIES the bleed->rollout model on schedule and identifies COVERAGE as the second fit lever. Fixes shipped (momentum-aware blend + hard veto, smoked); LoRA-FT baseline support built+smoked; next: lr2x+3072 / lr4xsched+3072 / LoRA-FT. 12 days to 70%)

### The three landed arms (stageB-verbatim + single delta; final rows 50-ep; baselines from E41)

| arm | e4 | e6 | e9 | e2 | e7 | mean init | final | give-back |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| stageB | 10 | 60 | 5 | 70 | 30 | 35.0 | 32.0 (20ep) | -3.0 |
| lr2x | 35 | 65 | 25 | 80 | 32 | 47.4 | 35.6 | -11.8 |
| steps7k | 10 | 55 | 10 | 70 | 32 | 35.4 | 36.0 | +0.6 |
| **bs64** | 10 | 40 | 0 | 60 | 26 | 27.2 | 30.4 | +3.2 |
| **softprotect** | 35 | 50 | 30 | 70 | 38 | 44.6 | **41.2** | -3.4 |
| **topt3072** | 15 | 35 | 5 | 85 | 34 | 34.8 | **37.6** | +2.8 |

Loss (wandb block-min/block-end mean): stageB .1274/.1410, lr2x .1132/.1239, bs64 .1176/.1355, softp .1139/.1270, **top3k .1086/.1215** (family-lowest). Probe battery run over ALL arms (chunk grid own-block AND final for t0-t3; slot autopsy L14+L8 with realized-write counts, displacement, self-coverage, full bleed matrix; softp MSE matrix): outputs/analysis/e42/.

### Arm 1 verdict — bs64: no gradient-accumulation bug; parity at 2x cost; RETIRED

- Full code audit CLEAN: Accelerate 1.13 scales loss under accumulation (averaged grads); mask/clip/step/zero + LR scheduler gated on sync_gradients; TF-IDF ranks the merged 64-frame indices; value-LR schedule = 5000 optimizer steps (wandb: 125 points, peak 9.82e-4); DF/IDF/protect stats scale-invariant. Config verified from checkpoint.
- "Lower loss, worse rollouts" dissolves: loss -8% is real (cleaner gradients); the FUNCTION (chunk) is at parity with stageB, slightly better (e4 .189 vs .191, e9 .448 vs .459, e2 .282 vs .304, e6 .115 vs .126); the init deficit was 20-ep cells; 50-ep finals ~= stageB per task.
- The E41 mask-stability mechanism fired EXACTLY as designed and bought nothing: written sets shrank 39-46% (e9 35.8k->21.9k L14), ev/slot up — but SELF-COVERAGE (task read mass on own-adapted slots) fell ~10pp/task (e9 67->56%). Concentration gain == coverage loss. At fixed top_t the two are the same budget spent differently; bs32's mask churn was a free coverage scan, not a pathology. Batch size was never the causal variable — retire the axis (and the "dilution" frame with it).

### Arm 2 verdict — softprotect: the mechanism did NOT bind (Adam momentum leak); the run is a second lr2x

- **THE DEFECT**: the post-step blend rescales a row only on steps the row is in the snapshot (mask ∩ scale<1). Adam keeps applying the row's momentum tail (~1/(1-beta1) steps) after it leaves the churning mask — unblended. Median mask slot is selected ~17-22 of 5000 steps => ~90% of movement leaks. PROOF from the run's own checkpoints: e9's block contains 79 mask slots with u=1.0 EXACTLY (blend scale exactly 0 — must be bitwise frozen); NONE are frozen, p50 |d| 1.55 vs lr2x's same-bin 1.75. Measured attenuation -11..-14% vs designed -59..-100%. The E41 smokes used a FIXED mask (every tail recaptured by the next step's blend) — structurally blind to churn. Second time Adam's statefulness defeated this mechanism (E41 caught grad-scale invariance pre-launch; the momentum tail is its sequel).
- Design itself validated: airtight (1-u)^4 applied offline to the run's real deltas keeps 48.1% of the e9->e6 bleed at L14 — matches the E41 calibration (48%) to 0.1%.
- What actually ran: grad_scale mode also removes the rank discount => softp ~= lr2x with PURE-TF ranking. Its bleed onto e6 was HIGHER than lr2x's (1.85% vs 1.62 full; core50 0.51 vs 0.16 — 3x, the discount used to deflect the very top slots).
- Twin comparison in function space (chunk, own-block -> final): e4 +1.6% vs +1.2%; e6 FLAT in both (.0968->.0965 / .0958->.0958); e9 lr2x .366->.382 (+4.4%) vs softp .344->.352 (+2.3%) — softp's small real edge (ranking purity made it a better writer; best e6/e9 own-block fits on record); e2 lr2x .224 vs softp .248 (-11%) — softp's e2 rollout deficit (70 vs 84) is REAL, the writer paid by task 4. Paired final-row diff +5.6 ± ~4.5 (p~0.2).
- **Attribution of 41.2**: the 2xLR family's true 50-ep level is ~36-41; softp drew the high end (small genuine e9/e6-fit edge, real e2 cost, remainder eval fortune). "Protection recovered r2244" is NOT demonstrated — the pre-registered e6 gate passes only via cells whose function shows no drop in either twin. lr2x's infamous -11.8 give-back is likewise mostly init-draw artifact: its function-space give-back is +0.5..+4.4%.
- softp MSE matrix: flat (+0.0..+2.2% diag drift) — the blend machinery didn't disturb stationarity.

### Arm 3 verdict — topt3072: real (+5.6 over stageB), mechanism = COVERAGE; and the pre-registered falsification LANDED

- Per-task footprints (JSONs, arm-invariant: 0.996 overlap across arms): table = 147,456 slots/layer; core-50 = 2.1-7.1k (1.4-4.8%); effnum 4.6-15.3k; core-90 17-41k. e9's warm-up dilution in slot units: core50 7,081 = 2.7x everyone (its footprint alone dwarfs any fixed budget).
- At top_t=1536, self-coverage was 62-86% (the un-adapted remainder = stale A-content in every retrieval mixture; the 1536 budget reaches ~the core-70-85 mass point). At 3072: 74-95% (+9-14pp/task) — and per-slot |d| UNCHANGED (p50 .88 vs .83) with ev/slot UP (26 vs 17): the bigger budget stops the mask ROTATING (holds core+shoulder continuously) rather than splitting gradient thinner. Family-lowest loss; give-backs flat-to-IMPROVED in function space (e4 -4.3%, e6 -1.8%, e9 -3.1%, e2 -0.1%).
- **Falsification**: writes ~1.8x broader, function-space bleed 2-3x LARGER everywhere (e9->e6 3.06% vs 1.3-1.9; e2->e9 6.0%!) — with ZERO rollout retention cost. Across 6 arms, bleed magnitude vs e6-drop correlates ~0.
- Caveats: inits FLAT (34.8 ~= stageB) — coverage converts to loss/finals, weakly to fresh inits (the Layer-2 conversion gap rules); e9 rollout unmoved at 80% coverage (amplitude-limited on every instrument, only LR moved it: 5 -> 25-30); one chunk misrank (top3k e2 own-block .287 near-worst vs its best-ever e2 rollouts 85/78 — the chunk metric resolves large gaps, not the mid-field).

### The e6-after-e9 story, revised (E41 partially corrected)

- Pooled post-e9 e6 (episode-weighted, all post cells): stageB 28.3, affine 41.1, lr2x 37.8, steps7k 37.8, bs64 38.9, softp 42.2, top3k 40.0 — ~38-42 in 6/7 arms. The dramatic per-arm drops were inflated by high 20-ep init draws (regression to the mean); a real average drop ~-15 remains in the four 1x-amplitude E40 arms (E41's pooled 3-sigma stands).
- **e6's FUNCTION on demo states is flat across e9's block in ALL SIX arms measured** (chunk own->final: -3%..+0.5%), at bleeds 0.8-3.1%. Whatever converts theta-drift into the residual rollout wobble is invisible on demo states and NOT proportional to bleed — most consistent with off-demo-distribution damage (closed-loop excursion states) + eval noise. softp-vs-lr2x (near-identical physics, 19pp apart on this statistic) bounds the per-arm noise at ~the effect size.
- **Model update**: at stageB-family exposure (RTO 10-13%, flat matrices, flat chunk retention), write-side interference management earns ~0 rollout points. E41's three arms optimized a solved margin. The bleed model is DEMOTED as a cross-arm decision tool; protection's value case moves to the 10-task extension (2x+ cumulative exposure) and bigger-write regimes.

### Fixes shipped (smoked, not launched)

1. **Momentum-aware blend**: `_blend_protected_rows(snap, optimizer)` scales the row's exp_avg by the same factor (kills the tail at source; exp_avg_sq untouched). New smoke S12 reproduces the leak (old blend keeps 0.79x movement under churn; designed 0.0625) and verifies the fix (0.0625 exactly; s=0 freezes bitwise).
2. **`protect_hard_u`** (both modes, default 0=off): slots with u >= threshold removed from top-t CANDIDACY (not just score-zeroed — with top_t >= candidates a zero-score slot still gets selected; smoke S13 caught this, veto made structural via counts masking). Never in mask => no grad, no momentum — airtight. At corefrac, u>=0.9 = the victims' true cores: ~19%/34% (L14/L8) of e9->e6 bleed mass on a few hundred mask slots.
3. Smokes: softprotect suite now 44 checks incl. S12a-e/S13 (ALL PASS); affine regression 20/20; boundary log now prints hard_u. NB the honest expectation even for the FIXED mechanism: per-slot LR scaling suppresses ENDPOINT displacement only where s x 5000 steps can't converge (u>=~0.4-0.5); the shoulder still converges slowly — the implementable ceiling is closer to the veto-mass curve (26% of bleed @ u*=0.5, L14) than the calibration's 48%.

### LoRA-FT per-task baseline: support built + smoked end-to-end

- Existing plumbing: `--peft.*` CLI -> `wrap_with_peft` -> pi05 default targets; `--dataset.episodes` filter; `--env.task_ids` single-task eval; PEFT save/load via factory `use_peft`.
- Missing pieces found+fixed: **peft not installed** (added 0.19.1); **`get_optim_params` on the wrapped model forwarded to the BASE policy => optimizer over 4.2B frozen params, 0 trainable — training would silently no-op**. Fixed in `wrap_with_peft` (returns trainable params). Smoked: attach (53.2M adapters: LM 39.2M + expert 13.9M + proj 0.1M at r=32, attn+MLP both towers, vision untouched), 4-step CLI train (num_learnable=53M/4B, loss+grads healthy, grad-ckpt composes, episodes filter live), checkpoint saves adapter_model.safetensors, lerobot-eval loads it and rolls out on env.task_ids=[4].

### Storage (was 94% full)

Deleted (analyses discharged; probes/matrices/autopsies persisted to outputs/analysis/e42): per-task intermediate checkpoints + ALL training_state for the six 5-task arms (finals + last + JSONs + evals + wandb kept); r2244-pretrain and frozenbase-40k intermediates (final weights kept); affine-A training_state; smoke dirs. **Freed ~1.1T -> disk 51%** (1.2T free). Untouched: stage-1 base, warmed router, stageB A, realworld_v2, all audits.

### Next (3 VMs; 12 days to 70% avg) — **UPDATE 16 Jul eve: ALL THREE ARMS RUNNING** (VM1+VM2 launched by Josh on the other boxes; VM3 = this box, chained behind the generalist-overlap audit via staged/vm3_audit_then_loraft.sh, tmux `vm3chain`, log outputs/vm3_chain.log)

| VM | script | config | pre-registered reads |
|---|---|---|---|
| 1 | staged/stageB_seq5_lr2x_topt3072.sh | value_lr 2e-3->2e-4 + top_t 3072 | inits >=~47 AND give-back >=~0 => final >=42-45 (50ep) = new frontier; e9 init >=20; FAIL: give-back <=-8 (levers interact) |
| 2 | staged/stageB_seq5_lr4xsched_topt3072.sh | value_lr 4e-3->2e-4 + top_t 3072 | e9 init/chunk vs VM1 (the amplitude-limited cell); block-ENDs vs VM1 (worse => L14 saturation binding, 2x = plateau, axis closed) |
| 3 | baselines/loraft_pertask_baseline.sh | 5x independent LoRA r32 on stage-1 base, 5k steps, 50-ep evals | e4/e9 ~35-50 => our machinery's conversion tax, priced; <=15 => frozen-backbone ceiling, scope the thesis claim; >=60 => rethink value path |

The 70% arithmetic, honestly: best staged 41.2; base-joint ceiling 74.8 on these 5. The compositions plausibly buy mid-40s. The remaining ~25pp is Layer 2/3 — which is exactly what VM3 prices. If LoRA-FT lands high, the known gap-closers in-protocol are r4-at-L12/14 on the staged track (never run; r2244's +6 came from it in the joint era), the top-p (mass-quantile) mask (e9's 2.4x footprint needs an adaptive budget), and inference-side seed-averaging (E41 #2, still unrun). If LoRA-FT lands low, 70% is not reachable by ANY frozen-backbone adapter method on this suite and the target needs renegotiating toward "joint-adapter-level fit with zero forgetting". Decision point lands with VM3 (~14h).

### Artifacts
- Code: lerobot_sequential_train.py (blend+veto fix), pretrained.py (PEFT get_optim_params), smoke_softprotect.py (S12/S13).
- Instruments: scripts/vla_analysis/{run_e42.sh,run_e42b.sh,e42_slots.py}; results outputs/analysis/e42/{probe_conversion.jsonl,mse_matrix_arms.jsonl,slots_summary.json,slots.out}.
- Scripts: staged/stageB_seq5_{lr2x_topt3072,lr4xsched_topt3072}.sh, baselines/loraft_pertask_baseline.sh (git add -f).
- Eval-comparability: all E42 finals 50-ep; init cells remain 20-ep (retired from decisions per E41; the probe battery is the ranking instrument, with the e2-misrank caveat above).

---
### Entry 42 addendum (16 Jul eve, discussion) — the loss-vs-rollout root cause narrowed to OFF-TRAIL behavior; the grounding-gradient family resolved; "freeze the generalist slots" queued behind an overlap measurement

Long discussion with Josh on THE standing question (why sequential loss == multitask loss does not give sequential perf == multitask perf). Distilled outcomes:

1. **Taxonomy + status.** Candidate mechanisms enumerated ground-up: (a) noise-floor masking [confirmed; the chunk probe is the workaround], (b) error accumulation along the denoise path [measured family-wide ~0.95 coherence incl. r2244 — shared, not the differentiator; NB never measured on the true full-finetune baseline, which was deleted at E31], (c) error placement within the chunk [late-chunk skew known; full per-position/per-dim profile still to run], (d) OFF-DEMO-TRAIL degradation [now the PRIME suspect: E42 showed e6's function flat on demo states while rollouts wobble — the damage lives at states only the closed loop visits; never directly measured], (e) frozen-feature extrapolation margins [E36 crowding, circumstantial], (f) sample/seed variance [spread measured substantial; seed-averaging still unrun], (g) the success-threshold cliff [shared amplifier, closed as a cause — and bs64's null doesn't bear on it: flat-minima batch lore is parameter-space, the cliff is task-metric-space].
2. **Grounding-gradient proposal (Josh) resolved after full derivation.** Any precomputed/cached gradient fails three ways: gradients at frozen params feed nothing (backprop chains through activation signals; weight grads are leaves); a dataset-averaged (signal x activation) product cannot be re-decomposed into state-conditional signal at new inputs; and at the anchor's own minimum the averaged gradient is ~0 by first-order optimality (gradient = slope, not height — the A-phase values are CONVERGED on libero-90, so the "generalist direction" is zero where we'd harvest it). The valid family members: live rehearsal (= averaging losses; targets retention, which is solved) and the curvature version (= EWC = our protection machinery aimed at the A-phase — same non-problem). The surviving state-conditioned carrier of "generalist opinion at new states" is the base FUNCTION: distillation anchor L = L_task + lambda*||f_adapted(x) - f_base(x)||^2 on new-task states (cache the function, not the gradient). Caveat: the anchor is task-incompetent (stage-1 floors ~0-2), so lambda trades fit for smoothness.
3. **Remedy shortlist for the off-trail story (all values-only, in-constraint), gated behind the off-trail instrument** (score executed-chunk error on observations harvested from our own rollout episodes; on-trail vs off-trail degradation per arm, staged vs r2244): demo-jitter augmentation (same targets, perturbed observations — local flatness, no labels needed), lambda-anchor (above), and:
4. **"Freeze the generalist slots" (Josh, from the original memory-layers paper: accumulate pretraining access stats, let CL touch only low-usage slots).** In our stack this is the E42 protection machinery with the store seeded from A-phase (libero-90) usage + protect_hard_u as the freeze (mass-threshold, not slot-count — corefrac handles it). As RETENTION machinery it targets non-problems (pretrain forgetting out-of-scope by E19 constraint; sequential forgetting solved). As FIT machinery the precedents are negative (E9 pretrain-IDF seeding tanked; E19 veto starvation; E42 softprotect's writers IMPROVED by writing into hot slots; coverage won by widening). The interesting reading: a STRUCTURAL distillation anchor — frozen generalist slots pin the shared ~23-44% of every retrieval mixture at the general function, so off-trail routing drift falls back toward the general policy instead of overwritten task-specific content. Zero hyperparameters beyond the mass threshold. If adopted, the claim is "reserved generalist substrate improves off-distribution fallback," NOT the paper's protect-pretraining story.
5. **RUNNING NOW (this box, tmux vm3chain): the gating measurement** — the A-phase never dumped memory_usage.json, but under frozen-base routing reads are VALUE-INDEPENDENT, so an inert-LR sweep of libero-90 through the A-checkpoint reproduces the A-phase read profile exactly (audit_libero90_usage_rwarmupB_A, 90 tasks x 40 batches, ~1.5h) -> scripts/vla_analysis/generalist_overlap.py computes each sequential task's read/update-event mass on the top-{50,20,10,5}%-mass A-phase sets (outputs/analysis/e42/generalist_overlap.json). High write overlap => E19 starvation regime, freeze is costly; low => freeze ~free (and its residual case is the structural-anchor story). LoRA-FT baseline chains automatically after. AWAITING RESULTS.
6. Also noted: VLM-side memory layers (Josh) are viable with a stationarity-safe placement — prefix layers ABOVE the highest expert-memory layer (>14) never enter the prefix caches that expert routing at 8-14 consumes (per-layer KV pairing) — parked as a compute-gated direction pending VM3's read on whether upstream trainable capacity is where the wins are; and inference-side re-planning (execute 25 of 50) noted as legitimate-if-applied-to-all-models, gated behind the per-position error profile.

**Addendum results (17 Jul, 00:30) — generalist-slot overlap MEASURED: the freeze is zero-sum-shaped; demoted.** Gotcha first: this libero_90 build has 73 task_index entries (0-72), not 90 — the audit crashed at "task 73" after sweeping everything that exists (data complete; scripts fixed to 0-72), but the crash idled the box ~7h before the LoRA baseline started (relaunched 00:05, tmux loraft; lands mid-afternoon). Results (outputs/analysis/e42/generalist_overlap.json): A-phase aggregate effnum 58-64k/layer; top-50%-mass core = 22-25k slots (15-17% of table), top-20% ~5.5k, top-10% ~2k. THE PATTERN: sequential tasks' read-overlap ~= write-overlap at EVERY mass threshold (top-50%: reads 26-42% vs writes 23-52%, e4@L14 51.8%(!); top-10%: both 3-8%) — the new tasks write where they read, proportionally at every depth of the A-core. No threshold exists with high read-dependence + low write-demand, i.e. no cheap-protection pocket even under the warmed router: a freeze deep enough to preserve meaningful generalist substrate blocks an equal share of write demand (E19 starvation shape, re-measured), and a free freeze (top-5-10%) protects a near-inert 3-8% of the mixture. VERDICT (revised after discussion, see below): the DEEP freeze (top-50%) is dead — E19 starvation shape re-measured; the SHALLOW freeze (top-10-20% mass, 2-6k slots) is a live bake-off candidate on equal standing with demo-jitter and the lambda-anchor. Side note: e4 (weakest rollout task) is the heaviest A-core writer (up to 52%), e9 the most private (23-27%) — consistent with e4's function living on contested generalist substrate.

**Verdict revision (17 Jul, discussion with Josh — the first "demoted" call was overzealous).** The zero-sum reading equated overlap mass with cost and benefit, but neither conversion is linear, and they bend in opposite directions:
1. **Cost is sub-linear in write-event overlap.** The hard veto removes frozen slots from top-t CANDIDACY (the S13 fix), so the 1536 budget redistributes to the next-ranked slots automatically — nothing is wasted and the adapted footprint keeps its size, just relocated. Reads are untouched (frozen slots keep serving their preserved A-content into every mixture — the point of the freeze). The residual cost is only that corrections at the 3-8% of read mass on frozen slots must be expressed through the other ~92-97% of each state's 144-slot mixture — mostly compensable (the beta-calibration's static-overstates-real result; LN renormalization). Estimated real fit cost of a top-10-20% freeze: ~1-3%, not the naive 3-16%.
2. **Benefit is plausibly CONCENTRATED where the freeze is cheap.** The structural-anchor value rides on off-trail routing drift landing on the hottest, most scene-generic slots — i.e. exactly the top-~2k, not the top-25k. Overlap mass is not the currency the benefit is paid in, so the 1:1 table cannot rule the shallow freeze out.
3. **It is the CHEAPEST bake-off arm to implement** (protect_hard_u shipped+smoked; the A-phase usage profile now on disk; missing piece = a ~20-line seed-the-store-from-stats hook), with no hyperparameters beyond the mass threshold — the opposite of the demotion rationale.
Honest cautions that stand: the benefit is contingent on the off-trail mechanism (instrument still to run — the gate for the whole remedy family), and the preserved fallback is a policy with 0-2% floors on these tasks (the bet is coherent-generic motion off-trail beats corrupted-mixture motion, keeping states recoverable — plausible, unproven). What the table genuinely killed: the deep freeze, and the paper's protect-most-of-the-substrate framing (our router sends new tasks INTO the pretraining core; theirs evidently didn't).

---
## Entry 43 - 17 Jul 26 (Composition arms: levers COMPOSE in function space (block-min 0.0880, project-lowest; 4x does NOT saturate) but 50-ep finals stay in the 40+/-2 band — the conversion slope measured across a 31% loss range (~+2pp per −10% loss, far too shallow for 70). FIRST real function-space give-back ladder (e9 own->final: 1x −3% / 2x +7.7% / 4x +14.5%, pre-registered predictions hit). LoRA-FT partials REPRICE the gap: e4=58 (above base-joint 52) with clean chunk 0.018 vs our 0.153 (9x) — the e4 deficit is substantially FIT, not pure conversion; e6=44 (BELOW staged 54-56). Freeze mechanism shipped (protect_seed_path); softprotect-fixed + freeze5k arms launched on the VMs; jitter probe built+smoked; LoRA compass (expert-only vs VLM-only) queued)

### The two composition arms (stageB-verbatim + deltas; init cells 20-ep, finals 50-ep; seed 1000; configs verified from checkpoints)

| arm | e4 | e6 | e9 | e2 | e7 | mean init | final |
|---|---|---|---|---|---|---:|---:|
| VM1 lr2x+topt3072 | 20->[32] | 55->[56] | 10->[14] | 70->[64] | [36] | 38.2 | **40.4** |
| VM2 lr4xsched+topt3072 | 30->[20] | 60->40->[54] | 35->[14] | 100->[80] | [44] | 53.8 | **42.4** |

Loss (block-min/block-END means): VM1 **0.0969/0.1077**, VM2 **0.0880/0.1019** — the two lowest in project history (top3k 0.1086, stageB 0.1274). Grad norms clean at 4e-3 (max 0.020); schedules honored (peaks 1.96e-3/3.92e-3). e7's loss response at 4x is the outlier: 0.1167 -> 0.0891 (−24%).

**Pre-registered gates scored:** VM1 "inits>=47 AND give-back>=0 -> final>=42-45": inits FAILED (38.2), give-back passed (+2.2), final missed (40.4) — neither the frontier nor the FAIL branch fired; the arm is a wash vs softp/lr2x. VM2 "block-ENDs worse => saturation binding": block-ENDs were BETTER — **the amplitude axis did not saturate in function space.** e9 init >=20: VM1 failed (10), VM2 passed (35) then gave it back to 14.

### What the family now measures: the conversion slope

Six sibling arms span a 31% loss range (0.127->0.088); their 50-ep final averages sit in [37.6, 42.4] (per-cell se +/-5-7 => family mean se ~+/-3 — all within ~1.5 sigma of each other). Across the eight staged arms loss rank DOES weakly predict rollout rank (Spearman ~0.6-0.7; softp the positive outlier, steps7k negative) at ~+2pp per −10% loss. At that slope, +28pp to 70% needs another ~90% loss reduction — arithmetically impossible (e9's loss floor alone is 2.4x e6's). The conversion slope exists and cannot be ridden to the target.

**Narrative self-check (E41 discipline applied in real time):** "e2/e7 converted amplitude to rollouts" survived until checking top3k's own e2 = 85 at 1x LR — family e2 finals 85/70/64/80 across near-identical physics = mid-field wobble, not an amplitude effect. Only e7 shows a monotone loss-backed response (finals 34/38/36/44 with the −24% loss jump), itself ~1sigma. 50-ep cells have a smaller noise radius than 20-ep cells, not a zero one.

**Give-back numbers are init-draw artifacts, again:** VM1 +2.2 vs VM2 −11.4 at near-identical write physics (bleeds within ~7% relative). VM2 drew high init cells (e2=100 at 20 eps = a ~4% draw of a true ~0.83; e9=35; e6=60). The "inits>=47" arm of my own pre-registration was built on the retired instrument — pre-registration design error, owned.

### Slot autopsy: the cleanest lever isolation the project has produced

Write topology is ARM-INVARIANT across the three 3072 arms (identical masks to the slot: t2 = 58,455-58,483 at L14; identical ev/slot, self-coverage 80.6% on e9 in all three) — frozen-route + same seed doing its job. The ONLY variable is per-slot endpoint displacement: L14 d_p50 = 0.88 / 1.38 / 2.05 at 1x/2x/4x => **displacement ~ LR^0.57** (sublinear — blocks converge, endpoint is landscape-limited — but nowhere near capped). Probe A reproduces the instantaneous L14 saturation (2x injected delta transmits 0.62-0.64x) on both new arms — training optimizes THROUGH the saturating layer norm; saturation caps the naive lever, not the optimizer. Total velocity-space throw stays ~25 (arm-invariant): higher LR re-places the correction, not enlarges it. At 4x, T_feat falls to 1.02-1.03 (vs 1.16-1.18 at 2x) — the extra correction is increasingly generic in direction.

Bleed grows sub-linearly with displacement (L14 full-mass): e2blk->e9 6.00/7.02/7.55%, e7blk->e9 6.60/7.90/8.43%, e7blk->e4 5.20/6.24/6.45%, e9blk->e6 3.06/3.64/4.06%. Cumulative absorbed field change: e9 ~15-16%, e6 ~12-13%, e4 ~10% (L8 e4 ~15%).

### THE new fact: a real function-space give-back ladder (chunk probe, e9 own-block -> final)

| arm | own (015k) | final (025k) | give-back |
|---|---:|---:|---:|
| top3k (1x) | 0.4067 | 0.3942 | −3.1% |
| VM1 (2x) | 0.3037 | 0.3271 | **+7.7%** |
| VM2 (4x) | **0.2792** | 0.3197 | **+14.5%** |

Both VM2 cells hit their pre-registered predictions (own 0.287-0.295 -> 0.2792; final 0.31-0.32 -> 0.3197). E42's "bleed doesn't convert" was scoped to <=6% channels at <=2x amplitude; at 2-4x + 3072 it converts, lawfully. VM2's endpoint is STILL family-best — the 4x fit edge survives its own bleed — and still rolls out at 14. **Chunk-rollout misrank #2 on e9:** softp chunk 0.3516 -> rollout 30; VM1 chunk 0.3271 -> rollout 14 (~2sigma). Our best function instrument cannot rank e9's mid-field; both misranks (e2 E42, e9 now) sit on marginal tasks where success rides states demos never visit. e9 seed-spread 0.21-0.23 ~ 2/3 of its chunk error — dwarfs every arm difference.

### LoRA-FT baseline partials (r32, attn+MLP on ALL 18 VLM-LM blocks AND all 18 expert blocks + action/state projections; vision untouched; 53.2M/task; per-task specialists, 5k steps, 50 eps)

- **t0/e4 = 58.0** — ABOVE base-joint e4 (52), 1.7-2.9x every staged e4 ever (best softp 35). Frozen backbone exonerated on e4; per the pre-registered rule this sits at the "~>=60 => rethink value path" boundary.
- **t1/e6 = 44.0** — BELOW staged e6 (54-56). The dense-adapter advantage is task-shaped: biggest exactly on the dual-cycle integration task.
- **Jitter first-read (MINI): LoRA e4 clean chunk 0.018 vs best staged e4 ~0.153 — 9x better on-demo function.** Its perturbed errors (state@0.1 0.048, image@0.05 0.146) stay below our CLEAN error. If it holds at full size, the e4 gap is substantially FIT (the achievable function level is far beyond ours), not pure conversion — a real reframe of the Layer-2/3 split: chunk~0.15 may simply be insufficient for >50% on e4, and our "conversion gap" partly = "everyone at our fit level converts like this".
- Params accounting for the machinery-tax discussion: our per-slot 4.1k params (r2 on the 1024-dim expert hidden). Per optimizer step the 3072x4-layer mask = ~50M eligible params (parity with LoRA's 53M); per task realized ~130-190k slots = 0.6-0.8B touched (>10x LoRA); per token per forward <=144 slots x 4 layers = **2.4M active vs LoRA's 53M in every token** (~22x density gap). We are not budget-starved; we differ in per-forward density, placement breadth (4 expert-MLP sites vs 36 blocks incl. attention), and perception adaptation (LoRA moves the prefix; ours frozen by construction).
- e9 cell lands mid-afternoon (watcher armed); e2/e7 tonight; full table ~02:00.

### Discussion outcomes (Josh, 17 Jul) — the plan

1. **Off-trail instrument, jitter version BUILT** (`scripts/vla_analysis/probe_jitter.py`, smoked both model classes): perturb raw demo obs (state sigma x per-dim std; image pixel noise; RNG seeded per task/batch/scale so all models see identical inputs), score the 10-step-denoised chunk vs the demo chunk; the READ is across-model degradation slopes at matched clean error (shared target bias cancels). Full version (score on rollout-visited states; target-free drift ||f_after − f_own_block|| on- vs off-trail) needs an obs-dump in the eval loop — deferred pending jitter results. Grid running: staged finals (VM1/VM2/softp/top3k/stageB) x {e4,e9} + LoRA t0; r2244 skipped (10-task exposure = protocol mismatch).
2. **Softprotect retest LAUNCHED (VM A)**: `stageB_seq5_lr4xsched_topt3072_softprotect_fixed.sh` — single delta vs VM2 = momentum-FIXED grad_scale blend + corefrac. Pre-registered: e9 own ~0.28 held, own->final <=+4% (vs +14.5%), e9/e4 finals up; FAIL = writer block-min MSE +10%. Framing: mechanism validation + de-taxing the LR axis (bank 4x fit for whenever conversion improves), not a points bet.
3. **Generalist freeze LAUNCHED (VM B)**: `stageB_seq5_lr4xsched_topt3072_freeze5k.sh` — top-5000 A-phase read-mass slots/layer (18-19% of A-phase mass; committed artifact `scripts/vla_analysis/data/a_phase_top5k_slots.json`) seeded u=1.0 + protect_hard_u=0.9 structural veto; single conceptual delta vs VM2. **Correction from discussion (Josh): my "freeze buys no fit so it can't push the frontier" was the fit=perf error this project exists to kill.** The mechanism: hottest A-slots = the generalist transforms in every mixture; sequential writes erode them; on-demo MSE never registers (own-task adaptation compensates on-trail); off-trail the eroded generalists are the fallback => rollout cost invisible in loss. Freeze removes the erosion; expected signature = flat MSE, better rollouts, shallower jitter slope. Pre-registered reads in the script header.
4. **LoRA compass QUEUED** (`baselines/loraft_compass_e4.sh`, auto-launch watcher armed for LoRA-table completion ~02:00): two e4-only arms — expert-only (attn+MLP+proj) vs VLM-only (attn+MLP+proj) vs the full-LoRA 58 anchor. A~58/B-low => n256/r4 staged build, VLM memory dead; A-low/B~58 => VLM-side memory is the build (placement chosen by running the E36 feature probe on PREFIX layers, not by guess — and low placement is allowed if the dual-path trick extends to the prefix KV, one extra prefix forward/batch); both mid => expert first; both high => cheapest wins. C (expert-MLP-only, attention ablation) dropped: we are not building attention-path memory regardless (parked; per-token routing makes it feasible in principle — each token routes itself, no seq-level router — but k/v corrections make interference cross-token), so its content only prices a caveat the 1-day n256/r4 arm tests directly.
5. **n256/r4 dose fixed**: n_keys 384->256 (65.5k slots, ~0.9x current param budget) x r=4, NOT n/4 (36.9k would dip below the A-phase's 58-64k effective slot usage => forced collisions at pretrain fill). Staged rebuild = warm-up ~3h + A ~3h + seq ~14h ~ 1 day/arm. Note: values-only stage-2/3 could likely afford r4-on-all-4-layers at the CURRENT bank too (the config joint training OOM'd on).
6. **Rejected**: seed-averaged inference (Josh: a hack, from other papers, lifts baselines too). Attention memory parked. r2244 as jitter anchor skipped.

### Code/artifacts shipped (commits 75f69f1c, c0facc6f)
- `--protect_seed_path` (trainer): seeds the prior-usefulness store u=1.0 from {module_key: [slots]} JSON before the task loop; with hard_u => structural candidacy veto for the whole run; validation requires protection on + file exists; '' = legacy byte-identical. Smoke S14a-e (suite 55/55; affine 20/20).
- `scripts/vla_analysis/data/a_phase_top5k_slots.json` (committed so clone VMs need no audit rsync).
- `scripts/vla_analysis/probe_jitter.py` + the E43 analysis artifacts: `outputs/analysis/e43/{slots_summary.json,slots.out,probe_conversion.jsonl,probe_jitter.jsonl}`, scratchpad e43 scripts mirrored in the session dir.

### ETA board (from 17 Jul ~12:00 UTC)
| when | what |
|---|---|
| ~15:40 | LoRA t2/e9 — the Layer-3 decision cell (watcher armed) |
| ~16:00 | jitter grid (staged x {e4,e9} + LoRA t0) |
| ~02:00 | full LoRA table -> compass auto-queue fires |
| ~03:30 | VM A (softprotect-fixed) + VM B (freeze5k) land -> chunk/jitter probes on landing |
| ~07:15 | compass A (expert-only) |
| ~12:30 (18th) | compass B (VLM-only) -> capacity-build decision (n256/r4 vs VLM memory) |

### Entry 43 addendum (17 Jul, discussion) — two working heuristics from the LoRA cells, and the regime filter on old counter-evidence

**Heuristic 1 (breadth law, Josh): perf ∝ (1/MSE) · n — read as an iso-performance surface, not a product.** Anchors: single-task LoRA rolls 44-58 at chunk ~0.02 (n=1); base-joint rolls 72.6 at train-MSE ~0.10-0.15 (n=100). The MSE required for a given rollout level RISES with the breadth of what the model trained on (two-point fit ~ MSE_req ∝ n^0.4 — decoration; the direction is the content). Mechanism: n proxies COVERAGE of rollout-visited states — a narrow model reaches 0.02 by interpolating its 36-episode manifold and is unconstrained one excursion away; a broad model at 0.12 is anchored by neighboring competence everywhere the rollout wanders. Correct functional form is a THRESHOLD, not proportional: success ≈ steep function of (threshold(task, support) − chunk), threshold shifted UP by support breadth. This single form retro-explains the E43 table: e4 (our 0.13-0.19 far above the n~substrate threshold => 20-35; LoRA meets its n=1 threshold => 58), the e6 sign-flip (our 0.10 meets OUR breadth-adjusted threshold => 54-56; LoRA's 0.02 can't fully buy out n=1 => 44 — the cell no demo-state instrument could explain), and the shallow conversion slope (we move loss 5-20% at a time, far above threshold, where the curve is flat; LoRA moved it 6-9x, through it). Jitter grid corroborates the support half: staged arms are NOT more brittle (absolute degradation LoRA >= staged at matched perturbations; the specialist wins by LEVEL, gap compressing 4.4x -> 1.5x as perturbation grows toward where rollouts live).

**Regime filter on the "lower MSE can't save us" corpus (Josh's point — prior findings are largely from other regimes):** E15 is SmolVLA-era; E19-t5/E30/E33 are joint-era AND pre-stationarity (drift tax on all their lever ROIs). Only E39-43 measure the current staged protocol. After filtering + the LoRA baseline: **E33's "the frozen backbone is the binding constraint" and E41's "value-path capacity is not where the offset mainly lives" are both FALSIFIED** (LoRA hits chunk 0.02 / 58% on e4 on the SAME frozen backbone via pure adapter capacity — the r4/bias levers supplied capacity in the wrong currency, and we blamed the backbone). Bin-1 of the old corpus was instrument error (one-step velocity-MSE blind to the integrated field — fixed by the chunk probe, E41); Bin-2 was small-moves-far-above-threshold (consistent with the law); the genuine residue is e9 (chunk −39% -> rollout pinned; decided by today's LoRA e9 cell: low-chunk+high-rollout confirms the threshold story, low-chunk+low-rollout breaks it) and E39's staged-vs-joint e4 pair at matched velocity-MSE (never re-measured at chunk level; may dissolve like lr2x-vs-steps7k).

**Heuristic 2 (read-write alignment): usable adaptation ≈ Σ over parameters of (accumulated learning signal) × (per-forward read participation).** Learning on barely-read params is wasted (dilution); reading never-trained params is wasted (stale A-content in the mixture). Both halves are already measured: read side = self-coverage (74-95% at 3072); learning side = update events/slot (p50 17-34, p90 740-1850). LoRA's product is degenerate-perfect (every param read every forward AND updated all 5000 steps); our CORE slots are in its ballpark (~1-2k events, read constantly) — **the deficit localizes in the mixture TAIL: ~20-50% of every retrieval mixture is A-content with zero sequential learning or shoulder slots with ~20 events.** E42's bs64 result ("concentration gain == coverage loss") was this product being conserved at fixed bank size — the trade is forced at n=384^2. **This is the sharpest argument for the n256/r4 arm**: shrinking the bank 2.25x raises BOTH factors instead of trading them (same knn x heads read budget on fewer slots => each read slot read more often; footprints shrink toward the mask budget => selfcov up; each masked slot collects ~2x the events), with r4 doubling per-slot expressivity. First bank-shape change that moves the product rather than sliding along it. Bounds ("within reason"): (i) over-concentration rebuilds the E21 per-task-bias pathology — q_intra + per-batch effnum stay gated; (ii) concentration serves the PRECISION axis only; heuristic 1 says the threshold half is won by NOT overwriting the generalist remainder — so the package is "concentrate learning+reads on the task subset, freeze/preserve the substrate" = n256/r4 + freeze, jointly.

**New standard for the battery:** compute the read-mass-weighted update-event distribution per task per arm from the autopsy JSONs ("the product") — pre-registered n256/r4 read: product up >=2x at flat q_intra; chunk down on e4/e9; rollouts follow where the threshold allows.

**Addendum 2 (17 Jul, ~16:00) — LoRA e9 lands: 70.0 @ 50 eps, chunk 0.0675. The registered prediction resolves FOR the threshold law (3/3).** e9's specialist beats base-joint (60) at 5x our best staged cell (14; softp 30), with function 4-5x below ours — low-chunk+high-rollout, the confirming branch. The e9 residue dissolves: our 0.28 was 4x the ~0.07 requirement; every within-family improvement moved us between points far above threshold (the flat region), which is why nothing ever converted on e9. Both decision cells now at/above the pre-registered ">=60 => rethink value path" boundary (e4 58, e6 44, e9 70; 3-cell avg 57.3 vs our ~30-35). Cross-task note: LoRA e6 (0.020 -> 44) vs e9 (0.068 -> 70) re-confirms thresholds are task-specific — chunk comparisons stay within-task only. One more jitter fact: e9's specialist is EXTREMELY image-brittle (image@0.05: 0.068 -> 0.452, +0.38 absolute vs +0.10 on e4/e6 — falls INTO our clean range) yet rolls 70 — the eval's visual variation evidently stays inside its competence; another hint that the perturbation shell and the rollout distribution are different objects (true off-trail instrument still the eventual arbiter).

---
## Entry 44 - 18 Jul 26 (Overnight E44 program: COMPASS answers the value-path question — expert-only dense LoRA FAILS e4 (14 @ chunk 0.229) while VLM-only nearly reproduces full (40 @ 0.030) => perception adaptation carries ~10x of the fit; span attribution finds the leverage CONCENTRATED in the language field (instruction + pi05's STATE-AS-TEXT tokens: 36-57% of dL/dh at 20x per-position density, L15 peak) with OPEN routing geometry (inter-task cos 0.73-0.86 vs 0.96+ image); softprotect-FIXED passes its gate (e9 give-back +14.5% -> +3.9%, victim cores 0.00% bleed) but taxes the last writer (+49% block-min); freeze5k mechanically perfect (0/5000 slots moved) and strategically null at 5 tasks (give-back WORSE +19.2%); multitask LoRA confirms the threshold law's e6 flip (52 vs specialist 44) and kills its iso-budget strong form. => THE BUILD: text-span VLM memory at LM [15,16], uniform n256/r2 across all 6 layers (expert right-sized to its measured 58-64k effective usage), VLM routing sweep LAUNCHED)

### Overnight results (all pre-registered reads scored; full battery in outputs/analysis/e43/e44_running_notes.md)

**Compass (e4, 50-ep + chunk):** full LoRA 53.2M -> chunk 0.020 / roll 58; VLM-only 39.2M -> 0.030 / 40; expert-only 13.9M -> 0.229 / 14 (train loss plateau ~0.18 = ceiling, not undertraining); staged memory (expert side) 0.118-0.19 / 20-35. VERDICT: the e4 fit requirement is UNMEETABLE from the expert side at any density — placement, not density/rank, is the currency (the "22x per-forward gap" framing dies: expert-dense has 6x our per-forward params and does worse than our sparse memory at matched placement, A-phase-substrate caveat noted). VLM necessary and nearly sufficient; the expert tower adds 0.030->0.020 + 18pp (e4's threshold sits ~0.02-0.03 — steepest part of the curve). Threshold law 4/4 then 5/5 (VLM-only high-chunk... low-chunk/high-roll branch consistent).

**Span attribution (new instrument, scripts/vla_analysis/probe_span_attribution.py, on the VLM-only LoRA):** pad control exactly 0.00% at every layer (instrument clean); Taylor term ~0 at displacement 0.35-0.94 (beyond first order — expected). Gradient share of the action loss by prefix span: image (~770 pos) 64->43% falling with depth; instruction (~24 pos) 13->8%; **STATE-AS-TEXT (~30-40 pos) 22->49% RISING with depth — pi05 tokenizes proprioception into the prompt (`Task: ..., State: ...;` — processor_pi05.py:83), and at L15 those positions carry HALF the loss sensitivity at ~20x the per-position density of image tokens.** The language FIELD (instruction+state) = 36-57% of leverage in ~60 contiguous positions. The adapter moved text-position hiddens as much as image ones (disp parity).

**Placement probe (feat_probe_vlm.py, corrected text-span pooling after a padding artifact):** linear task probe 100% at every LM layer 2-16 (98.8-99.3 at 17); all-token (image-dominated) inter-task cos 0.962(L12)-0.995(L17) — crowded, avoid 16/17; **text-span positions: inter-task cos 0.73-0.86, the most open task geometry ever measured in the project, basket family 0.86-0.95 (vs 0.99+ image / 0.978-0.994 expert-side) — the instruction stream separates the lookalike pairs the scene cannot.** Image side: discriminative-but-crowded (task signal spatially sparse in object patches; scenes genuinely ambiguous on lookalikes — the E28 floor at representation level).

**Protection arms (both single-delta twins of VM2=4x+3072 @ 42.4):**
- softprotect-FIXED (grad_scale/corefrac/beta4): **mechanism works in production** — u-bin displacement exactly inside the (1-u)^4 envelope (u>=0.9 slots 0.0000; the E42 leak's same bin moved 1.55), victim-core bleeds 0.00% everywhere, e2blk->e9 full bleed 7.55->1.54%. **e9 chunk gate PASS: own 0.2665 (best-ever) -> final 0.2770 = +3.9% (gate <=4; twin +14.5%)**, e9 roll 14->26, e4 20->30. THE BILL: last-writer starvation — e7 block-min +48.7% -> roll 44->28 (the corefrac store accumulates monotonically; by t4 the writer is attenuated across 4 prior profiles); e2 +10.5% absorbed. Final 39.6 ~= twin 42.4 (victims up, writers down). e6 34-vs-54 = **chunk-rollout misrank #3** (its function flat + family-best 0.0714->0.0728) — e6 cells carry no attribution weight, now demonstrated three ways. NEXT (queued): a non-accumulating/decaying protection store to keep e9's +12 without e7's -16.
- freeze5k (top-5k A-core seeded, hard_u 0.9): 0/5000 seeded slots moved, bitwise, 40/40 checks — the protect_seed_path mechanism is perfect. Strategically NULL at this horizon: selfcov -10pp/task (frozen slots carry 10-24% of read mass, un-adaptable — a direct read-write-product hit), e7 block-min +26.5%, bleeds UP (displaced budget crowds victim shoulders) => e9 give-back WORSE (+19.2% vs twin +14.5%), finals 39.2, all 5 cells <= twin, jitter slopes ~= twin (no off-trail dividend in-shell). The A-core erosion it guards against measures 0.2-0.5% core50 here — it prevented non-damage. RETIRED at 5-task scale; revisit only if a 10-task run shows real A-core erosion.

**Multitask LoRA (one adapter, 5 tasks, SAME 5k total budget):** 49.2 overall — e4 38 / e6 52 / e9 36 / e2 72 / e7 48. Pre-registered: e6 UP vs specialist 44 ✓ (breadth removes the n=1 penalty, lands in the staged 54-56 band); e4/e9 down ✓ (precision-bound, 1/5 per-task exposure); **iso-budget strong form FAILS** (matched-3 avg 42.0 vs specialists 57.3) — the two currencies (precision, support) trade at task-specific rates; support pays near threshold, precision rules far from it. e7=48 = best e7 measured anywhere.

### THE BUILD (decided with Josh; code SHIPPED + smoked)

**Text-span VLM memory:** modules on paligemma LM layers [15,16], attached to the LAST-200 prefix positions (the tokenized language field = instruction + state-as-text), retrieval computed on that slice only. Placement logic: L15+ keeps the prefix KV at <=14 bit-identical => the expert router's certificate, its frozen-base routing, AND its value-path inputs all survive untouched (per-layer KV pairing: suffix i attends prefix i only); the lowest VLM module's router input is memory-free by construction. Rejected: full-attachment v1 (routing on the crowded 0.96+ image cone, 14x counter blow-up — prefix is ~770 tokens vs the suffix's 51, the "why were VLM forwards so high" mystery from day-2, now a design input); layers 16/17-only (geometry degrades); <=L14 (dual-path prefix build not warranted for the leverage delta).

**Uniform n256/r2 across all 6 memory layers (Josh):** the expert bank right-sized 384^2->256^2 (65,536 ~= its measured 58-64k aggregate effective usage — right-sizing a FRESH bank, not slot-transplant pruning) + VLM 256^2 r2 on the 2048-dim hidden. VRAM: total values 2.15B (quadruple 34.4GB < today's expert-only 38.7) — the expert shrink pays for the VLM addition; residual lever = bs16/accum2 or VLM knn16 vs the gather-activation term (stored per-token slot matrices — the cost term bank size does NOT reduce), chosen by a measured VRAM gate before A-phase. VLM knn=16 (routing_loss_topk aligned =16, E24 rule).

**Code (commit pending this entry):** memory_config.py — vlm_layers/vlm_mem_n_keys/vlm_lora_rank/vlm_mem_knn/vlm_text_span + internal text_span; memory_lite.py — MLPPlusMemory text-span forward (slice retrieval, containment: pre-span positions bitwise plain-mlp; memory_only guarded); modeling_pi05.py — VLM attach via derived cfg (dataclasses.replace) with placement guard (vlm_layers must sit ABOVE the highest expert memory layer). The joint training path already dispatches wrapped MLPs generically for BOTH towers (task_ids/lang_emb reach VLM modules with zero new threading); prefix-only inference calls mlp(x) plain (losses off, memory active) — correct. Smokes: module suite 16/16 (smoke_vlm_memory.py: containment, slice-only retrieval, short-seq guard, grads, legacy identity, derived attach) + policy-level on the real stage-1 checkpoint (attach logs, 6/6 trainable tensors on LM15/16 only, retrieval token dim exactly B*200, aux losses live and keys grads NONZERO on the two-forward queue test — the first-forward zero-grad is the known E23 single-task/empty-queue artifact).

### VLM routing sweep (3 arms, ~2.3h each + 35min audit; regime hazards: (1) geometry already open => sep may need less force; (2) instruction constant within task => contrastive intra-pull is an elevated E21 collapse risk — the state tokens carry the within-task variation the router must keep)

Gates (vlm_audit_analysis.py, re-anchored for the 65,536 bank): PASS = held-out famIoU <= ~0.25 AND per-task core50 >= ~650 AND effnum >= ~500; COLLAPSE tripwire effnum <= ~150 (per-task-bias signature, kill regardless of IoU); stretch famIoU <= 0.15. Winner -> joint A-phase (both value banks; expert re-warmed at n256 in parallel on this box, gate: famIoU <= ~0.20 / core50 >= ~1300 vs the n384 certificate 0.145/2955) -> e4 1-task plasticity probe (chunk toward <=0.08 = build validated; ~0.12 = restricted attachment failed -> full-attachment fallback arm) -> 5-task sequential.

Scripts to run (job_scripts/nebius/libero_90/staged/):
- vlm_rwarmup_sweep_armA_c0.05_sep5.0.sh   <- Running on base VM (expert_rwarmup10k_n256.sh chained behind it)
- vlm_rwarmup_sweep_armB_c0.0125_sep2.0.sh
- vlm_rwarmup_sweep_armC_c0_sep5.0.sh

### Entry 44 addendum (18 Jul ~14:00) — PAD CONTAMINATION bug in the v1 text-span attach: found at arm A's audit, mechanism-verified, FIXED, sweep relaunched as _padfix

Arm A's audit failed both gates (famIoU 0.466/0.469, core50 95-418) with a signature that unmasked a v1 design bug: the last-200 span includes ~140 PAD positions (valid language tokens ~52-57), and pad hiddens — near-identical across all samples and tasks — query the memory too. Direct verification on the arm-A checkpoint: the pad region's ~40k slot-draws collapse onto ~230 unique slots (the shared core) while the text region routes richly (8-9k unique slots/task-batch) — the router is healthy on real tokens; the statistics, TF counts, contrastive means, and routing histograms were drowning in 70% pad mass. Contaminated B/C audits confirm ARM-INVARIANCE (famIoU 0.454-0.507 for all three recipes — the sweep measured the pad floor, not the routing configs).

FIX (commit with this entry): `token_mask` threaded through HashingMemoryLite.forward — the language attention mask reaches the VLM wrappers via `set_vlm_token_mask` at both pi05 embed_prefix call sites. Valid tokens are a contiguous PREFIX of the field, so the loss/queue machinery runs on a rectangular [:, :Tv] slice (no internal loss-fn surgery); last_indices/last_weights + per-task diagnostics filter by the full mask (audit/TF-IDF/protection see valid tokens only); memory output zeroed at pads (bitwise plain-mlp there); stale-mask shape guard -> unmasked behavior. Ring-buffer sizing pinned via size_T (varying Tv would have churned the routing queue). Smokes: module suite 20/20 (S9a-d pad exclusion) + policy-level (last_indices token dim 431 = sum of valid tokens across the batch, aux losses + keys grads live). Sweep run/audit names bumped to _padfix; contaminated checkpoints retained for the before/after comparison. Arm A relaunched on this box (concurrent with the expert n256 re-warm); B/C need a pull + relaunch on the VMs.
## Entry 45 - 18 Jul 26 (VLM routing sweep: ALL THREE ARMS FAIL famIoU (0.42-0.53) — failure is CONTENT-STRUCTURAL, not dose: the state-as-text sub-span routes on shared digit vocabulary (sub-span probe: state region famIoU 0.36-0.40 vs instruction 0.21-0.22, which PASSES gates on the failed checkpoints). Aux-only regime: contrastive = the SPREADING force (breadth monotone in c; B-vs-D single delta). Querystats probe: state positions carry task signal at 0.11-0.13x their digit noise; pooled-state key near-degenerate (family cos 0.98-0.99); instruction anchor alone already carries within-task conditionality (intra-cos 0.864-0.890). => POOLED-ROUTER build shipped (route state region on one per-sample anchored key; value path per-position) + 3-arm overnight chain to the e4 decision instrument)

### Sweep verdict (padfix arms; gates famIoU<=0.25 ∧ core50>=650 ∧ effnum>=500; collapse effnum<=150)

| arm | c | sep | famIoU L15/L16 | bgIoU | core50 mean L15/L16 | verdict |
|---|---|---|---|---|---|---|
| A | 0.05 | 5.0 | 0.467/0.502 | 0.294/0.349 | 8.4k/10.3k | FAIL sprawl |
| B | 0.0125 | 2.0 | 0.418/0.463 | 0.200/0.272 | 4.2k/6.9k | FAIL (best) |
| C | 0 | 5.0 | killed <1k steps (top-1 0.7, support ~300) | | | COLLAPSE |
| D | 0.5 | 2.0 | 0.501/0.529 | 0.364/0.411 | 11.0k/13.1k | FAIL (pre-registered sprawl signature) |

Integrity: configs verified single-delta from checkpoints; the 5a64d2b4-vs-b1ce1350 code skew between arms is immaterial (diffed: loss/queue machinery ran on the valid-token slice in both; never-attach = hygiene/VRAM); pad fix verified working (cores 3-13k with real arm variance vs the contamination's 95-418 shared pad core — same famIoU number, completely different object). No collapse in A/B/D (effnum min 1286).

### Mechanism (three facts + one instrument)

1. **c sets breadth, monotonically** (core50 4.2k -> 8.4k -> 11k at c 0.0125 -> 0.05 -> 0.5; B-vs-D is a clean single delta). In the aux-only warm-up (values pinned => MSE grad on router exactly 0, E36 smoke) SupCon's SAME-TASK-DENOMINATOR term (uniformity) is the only anti-collapse force — its coefficient sets the equilibrium radius. The joint-era "c = compaction" calibration inverts here: there MSE spread and alignment tightened; here sep collapses (arm C) and uniformity spreads. On this tower same-task mean queries differ ONLY via state digits => the uniformity push inflates exactly the generic subspace.
2. **sep bought nothing** (A sep5 >= B sep2 on famIoU) — translation can't keep clouds this large disjoint (E25-P7 replayed).
3. **The protocol is fine**: the expert n256 re-warm ran the identical aux-only recipe the same day and PASSED (famIoU 0.145 = the n384 certificate; core50 1713; family 0.267/0.115/0.055). The difference is the routed content.
4. **Sub-span probe** (NEW instrument, scripts/vla_analysis/probe_subspan.py — per-position routing reconstructed from the order-preserving eval-path stats; region split instr[0:16)/mid/state[28:]): the language field is two objects. State region (~2/3 of valid positions): each task routes onto ~45k slots (69% of bank) of shared numeric vocabulary — famIoU 0.36-0.40, bg 0.20-0.29. Instruction region: famIoU 0.212-0.224, core50 407-1610, bg 0.051-0.091 — arm A's instr region PASSES all three gates on the checkpoint that just failed; instr famIoU ~0.22 is ARM-INVARIANT (the genuine word-sharing floor, 3.4x bg). Caveat: per-position IoUs are estimator-depressed (sparse-vs-sparse, E24); region aggregates are the honest numbers. Artifacts outputs/analysis/e44/subspan_arm{A,B}.json.

### Discussion (Josh) -> the design

Options for the router input: (1) pool everything (rejected — sacrifices the measured-good per-token instruction routing); (2) per-token instr + pool(state) key for the state region; (3) per-token instr + state keyed by pooled(instr(+state)) — anchor + smooth offset = footprint translation by construction. Josh's push: measure the component variances ANALYTICALLY first and scale-match (the E28 lesson: ||scene||/||beta|| = 17-21x WAS the routing outcome).

**Querystats probe** (NEW, scripts/vla_analysis/probe_querystats.py, stage-1 features = the router input, libero_10; outputs/analysis/e44/querystats_stage1.json):
- State positions: between-task/within-task variance = **0.126 (L15) / 0.112 (L16)** vs instruction 0.985/0.810 — task context IS mixed in (Josh's hypothesis, confirmed present) but sits 8-9x under the digit noise => per-token state routing routes on the generic 88% (the sprawl, quantified).
- Pooled state component: between-task cos 0.931/0.970, **family 0.981/0.991** — near-degenerate; a state-only palette key has almost no family separation to work with.
- Instruction pool: between 0.800/0.866, family 0.882/0.919; **intra-cos 0.864/0.890** — the anchor alone already carries within-task variation (image/state context bleeds into instruction-token hiddens) => the E21 constant-palette fear is empirically weakened at b=0.
- Composite grid k = a*nrm(instr)+b*nrm(state): separation degrades monotonically in b (inter 0.800 -> 0.898 -> 0.931 across b=0 -> 1 -> state-only). **(1,0) dominates the init geometry**; b>0 only pays if the trained proj converts the state channel into conditionality worth the separation cost. Pooled means arrive at sqrt(rho)-shrunken norms (coherence 0.22-0.49) => component RMS-normalization is mandatory (the lop-sidedness Josh predicted, measured).
- Multiplicity design note: all state positions share one key => the palette enters losses/stats at xn_state. Deliberate split: TF/write-demand counts keep the multiplicity (slots serve n_state positions); routing-loss dedup (loss-mask) DEFERRED as a fast-follow — tonight's arms carry the documented z-mean weighting (~2/3 palette).

### THE BUILD (shipped, smoked)

- `vlm_router_pool` ("" legacy / "anchored" / "state") + `vlm_router_pool_weights` (memory_config.py); `MLPPlusMemory._pooled_router_keys` — router keys ONLY (value path per-position on live x): instr positions [0,b) per-token; positions [b,v) share a*nrm(pool instr[3:b]) + b*nrm(pool state[b+3:v-5]), rescaled to batch-mean token RMS; per-row fallback to per-token when the boundary is missing (memory_lite.py). Boundary threaded from the embed_prefix call sites via token id 3040 ("▁State", tokenizer pinned in processor_pi05.py): `_vlm_instr_len_from_tokens` -> `set_vlm_token_mask(masks, instr_len=...)` (modeling_pi05.py).
- Smokes: module S11a-i (broadcast-region routing identical / instr per-token / per-position value outputs / marker-less fallback / missing-instr_len == legacy bitwise / mode+weight variants / pre-span + ragged-tail invariants / grads) — suite ALL PASS; policy-level on the real stage-1 ckpt: instr_len=17 usable 8/8, **broadcast-region routing shared: True at L15+L16**, two-forward queue test keys grads 14.0/16.6.
- Merge tooling: `extract_router_bank.py` (expert tower's 60 memory tensors from the n256 re-warm, 1.11B params, bit-exact — values are the router-only warm-up's slot_down-random/slot_up-zero) -> outputs/analysis/e44/expert_bank_n256_rwarmup10k.safetensors (4.4G) + expert_bank_n256_config.json; `merge_banks.py` (pure safetensors union + config merge, no model build; placement guard makes the certificates provably survive: expert routing reads prefix KV <=L14 which VLM mem at 15/16 cannot touch; VLM routing input memory-free below).

### Tonight (3 boxes, all chained to the decision instrument; scripts staged/vlm_pool_*)

`vlm_pool_chain_common.sh`: warm-up 10k (router-only, c0.05/sep5.0) -> audit -> vlm_audit_analysis + subspan -> PERMISSIVE gate (kill only min-effnum<=100 or famIoU>=0.45 both layers — strict E44 gates don't transfer to palette-dominated famIoU; morning reads decide) -> merge -> A-phase 10k (values both towers, routers frozen, frozen-base expert routing, bs32 w/ bs16xaccum2 fallback) -> e4 1-task probe (C-config: beta4/top_t1536/5k steps/lr 1e-3->1e-4, 20-ep eval). Chunk probes centrally in the morning; winner -> 5-task attribution run (C-config, vs stageB 32.0/35.0 its near-twin). Arms: **A anchor10 (1,0)** [base box] / **B anchor1005 (1,0.5)** / **C statepool** [bracket end + querystats-calibration cell]. Deferred: 10-task extension (post-validation), sep refinement wave (tomorrow daytime if cells marginal), loss-mask dedup.

### Entry 45 addendum (18 Jul eve) — the pooled-state-key question resolved by the variance ledger; merge tooling tested end-to-end; 3-arm chain LAUNCHED

**"Doesn't the pooled state key also carry within-task information?" (Josh) — yes, and the probe says the anchor already carries more of it.** The ledger, from querystats: after pooling, the INSTRUCTION pool's within-task variance is 165.9/150.8 (L15/L16) vs the state pool's 77.9/84.9 — attention has already mixed image/proprio context into the instruction-token hiddens, and that contextual bleed survives pooling ~2x better than the state signal survives its own digit noise. Sharper still, the state pool's variation is task-INTERLEAVED: intra-cos 0.852 < inter-cos 0.931 (two same-task samples are less similar than two task centroids — task structure buried), while the instr pool is coherent (intra 0.864 > inter 0.800). So b>0 buys little conditionality not already present at (1,0), at a measured separation cost (inter 0.800->0.898 at b=1). Framing worth recording: **(1,0) is not state-blind memory — state enters through the per-position value path (U_sV_s x_p on each live hidden) and through the anchor's contextual bleed; b controls only whether SLOT SELECTION keys on direct proprioception.** Standing caveats (why (1,0.5) still gets a box): these are init-time statistics through the identity map — a trained proj could extract the task-orthogonal component of the state pool; and variance accounting is not information accounting (phase gating could need the direct channel).

**Merge tooling tested end-to-end** (old armA ckpt + the shipped bank): 895 tensors (22 VLM mem + 60 expert mem), config graft correct (expert [8,10,12,14]/n256/knn36 + VLM fields); the merged artifact loads in exact A-phase mode — **48/847 trainable** (= values + gate/value_proj/swilu x 6 memory layers, both towers), forward/backward clean at 1.0s/step. Tolerance fix for pre-pool configs (66d46b84). Mode flags (train_router_only=True etc.) ride in checkpoint configs and are explicitly overridden at every downstream stage's CLI (E37 rule, applied).

**Launch board (21:15-21:30 BST):** arm A anchor10 (1,0) = base box, tmux `poolA`, log outputs/e45_armA.log; arm B anchor1005 (1,0.5) = VM2 `poolB`; arm C statepool = VM3 `poolC`. Bank files shipped to both VMs; local-Claude instructions issued (launch, health signs, no-relaunch rule, morning rsync list, final sync ~07:30 UTC). Chains land ~07:00-08:15 BST. Morning protocol: chunk probes on the three e4 checkpoints (pass toward <=0.08; staged-best 0.153, LoRA target 0.020) + audits + subspan region reads -> winner -> 5-task attribution run (C-config) at breakfast. A hard-gate stop overnight ("GATE: HARD FAIL") is a result, not a bug.

---
## Entry 46 - 19 Jul 26 (Pooled-router sweep verdict: BOTH anchored variants hit the expert-certificate routing level (famIoU 0.145-0.151) and produce the largest e4 fit gain in the project (chunk 0.153 -> 0.099, -35%); winner = anchor+state (arm B). Discussion corrections: capacity ledger = read-write product (params fell 11% while the palette became an always-read/always-trained block); L16 routing was NON-stationary (Josh's catch) -> frozen-route extended to the VLM tower (bitwise-exact smoke); route-once shipped (retrieval computed once per shared key; knn36 VRAM 136 -> 127GB); bank-scaling law measured (footprints track bank size, famIoU invariant). -> E46: three like-for-like joint router re-warms (incumbent / vlm-knn36 / uniform n128-r4) LAUNCHED, stop-after-audit)

### Results — the overnight pooled-router sweep (three chains, all completed unattended)

Each chain: VLM router warm-up (10k, aux losses only) -> held-out audit -> merge with the
certified expert n256 bank -> joint A-phase (values both towers, 10k on libero_90) -> single-task
e4 adaptation probe (5k steps, C-config) + executed-chunk error. famIoU = read-mass overlap
between the three near-identical basket tasks (pass <= 0.25); chunk error = deviation of the
executed 50-step action sequence from the demonstration (the rollout-predictive fit metric;
previous staged best 0.153, single-task LoRA specialist 0.020).

| arm (state-region key) | famIoU L15/L16 | e4 chunk | e4 roll (20 ep) | e4 block loss |
|---|---|---|---|---|
| A anchor only (1,0) | 0.151/0.145 | 0.0996 | 25 | 0.092 |
| B anchor+state (1,0.5) | 0.149/0.147 | **0.0994** | **35** | 0.091 |
| C state-only pool | 0.227/0.248 | 0.1132 | 25 | 0.094 |

- Routing: the anchored variants land AT the expert-side certificate level (0.145) — 3x better
  than the failed per-token design (0.42-0.53). Region split (sub-span probe): the state region,
  0.38-0.40 under per-token routing, sits at 0.08-0.10 with near-private palettes (background
  ~0.01); the instruction region keeps its measured word-sharing floor (~0.19-0.21).
- Fit: chunk 0.153 -> 0.099 is the largest single-move e4 improvement recorded (ten prior levers
  each moved it <= 10%); one-step block loss 0.091-0.094 vs best-ever 0.114. Against the
  pre-registered gate (<= 0.08 validated / ~0.12 fallback): between, leaning pass; the 25-35%
  rollouts sit exactly on the fit-to-success curve through the compass anchors.
- Controls: the expert tower's usage was BITWISE identical across all three arms (same shipped
  bank + stationary routing + same seed) — differences attributable purely to the VLM change.
  Arm C validates the querystats probe's ORDERING while showing trained projections recover more
  than raw-feature geometry suggests (predicted near-hopeless at family-cos 0.98, trained to
  0.227) — probes rank, never veto.
- Expert side-by-side (same day): n384 and n256 audits both famIoU exactly 0.145; the shared-soup
  task pair floors at 0.23-0.27 on BOTH towers under three different routers — the residual
  overlap is task-semantic, not router-specific.

### Conclusions

1. Routing content determines routing geometry: dose-sweeping losses never fixed digit-token
   routing; changing the KEY CONSTRUCTION fixed it in one shot at unchanged doses. The anchored
   key implements footprint translation structurally; the losses only arrange anchors.
2. The correct capacity ledger is the READ-WRITE PRODUCT, not parameter count (Josh): total
   adaptation params FELL 11% in the new build while e4 fit jumped 35% — because the pooled
   palette is an always-read, always-trained block (top-100 palette slots inside the write mask
   4,886/5,000 steps — the degenerate-perfect product that makes dense LoRA strong). Placement
   and product are now separately evidenced: placement by the compass (expert-side dense adapter
   train-loss CEILINGS at ~0.18 — no exposure story rescues it); product by arm C (same
   placement/params, broader palettes -> worse fit).
3. The "chunk ~0.03 for 50%+" target is the CONSERVATIVE (specialist-anchored) bound; the breadth
   law says our 90-task support should discount it (not yet visible on e4 — 0.099 -> 25-35 sits
   on the specialist curve).
4. Bank-scaling law (answers the n128 sizing question): per-task footprints are NOT a fixed
   absolute — the n384->n256 shrink (2.25x) shrank core50 1.7x (2955->1713) at famIoU exactly
   0.145 both. The losses re-express the same angular geometry at the bank's resolution. Residual
   risk at a further 4x: the fixed 144-slot per-query draw becomes a floor.
5. L16 routing was non-stationary (Josh's catch): its router read the live stream containing
   L15's memory output, and was certified with values at zero — the one-layer edition of the E38
   drift channel, silently accepted in the E44 build. Also: knn deployment must match the
   routing-loss candidate pool the keys were trained with (E14-16 alignment) — flipping knn on
   16-trained keys is an uncertified router.

### Builds shipped today (all smoked + pushed: 543cb4e1, c74862de, + topk alignment)

- **Frozen-route extended to the VLM tower** (same flag governs both towers): a no-grad fork
  advances a memory-free prefix stream from L15 and recomputes L16's attention front on it;
  inference reuses the capture/stash dual pass. 15/15 smokes; router features BITWISE-exact on
  every valid token (the only diffs live on pad rows, which attend nothing and are excluded from
  memory — they also explain the first smoke run's spurious failures). No router retrain needed
  in principle (memory-free = the certification distribution up to the value_proj-bias residual,
  the known 0.98-transfer term).
- **Route-once** for shared router keys: the state region routes ONCE per sample on a compact
  sequence [state key, instruction tokens] (key first keeps valid tokens a contiguous prefix for
  the loss machinery); slot params gathered once per row, applied per-position
  (apply_shared_palette). Output parity with the redundant path 1.2e-07; usage/TF/audit stats
  keep served-position multiplicity (numbers comparable); routing/contrastive losses now count
  each unique key ONCE — the pre-registered dedup, a deliberate change to future warm-ups'
  training signal. VRAM: A-phase 127.8 -> 122.1GB (knn16); **knn36 136.4 -> 126.9GB** — the knn
  axis is unblocked (was ~4GB headroom, now ~13.5GB).
- **Per-tower routing_loss_topk alignment** codified: the derived VLM cfg sets topk = vlm_knn.
- Also measured: knn is now a per-task palette-capacity lever (64 -> 144 slots x r) under pooled
  routing.

### Discussion -> the E46 design (with Josh)

Selection of the next frontier config runs as THREE like-for-like joint router re-warms — one
protocol, one loss semantics (the deduped losses), frozen-route-consistent inputs, per-tower topk
alignment — differing only in declared shape knobs. Rationale for re-warming even the incumbent:
(a) arm 3's knn36 requires keys TRAINED at topk 36; (b) the old warm-up carried the bias-residual
input gap and the old multiplicity-weighted losses; (c) uniform protocol makes the certificates
comparable. Each arm: warm-up -> audit (expert + VLM analyses + sub-span probe) -> STOP — no
value-filling on an uncertified router. Plasticity comparison happens AFTER certification via
A-phase -> filled e4 probe + chunk (comparable to last night's anchors; the blank-bank shortcut
was considered and rejected for cross-session comparability).

| arm | box | expert | VLM | question |
|---|---|---|---|---|
| 1 incumbent | VM2 | n256/r2/knn36 | n256/r2/knn16 | the baseline, re-certified under the new protocol |
| 3 knn axis | VM3 | n256/r2/knn36 | n256/r2/**knn36** | does palette capacity 64->144 pay? |
| 2 uniform | **base box (RUNNING, tmux jointA2)** | **n128/r4/knn36** | **n128/r4/knn36** | iso-rank-unit concentration: 4x fewer, stronger slots |

All: sep 5.0, c 0.05, anchored (1.0, 0.5), 10k compressed, router lr 1e-4. Audits land
~18:30-19:00 BST (the re-assess point). Pre-registered arm-2 read: expert core50 ~400-800 at
famIoU ~0.145 = scaling law holds; famIoU up at scaled cores = the per-query-draw floor binds.

### Next steps

1. **RUNNING: arm 2 uniform warm-up on the base box** (launched 09:49 UTC); arms 1/3 to launch on
   the VMs (local-Claude instructions issued).
2. Afternoon re-assess on the three audit certificates -> A-phases on survivors (VRAM-gated for
   arm 2's r4) -> filled e4 probes + chunk -> select -> graduate ONE config to the 5-task
   sequential (C-config, vs stageB 32.0/35.0).
3. Queued behind selection: 10-task extension; LoRA specialist cells e2/e7; the sep-response
   question in the anchored regime; protection-store decay fix before any multi-task run that
   needs it.

---
## Entry 47 - 19 Jul 26 (E46 three-arm audit verdict: the new protocol re-certifies the incumbent EXPERT router to the third decimal, but the route-once LOSS DEDUP collapsed the VLM state palette to ~2 query-draws (famIoU 0.08 -> 0.19-0.24) — the deduped loss under-weights the palette relative to its deployment read mass; arm 2 (n128) breaks the bank-scaling law on both towers. Fix: vlm_route_once flag (warm-ups broadcast, downstream compact) + arm 1'/3' re-warms LAUNCHING)

### Results — the three like-for-like joint router warm-ups (all completed + audited 19 Jul)

Recap of the design: each arm retrains BOTH towers' routers from the stage-1 backbone
(values pinned at zero, aux losses only, frozen-route inputs, per-tower routing-loss
topk aligned to knn), then runs the held-out audit and stops. famIoU = read-mass overlap
between the three near-identical basket tasks; core50 = slots carrying half a task's
read mass; effnum = effective slot count (exp of read-mass entropy). Anchors: the old
expert n256 certificate (E44 protocol) and the E45 pooled-router winner "poolB"
(anchored (1.0, 0.5) state key, VLM tower).

EXPERT tower (famIoU L8/L10/L12/L14; mean core50):

| arm | famIoU | core50 |
|---|---|---|
| anchor (old n256 cert) | 0.172 / 0.145 / 0.142 / 0.145 | ~1700-1800 |
| arm 1 incumbent (n256/r2/knn36) | 0.167 / 0.152 / 0.149 / 0.146 | ~1650-1840 |
| arm 3 (same expert config) | 0.167 / 0.152 / 0.150 / 0.146 | ~1650-1840 |
| arm 2 uniform (n128/r4/knn36) | 0.231 / 0.218 / 0.183 / 0.214 | ~715-930 |

VLM tower (audit famIoU L15/L16; audit bg; palette = the shared state-region key's
slot set, decomposed below):

| arm | famIoU | bg | palette famIoU | palette core50/effnum |
|---|---|---|---|---|
| anchor poolB (E45 protocol) | 0.149 / 0.147 | 0.071 / 0.085 | 0.087 / 0.076 | 258-342 / 619-803 |
| arm 1 (vlm knn16) | 0.229 / 0.220 | 0.024 / 0.032 | 0.239 / 0.223 | 46-50 / 126-139 |
| arm 3 (vlm knn36) | 0.204 / 0.210 | 0.030 / 0.038 | 0.191 / 0.186 | 109-110 / 306-310 |
| arm 2 (n128) | 0.326 / 0.267 | 0.048 / 0.055 | 0.301 / 0.224 | 89-102 / 246-282 |

Instrument note (fix deferred): route-once reordered the per-sample stats rows to
[palette x n_state, then instruction tokens], so the sub-span probe's position->region
labels are scrambled on route-once checkpoints — its "instr[0:16)" block is pure
palette. The audit summaries are position-agnostic and unaffected; the palette numbers
above were recovered from the constant-prefix block. The probe needs a route-once-aware
row mapping before its labels can be trusted on compact checkpoints (the E47 re-warms
run broadcast, where labels are correct again).

### Conclusions

1. **The E46 protocol changes are benign where intended.** Arms 1/3 reproduce the old
   expert certificate to the third decimal — frozen-route-consistent inputs and topk
   alignment cost nothing. Arms 1 and 3 matching each other that closely is the built-in
   control (with values pinned, the VLM knn difference cannot reach the expert stream).
2. **The bank-scaling law has a measured floor.** At n128 (16,384 slots), cores scaled
   into the predicted 400-800 band but famIoU rose ~47% and background ~50% on the
   expert tower — the pre-registered "per-query draw floor" branch fired: 144 drawn
   slots is 0.88% of the bank per query, and per-task cores ~5% of the table make
   collisions geometric. The VLM tower is worst-of-three on every number. **Arm 2 dead**
   (n256/r4 remains a possible future concentration arm — VRAM now plausible with
   route-once — but n128 is closed).
3. **The route-once loss dedup moved the VLM router's equilibrium, adversely.** In the
   aux-only warm-up regime, MSE has exactly zero router gradient; the palette's only
   spreading force is the contrastive same-task-denominator (uniformity) term (E45
   finding 1). Deduped, the palette key enters the losses/queues once per sample
   instead of once per served position (~35x): palette-palette pair mass fell ~3
   orders of magnitude, the uniformity pressure collapsed, and the trained projection
   stopped amplifying the pooled key's within-task contextual variation. Signature:
   palette effnum landed at almost exactly 2 query-draws in BOTH knn variants
   (126 ~ 2x64, 306 ~ 2x144 — footprint set by retrieval geometry, not the losses)
   vs poolB's 10-12 draws. Consequences: family palette overlap tripled (near-identical
   anchors' small palettes collide), background halved (small footprints separate
   everywhere else). Confounds ruled out: warm-up grad norms 0.048-0.058 (clip 1.0
   never engages -> no cross-tower clip sharing), frozen-input delta is the known
   0.98-transfer term.
4. **The principle (Josh's read, adopted): the warm-up loss should weight routing keys
   by their deployment read mass.** The palette serves ~2/3 of the language field's
   read mass every forward; the audit weights it accordingly (stat_repeat); the deduped
   loss weighted it at 1/18 of a sample. The old broadcast semantics were mass-correct.
   Dedup optimized a router for a read distribution we do not deploy.
5. Why the certificates matter beyond geometry: the palette is the ALWAYS-READ block —
   family overlap there is maximally-leveraged sequential exposure (a later basket task
   writing shared palette slots perturbs the earlier one at every state), and palette
   constancy hands all within-task state-conditionality to the value path (untested
   regime; poolB's state-conditional palette is the certified, fit-measured one at
   chunk 0.0994). Counterweight recorded honestly: a near-constant per-task palette is
   an even more degenerate-perfect read-write product (LoRA-shaped), so the fit
   consequence is genuinely unknown — testable cheaply as an A-phase hedge on today's
   arm 3 if wanted.

### Builds shipped (with this entry's commit)

- **`--policy.memory_layer.vlm_route_once`** (default true, memory_config.py /
  memory_lite.py): gates the compact route-once dispatch. False forces the byte-exact
  legacy broadcast path (shared key routed at every state position -> losses/queues
  carry served-position multiplicity). Warm-ups run false; A-phase/sequential/inference
  keep true (router frozen -> paths numerically interchangeable, parity 1.19e-07, and
  the compact path saves ~6-10GB VRAM at knn16/36). Smokes S13a-g appended to
  smoke_vlm_memory.py (suite ALL PASS): default-true attr, flag-off broadcast call
  shape, output parity 1.19e-07, per-position stats layout restored, stats row count,
  routing-queue rows 48-vs-24 (multiplicity restored), contrastive per-sample mean
  shifts 0.58 (palette weight back). Also verified the field rides the derived-VLM-cfg
  dataclasses.replace path.
- Scripts: joint_rwarmup_common.sh now passes vlm_route_once=false (+ header rationale,
  + BATCH_SIZE/GRAD_ACCUM env fallback for OOM); new wrappers
  joint_rwarmup_arm1p_incumbent_bcast.sh (VM2) / joint_rwarmup_arm3p_vlmknn36_bcast.sh
  (VM3), tags arm1p_n256r2_vlmknn16_bcast / arm3p_n256r2_vlmknn36_bcast.

### Next steps

1. **LAUNCHING: arms 1'/3' re-warms on VM2/VM3** (local-Claude instructions issued;
   same chain shape: warm-up -> audit -> analyses -> sub-span -> STOP, ~4.5-5h).
   Pre-registered reads: arm 1' ~ poolB's certificate (VLM famIoU ~0.149/0.147, palette
   ~0.08 / effnum 600-800) — it is poolB's exact replica plus the protocol
   improvements; arm 3' answers the knn36 palette-capacity question at matched loss
   semantics (the E46 comparison was confounded by the dedup pinning palette size to
   ~2 draws in both variants). Expert side expected unchanged (~0.145-0.17) for both.
   A broadcast joint warm-up at knn36 has never run — VRAM estimated ~125-135GB; the
   BS/ACC fallback covers an OOM.
2. **RUNNING on VM1 (this box): arm 2' = uniform n192/r4/knn36 both towers,
   broadcast losses** (decided with Josh; script
   joint_rwarmup_arm2p_uniform_n192r4_bcast.sh, run
   libero_90_pi05_jointwarm10k_arm2p_n192r4_knn36_bcast, BS16/ACC2 for the
   broadcast+r4 VRAM term). Why: n192 is the unmeasured scaling-law midpoint (4x
   cumulative from n384; the law held at 2.25x, broke at 9x), and n192/r4 is the
   CLEAN iso-budget concentration arm (147,456 rank-units/layer vs n256/r2's
   131,072; the original n128/r4 arm was only HALF the budget — a mislabeling in the
   E46 design). Pre-registered: famIoU ~0.145-0.16 at core50 ~1,100-1,300 = law
   holds; ~0.18+ = branch closed.
3. **Tonight (~21:30-22:15 BST): review ALL THREE audits together** (arm 1' knn16,
   arm 3' knn36, arm 2' n192 — every n and knn variant in one place) and graduate
   the winners. Realistic outcome sketched in discussion: discard knn16; graduate
   some subset of {arm 3-old (n256/knn36, compact "one state token" palette — its
   certificate defects are family-side and the 5-task window has one basket task),
   arm 3' (n256/knn36 broadcast), arm 2' (n192/knn36 broadcast)} to A-phase ->
   sequential. The compact-vs-broadcast pair at matched shape would also isolate the
   palette-equilibrium fit question at full pipeline fidelity.
4. After graduation: A-phases -> t0-block chunk (== the e4 probe: same C-config 5k
   steps; anchors 0.153 / 0.0994 / 0.020, kill >= ~0.12) -> 5-task attribution runs
   (C-config, vs stageB 32.0/35.0).
5. Deferred: sub-span probe route-once-aware row mapping; protection-store decay fix
   before multi-task runs needing it; 10-task extension behind selection.

---
## Entry 48 - 19 Jul 26 (E47 broadcast re-warm verdict: arm 1' REPRODUCES the poolB certificate (dedup attribution closed end-to-end); knn36's apparent advantage was a dedup artifact (score-profile-depth mechanism) but remains a 2.25x per-forward capacity lever; n192 FAILS via a newly measured mechanism — per-half subkey utilization crosses the pigeonhole knee (demand ~ n^0.74; n256 sits AT the knee) — the bank axis is closed with a sizing rule. GRADUATION: three chains on the palette-constancy axis (arm 1' / arm 3' / arm 3-old) -> A-phase -> 5-task, RUNNING)

### Results — the three broadcast re-warms (vlm_route_once=false; audited + analyzed 19 Jul eve)

Recap: the E46/47 arms retrain both towers' routers from the stage-1 backbone with
values pinned (aux losses only), frozen-route inputs, per-tower topk=knn — differing
only in bank size / knn / loss semantics. famIoU = read-mass overlap among the three
near-identical basket tasks; palette = the slot set selected by the shared state-region
key; core50/effnum = footprint size measures. Anchors: the n256 expert certificate and
the E45 winner poolB (VLM tower, old broadcast protocol).

EXPERT (famIoU L8/L10/L12/L14; mean core50):

| arm | famIoU | core50 |
|---|---|---|
| n256 certificate | 0.172 / 0.145 / 0.142 / 0.145 | ~1700-1800 |
| arm 1' (n256/r2) | 0.166 / 0.151 / 0.149 / 0.146 | ~1650-1840 |
| arm 3' (same expert) | 0.167 / 0.151 / 0.150 / 0.146 | ~1650-1840 |
| **arm 2' (n192/r4)** | **0.212 / 0.205 / 0.157 / 0.211** | ~1280-1480 |

VLM (audit famIoU L15/L16 | palette famIoU | palette effnum | instr famIoU (instr bg)):

| arm | audit | palette | pal effnum | instr |
|---|---|---|---|---|
| poolB anchor | 0.149/0.147 | 0.087/0.076 | 619/803 | 0.190/0.207 (0.13/0.15) |
| **arm 1' knn16** | **0.136/0.156** | **0.074/0.084** | **685/851** | 0.198/0.219 (0.13/0.16) |
| arm 3' knn36 | 0.170/0.192 | 0.111/0.125 | 765/1039 | 0.261/0.283 (0.17/0.21) |
| arm 2' n192 | 0.202/0.195 | 0.136/0.111 | 650/851 | 0.309/0.330 (0.21/0.25) |
| arm 3-old (dedup) | 0.204/0.210 | 0.191/0.186 | 306/310 | ~0.09-0.10 per-pos |

Incident note: arm 2''s audit OOMed twice (the audit script ran the checkpoint's saved
broadcast+r4 config; 8.4GB short at bs32, then 1.5GB short with the route-once
override). Fixes shipped: audit_heldout_routing.sh now forces the compact route-once
path (routing-identical; parity-smoked) and takes AUDIT_BS/AUDIT_STEPS env overrides —
arm 2' re-audited at bs16 x 200 steps/task (total audited samples MATCHED to every
other audit; famIoU/core50/effnum are mass-normalized aggregates, invariant to the
batch/step split at matched coverage).

### Conclusions

1. **The dedup attribution is closed end-to-end.** Restoring the broadcast loss
   semantics (one flag) reproduced poolB's certificate within noise on both regions —
   palette famIoU 0.074/0.084, palette effnum 685/851 (the state-conditional ~10-draw
   palette), instruction region at its word-sharing floor — while KEEPING the two
   protocol improvements poolB lacked (frozen-route training inputs; topk alignment).
   Arm 1' is the best-certified router in the project. Expert side: four warm-up
   variants now land on the same certificate to the third decimal — axis closed.
2. **knn36's apparent E46 advantage was a dedup artifact, but knn remains a genuine
   per-forward capacity lever (Josh's catch).** Two mechanisms, now separated:
   (a) FOOTPRINT: under broadcast losses the palette footprint is set by the loss
   equilibrium (~700-1000 effnum at either knn), not the draw — so knn36 buys no
   footprint. Under dedup the footprint HAD collapsed to ~one draw, so the bigger draw
   masqueraded as capacity. (b) OVERLAP — the score-profile-depth mechanism: a query's
   slot-score profile is steep at the top (key-specific) and flattens into a generic
   shoulder (shared structure); knn sets how deep the read cuts. Family keys agree in
   the shoulder and differ at the peak, so reading 144 slots instead of 64 pulls in
   the shared shoulder: palette famIoU 0.074 -> 0.111, instr 0.198 -> 0.261 at
   identical recipes. BUT (c) per FORWARD, knn36 places 144 x r2 adaptive params in
   every mixture vs knn16's 64 — 2.25x the read-participation half of the read-write
   product, with the E14-16 expert-tower precedent (fit monotone in knn 8->36) behind
   it. Certificates favor knn16; per-forward capacity favors knn36; the t0 chunk
   arbitrates. Sequential-side caveat both ways: E13-14 showed smaller knn makes each
   shared slot MORE load-bearing per overwrite.
3. **The bank-scaling law's breakdown mechanism is measured: subkey pigeonhole.**
   Each slot is an (i1,i2) pair of per-half subkeys (bank n^2 = n x n). Per-task
   effective subkeys used per half at L14: n384 ~192 (50% of the half), n256 140-144
   (55%), n192 114-120 (61%), n128 80-84 (64%) — demand compresses SUBLINEARLY,
   fitting demand ~ n^0.74 across all four points. Once two tasks' half-marginals each
   cover >~60% of the same n subkeys, the pigeonhole principle forces overlap no
   matter how well separation translates footprints (two 62% subsets of 128 must share
   >=24%; two 55% subsets of 256 only >=10%). The forced floor tracks the measured
   famIoU rise (0.145 -> ~0.21 -> 0.21-0.23). n256 sits AT the knee — which is WHY it
   held and everything below broke, and why n192 was nearly as bad as n128. **Sizing
   rule: keep per-half subkey demand under ~55-60% of n.** Bank axis closed: 384 ok,
   256 ok (operating point), 192 dead, 128 dead. The n192/r4 iso-budget concentration
   bet dies with it (the failure is routing-side; no value configuration can rescue a
   colliding router).
4. **The three graduates form ONE axis: palette constancy** (footprint/draw = how
   state-conditional palette selection is): arm 1' ~13 draws per footprint (most
   state-conditional) -> arm 3' ~6 -> arm 3-old ~2 (near-constant per task = the
   LoRA-like degenerate-perfect read-write product). Tonight's chains are a 3-point
   read-out of the constancy-vs-fit curve, not just pairwise contrasts.

### Graduation (decided with Josh) — RUNNING

Each chain (grad scripts, commit c8d301f0): A-phase (10k values-only both towers on
libero_90, routers frozen, frozen-route ON, compact path restored downstream) -> 5-task
sequential (C-config: beta4, top_t 1536, 5k steps/task, value lr 1e-3->1e-4, 20-ep
intermediates + 50-ep final; comparator stageB 32.0 final / 35.0 init). Arm 2' NOT
graduated (pre-registered kill: famIoU >= ~0.18 at 3 of 4 expert layers).

| box | chain | axis position |
|---|---|---|
| base box (RUNNING, tmux grad_a1p) | arm 1' | state-conditional end, best certificates |
| VM3 (instructions issued) | arm 3' | middle; the knn/per-forward-capacity contender |
| VM2 (instructions issued) | arm 3-old | constant end; compact-palette fit hypothesis |

**Gate 2 (pre-registered, checked at breakfast ~07:30 BST):** each sequential's t0
block IS the e4 probe (same C-config 5k steps) — chunk error on checkpoints/005000 vs
anchors 0.153 (old staged best) / 0.0994 (poolB) / 0.020 (LoRA specialist); kill any
run >= ~0.12 before its remaining blocks waste the morning. VM claudes rsync t0
checkpoints to the base box when they appear (~02:30-04:00); chunk probes run
centrally with the standard instrument.

### Next steps

1. Breakfast: Gate-2 chunk read on all three t0 checkpoints (+ kill/continue calls).
2. Midday: 5-task finals + probe battery -> select the 10-task carrier. If arm 3'
   wins fit materially, the knn choice becomes capacity-vs-family-exposure and the
   10-task decides; if fit ~equal, arm 1' wins outright on certificates.
3. Deferred (unchanged): sub-span probe route-once-aware row mapping (only needed for
   compact checkpoints); protection-store decay fix before 10-task; 10-task extension
   behind selection.
4. Cleanup this session: arm 2' warm-up checkpoints deleted (dead branch; audit +
   sub-span + wandb retained); arm 3-old's staged local checkpoint copy kept until its
   VM2 chain completes.

---
## Entry 49 - 20 Jul 26 (E48 graduation verdict: arm 1' = NEW FRONTIER 40.0 at plain C-config — the palette-constancy axis is monotone on every instrument incl. all 15 cross-arm final-chunk cells; knn axis closed at 16; the read-write product heuristic gains a STATE-CONDITIONALITY factor; new protection-x-palette starvation mechanism measured; retention POSITIVE in function space. -> composition + VLM-r4 arms scripted for the VMs (graft tooling shipped); image-span plan fixed (step 1 keep-expert single-delta / step 2 drop-expert reallocation); 1A running)

### Results — the three graduation chains (all completed; battery in outputs/analysis/e48/)

Recap: three chains from the E47/48 certified joint router warm-ups, differing ONLY in
the VLM router — arm 1' knn16 broadcast-loss keys (state-conditional palette, ~11-13
slot-draws per footprint), arm 3' knn36 broadcast (~5-7 draws), arm 3-old knn36
dedup-loss keys (near-constant ~2-draw palette). Expert tower config-identical
(n256/r2/knn36); each chain = A-phase (10k values-only, both towers) -> 5-task
sequential (C-config: beta4, top_t 1536, 5k steps/task, value lr 1e-3->1e-4, 50-ep
finals). Terms: palette = the slot set retrieved by the pooled state-region key (the
always-read block); chunk error = executed 50-step action-sequence deviation after a
real 10-step denoise (rollout-predictive fit metric; anchors 0.153 old staged best /
0.0994 poolB / 0.020 LoRA specialist); RTO = fraction of a task's read mass on slots
later tasks updated.

| arm | final (50ep) | e4/e6/e9/e2/e7 | block-min loss | e4 own chunk |
|---|---|---|---|---|
| arm 1' | **40.0** | 14/58/16/82/30 | **0.0940** | **0.112** |
| arm 3' | 37.2 | 16/56/14/72/28 | 0.0984 | 0.126 |
| arm 3-old | 32.4 | 24/42/12/66/18 | 0.1082 | 0.127 |
| stageB (comparator) | 32.0 (20ep) | — | 0.1274 | 0.160 |

Comparators (50-ep finals): lr2x 35.6 / steps7k 36.0 / top3k 37.6 / softp 41.2 /
lr2x+3072 40.4 / lr4x+3072 42.4. Josh's read on first sight — "similar boost as higher
lr and higher top-t" — is correct at the scoreboard level AND the crucial difference:
arm 1' reaches the composition band at 1x LR / top_t 1536, i.e. with those levers
still unspent on this substrate.

Probe battery (all pre-registered instruments run):
- arm 1' own-block chunk grid vs stageB: e4 0.112 (0.160), e6 0.078 (0.127), e9
  **0.223** (0.407 — best e9 function ever measured, beats the 4x-LR arms), e2 0.164
  (0.287), e7 0.096 (first-ever e7 cell). Mean −40-45%.
- Cross-arm FINAL-checkpoint grid: **all 15 cells monotone** on the constancy axis
  (means 0.130 / 0.148 / 0.167). arm 3-old's worst relative cells are the mechanism
  cells: e6 +53% (see protection-x-palette below), e7 +40% (last writer).
- own->final: arm 1' IMPROVED on all four earlier tasks (−2.3..−5.0%) — the first
  uniformly-improving retention grid in the project (lr-era e9 gave back +7.7/+14.5%).
  Rollout give-back +5.0.
- MSE forgetting matrix (arm 1', 5x5 paired-noise): flat — diag drift e4 +4.8%, e6
  +2.2%, e9 +2.4%, e2 +1.4%, e7 +0.0%. Forgetting stays solved in the 6-module
  architecture. e4's +4.8% (~3x the stageB-era ceiling) is the one-step fingerprint of
  the new VLM exposure — its chunk moved the OTHER way (−3.5%), so no functional cost
  in-window.
- Slot autopsy: expert tower ARM-INVARIANT to ~1% (the built-in control — all deltas
  are VLM-side); computed read-IoU matches logged memory_iou to 4 decimals on all
  three runs. VLM footprints track the warm-up certificates into production (VLM15
  c50, e2: 365 / 376 / 95). NEW exposure surface: VLM RTO on early tasks 47-55% (arm
  1' e4, L15/L16) vs expert ~16-18% — currently benign (above). e9's expert footprint
  dilution persists at n256 (c50 2.4x median; its read-write product lowest).
- NEW MECHANISM (measured): protection x palette constancy. In arm 3-old, e6 holds
  11.1% of its VLM read mass on prior-hot (u>=0.5) slots and updated 0.0% of them —
  beta4's peak-normalized u vetoes the always-read palette core when the palette is
  near-constant (the palette IS the mega-hot set), so every later task is locked out of
  writing where it always reads. In the state-conditional arms the same number is ~0%
  (protection inert on the VLM tower). Note the semantics: e6 (mug+pudding) collides
  with e4 (two mugs) on shared "mug" content — palette overlap follows objects.

### Conclusions

1. **The VLM build is the largest fit mover of the staged era** (−26% one-step, −40-45%
   chunk at matched optimization), and rollouts moved exactly where the threshold law
   allows: e6 (0.078 -> 58, best-ever) and e2 (0.164 -> 82) converted; e4/e9 stayed
   pinned at 3-4x their thresholds despite best-ever function. Threshold law now ~9/9.
2. **State-conditionality of the read set is the capacity currency — the read-write
   product heuristic (E43) gains a third factor.** arm 3-old has the HIGHEST product
   ever measured (3.9-4.3k read-mass-weighted update events, the "degenerate-perfect"
   shape) and the worst fit: a constant palette is a fixed ~64-slot adapter per task,
   while a state-conditional palette addresses 600-850 slots across the task's state
   distribution at the same parameter count. Product x conditionality converts; product
   alone does not.
3. **knn axis closed at 16.** knn36 lost on every instrument despite 2.25x per-forward
   read participation — the score-profile shoulder carries into production (extra
   slots are family-generic; selfcov lower; the stable shoulder TF stops the write
   mask rotating over task-specific slots). Read participation without task-specific
   content is not capacity.
4. **The deployment-mass-weighting principle is validated at rollout level**: arm 3-old
   (dedup keys) == stageB (32.4 vs 32.0) — the identical architecture under collapsed
   palettes buys ~nothing; the one-flag broadcast fix is worth +7.6pp.
5. Honest negatives: arm 1' did NOT reproduce poolB's e4 chunk (0.112 vs 0.0994,
   mid-field gap at instrument resolution, one-step loss BETTER — the two protocol
   fixes bought nothing measurable); Gate-2 counterfactual — the ~0.12 kill line would
   have killed arm 3' (0.126) and arm 3-old (0.127), right that neither graduates,
   slightly conservative on arm 3' (final 37.2 > stageB); 20-ep init cells
   re-vindicated as retired (arm 3-old init-mean 25.6 -> final 32.4; arm 1' e6 init 20
   -> final 58).
6. Warm-up certificates are predictive through the full pipeline for the third
   consecutive time. The 3h certificate continues to be worth ~30 GPU-hours.

### Discussion (Josh) -> decisions

- **The 8-day / 70% arithmetic, corrected and re-anchored.** Josh challenged the
  pessimism citing three lifters; the ledger has TWO (freeze-top-5k was misremembered
  — E44 measured it null-to-negative, 39.2 vs twin 42.4; protection is insurance
  already baked into every run). Composition of the two real lifters (arm 1'
  substrate + lr2x/top3k) projects loss ~0.070-0.080 and front-5 ~44-49; 70 requires
  dual-cycle chunk ~0.03-0.07, and the one place with compass-measured headroom of
  that size is the un-adapted image span (VLM-only dense LoRA = 0.030 on e4 vs our
  0.099; span attribution: image tokens carry 43-64% of dL/dh). 10-task PARKED (Josh:
  no point reaching for 10 if we can't nail 5).
- **steps/task stays 5k for the composition arm** (7k considered, rejected): E41
  measured staged blocks converging by ~2.5-3k (the steps7k arm's endpoint function
  matched every 5k arm; "5k->7k retired"), 2x LR moves the knee EARLIER, and all
  three new arms show the converged signature (block-end ~11% above block-min =
  oscillation band). Revisit trigger: t0 block still descending at 4.5-5k.
- **Two VM arms scripted + shipped (commit 6edb54e7), launch via Josh's local claude**
  (VM aliases: nebius3 = the arm-3-old box, gets the r4 arm; nebius4 = the arm-3' box,
  gets the composition arm; A-checkpoint staged there):
  1. seq5_arm1p_lr2x_topt3072.sh — composition, sequential-only from arm 1's existing
     A checkpoint (lr 2e-3->2e-4, top_t 3072).
  2. grad_arm1p_vlmr4.sh — VLM rank 4 via graft_vlm_rank.py: the warm-up trains
     keys/proj only and frozen-route makes routing values-independent, so the r2
     certificate transfers WITHOUT re-warming; torch raises on shape-mismatched
     tensors even at strict=False, so the graft drops the 4 r2-shaped VLM slot tensors
     (fresh r4 init; slot_up zero => memory output 0 at start). Smoked: 4 tensors
     fresh, 34 router tensors bit-identical, 48/895 trainable. Sequential forced to
     bs16xacc2 (r4 ~135GiB at bs32 vs ~137-139 usable; accum overhead measured +12%).
  The common chain body gained SEQ_* env overrides (defaults byte-identical to E47).
- **Image-span design settled** (one history inversion corrected en route: pooling was
  never the collapse — per-token state routing was the E45 failure and the E47
  collapse was the loss DEDUP; pooled retrieval + broadcast loss accounting is the
  settled, now rollout-validated pattern): image region keys = a*nrm(pool(instr)) +
  b*nrm(pool(image region)), route-once, broadcast losses at TRUE deployment mass (NO
  per-region loss normalization — Josh; the ~94% image loss share is watched, not
  pre-engineered), write-budget allocation likewise watched. Region count + (a,b) +
  normalization constants to be FROZEN FROM MEASUREMENT (querystats-image probe, the
  E45 playbook that correctly ordered the pooled-key arms).
- **Two-step experiment plan:** STEP 1 = keep expert memory, add image-pooled routing
  at VLM [15,16] — the single-delta attribution cell vs arm 1' (1A: probe -> design
  freeze -> joint warm-up -> audit + REVIEW; 1B: A-phase + 5-task). STEP 2 = drop the
  expert tower's memory entirely (compass: VLM-only LoRA 0.030/40 vs expert-only
  0.229/14) — this voids the placement guard (it exists to protect expert routing), so
  VLM memory can go to lower layers where image leverage concentrates; freed budget =
  16GiB. Capacity split (4 layers r2 vs 2 layers r4, both budget-neutral ~122GiB;
  r4x3 marginal at bs16xacc2) decided by 1A's audit + the nebius3 r4 arm (which IS the
  rank-axis measurement on the right tower). Step 2 runs the full chain without a
  human review stop but keeps automated hard-fail tripwires. Lean: layers > rank
  (rank's record is weak everywhere tested; layers buy new routing surfaces =
  conditional capacity, the currency conclusion 2 just validated; image leverage sits
  earlier in the stack). Bonus physics: prefix memory runs 1x per 50-action chunk at
  inference (cached) vs the expert tower's 10x per denoise.
- 1A scope addition: the sub-span probe's route-once-aware row mapping (deferred since
  E47) is now load-bearing — audits force the compact path and the compact row layout
  with image keys ([region keys, state key, instr tokens]) can't be recovered by the
  constant-prefix trick.

### Next steps

1. [LAUNCHING via local claude] nebius4: composition arm; nebius3: VLM-r4 arm. Gate-2
   chunk probes on their t0 checkpoints run centrally here as they land (~0.12 kill).
2. [RUNNING here] **1A**: querystats-image probe (layers incl. candidates below 15 to
   inform step 2) -> freeze region count / (a,b) / normalization -> code (vlm span
   extension, k region keys, row mapping) + smokes -> joint warm-up (broadcast) ->
   audit with image-region gates -> review with Josh.
3. Then 1B (A + 5-task, single delta vs arm 1' 40.0/0.0940/chunk grid); then step 2.
4. Deferred, unchanged: protection-store decay before amplitude-at-scale; e9
   warm-up footprint-dispersion gate; 10-task behind nailing 5.

Artifacts: outputs/analysis/e48/{working_tables.md,slots.out,probe_conversion.jsonl,
mse_matrix.jsonl}; graft at outputs/train/libero_90_pi05_jointwarm10k_arm1p_vlmknn16_r4graft;
commits 6edb54e7 (tooling+scripts) + this entry.

---
### Entry 49 addendum (20 Jul eve) — 1A design frozen from the querystats-image probe; build shipped (smoked 15/15 + policy-level); warm-up chain RUNNING

**Probe results** (`outputs/analysis/e49/querystats_image_stage1.json`, layers 7-16, stage-1
features, libero_10): (1) patch-level between/within task variance on real cameras =
0.06-0.22 — squarely the state-digit sprawl band (state 0.11-0.13, instruction 0.81-0.99)
→ per-token image routing confirmed dead before spending a warm-up. (2) Pooled image
geometry MORE degenerate than the state pool was (family cos 0.975-0.988) and composites
degrade separation monotonically in b — the same init pattern the state pool showed,
where trained (1.0,0.5) still won; probes rank, never veto. (3) NEW: the instruction
anchor's separation is best LOW in the stack and degrades monotonically with depth
(composite b=0 inter: L7 0.722 → L9 0.785 → L11 0.820 → L13 0.869 → L15 0.840 → L16
0.898), and patch-level task signal is also strongest early — both strengthen step 2's
guard-free lower-layer placement, and mean step 1 at [15,16] tests image pooling at the
least-favorable depth (the price of single-delta discipline). (4) The two empty camera
slots are unambiguous in the stats (identical, ~15x lower variance) — excluded in-model
via img_masks. **Frozen: g2 (2x2 regions per real camera → 8 image keys/sample, 64
positions each), a=1.0 b=0.5, layers [15,16], component-RMS normalization rescaled to
the language-field token RMS.**

**Build** (commit a8b8a85f): `vlm_image_regions`/`vlm_image_pool_weights` +
`router_only_fast` (exact value-path skip at pinned-zero values — makes the literal
571-row broadcast warm-up affordable; A/seq/audit gained explicit
`router_only_fast=false` overrides per the E37 rule, without which the A-phase would
silently train nothing). Route-once compact layout [8 region keys, state key, instr
tokens] with per-region palette application; literal-broadcast path drops inactive-camera
positions so valid tokens stay a contiguous prefix for the loss machinery. Smokes S15a-h
ALL PASS — the load-bearing two: route-once↔broadcast parity 1.19e-07 at NONZERO values,
and router_only_fast bitwise-exact on both paths. Policy-level smoke on stage-1:
broadcast T=571, keys grads live through the image keys, no fallback. The sub-span
probe's route-once-aware row mapping (deferred since E47) shipped — auto-detects
image-compact / state-compact / literal layouts by row count.

**RUNNING** (tmux `imgspan`, log `outputs/e49_imgspan_warmup.log`): warm-up
`libero_90_pi05_jointwarm10k_imgspan_g2_n256_vlmknn16_bcast` — 0.87 s/step at bs32,
32 GiB (the value-skip makes the image-broadcast warm-up FASTER than the text-only
ones), ETA ~2.5h → audit → analyses → region-split sub-span → STOP for review.
Review gates (wrapper header): image-region famIoU <= ~0.25 at per-region effnum >=
~300 (no ~2-draw collapse), state/instr regions within ~20% of arm 1' (0.074/0.084
palette, ~0.20 instr), expert certificate reproduced. The audit famIoU topline is
image-mass-weighted — read the sub-span region table, not the topline.
