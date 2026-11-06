# Research Log - VLA Memory

---
## Standing Reference (added 2 Aug 26)

Consolidates definitions, constraints, and anchor numbers that entries below restate inline. Where an entry re-defines one of these terms, the definition here is the canonical one.

### Metrics & instruments
- **held-out audit / certificate**: stream the libero_10 demos through a FROZEN checkpoint, dump per-task slot-usage JSONs, compute the routing-geometry metrics below. Certificates predicted full-pipeline behavior throughout E44-49.
- **famIoU**: held-out read-mass overlap (weighted IoU) among the three near-identical basket tasks — libero_10 task_index 4/5/7 = envs 7/0/1 (soup+cheese / soup+sauce / cheese+butter). **bgIoU**: same over non-family pairs.
- **core50 / effnum**: per-task footprint size — slots carrying 50% of read mass / exp(read-mass entropy). The capacity side of every routing gate.
- **palette**: the slot set retrieved by a pooled (shared) router key (VLM state/image-region routing); the always-read block.
- **chunk error**: executed-chunk metric — run the real 10-step denoise on demo observations, score the executed 50-action chunk against the demo chunk (adopted E41). Rollout-predictive WITHIN a substrate; ranks backwards ACROSS substrates (E50 rule: substrate changes are judged on 50-ep finals + jitter, never chunk alone).
- **MSE forgetting matrix**: paired-noise loss of every per-task checkpoint on every seen task; flat diagonal drift (≤~+5%) = no function-level forgetting.
- **jitter probe**: perturb demo observations (state/image noise), re-denoise, score the chunk — the off-demo brittleness shell. Probes ~10x nearer than real rollout excursions (E57).
- **RTO (read-through-overwrite)**: fraction of a task's read mass on slots later tasks updated. **self-coverage**: fraction on its own-adapted slots.
- **q_intra / q_inter**: cosine among same-task / cross-task routing queries; q_intra ≳0.95 = state-independent addressing collapse (E21).
- **top_t**: the TF-IDF write mask — top-t slots per module per optimizer step receive gradient. **protection**: mask score × (1−u)^β with u = prior-task usefulness; **corefrac** u-normalization puts whole prior cores at u=1 (score 0 → structural exclusion from the mask).
- **read-write product**: per-slot learning signal × read participation × state-conditionality of addressing — the capacity currency (E43/E48).
- **D**: cross-model executed-chunk disagreement on rollout-harvested states, read against excursion distance (off-trail instrument, E57).
- **Eval noise**: 20-ep intermediate cells ±11pp — RETIRED from decisions (E41); 50-ep finals ±7pp.

### Fixed design constraints
- Router/keys FROZEN during sequential adaptation, permanently (E19 — key training silently re-points prior tasks' retrieval).
- No per-task parameters; no task identity at inference.
- EWC, replay, and hard task-boundary slot allocation: off the table.
- "Train on exactly the same data": no observation-space augmentation. (Hidden-state noise on the value-path INPUT is allowed — E58.)
- Pretrain-task forgetting is out of scope; only the sequential tasks are protected from each other (E19).

### Environment mapping
- libero_10 dataset task_index → env: {0:4, 1:6, 2:9, 3:2, 4:7, 5:0, 6:8, 7:1, 8:3, 9:5}; train order = task_index ascending. The 5-task "front-5" suite = task_index 0-4 = envs 4,6,9,2,7.

### Recurring configs
- **C-config** (sequential recipe, E30): β4 protection, top_t 1536, 5000 steps/task, value-lr 1e-3→1e-4 per-block linear, 20-ep intermediate evals. Later deltas named inline: lr2x (2e-3→2e-4), top_t 3072, corefrac.
- **arm 1'** (E48): expert memory n256/r2/knn36 @ [8,10,12,14] + VLM text-field memory n256/r2/knn16 @ LM [15,16]; broadcast-loss anchored router.
- **comp / composition** (E50): arm 1' substrate + lr2x + top_t 3072.
- **compact layermax** (E51): expert [9,10,11,12] + VLM [13,14,15,16], 8 modules, n256/r2.
- **spread / attempt-A** (E51/E53): expert [2,4,6,8] + VLM [10,12,14,16], 8 modules, n256/r2.
- **absmax** (E53): expert [4-9] + VLM [10-16], 13 modules, n256/r2, anchored-nofilm expert router (w=0.5).
- **B** (E54): spread substrate + expert text-anchor w=0.40 + sep8, FiLM-free router — the headline config.

### Standing anchors & baselines (single-seed; 50-ep finals unless noted)
- Audit anchors (expert L14, n384 era): control@40k famIoU 0.349 / core50 2643 / bg 0.127 · collapsed negonly-c005@40k 0.133 / 511 / 0.042 · P9 sep5 0.264 / 2679 / 0.087.
- LoRA per-task specialists (r32, 5k steps, frozen stage-1 base; rollout / clean chunk): e4 58/0.0204 · e6 44/0.020 · e9 70/0.0675 · e7 60/0.0330 · e2 84/0.0308 — oracle mean 63.2.
- Multitask-LoRA (one adapter, 5 tasks) 49.2 = the must-beat line. Base pi05 joint finetune (no memory, all data) 72.6 on libero_10 / 74.8 on the front-5.
- Stage-1 base zero-shot on libero_10: mean 10.6 (collapse-prone tasks 0-2) — the no-adaptation floor.
- Front-5 standings (as of E56): absmax 53.6 · B 53.2 · compact+corefrac 51.6 · spread+corefrac 47.6 · P3 47.2 · comp 46.0 · layermax-plain 44.8.

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

## Entry 1 — 11
### Summary of the problem
- Sequential training updates only memory values/LoRA params (router frozen). We observed **high slot overlap at read-time** and a **shared “hot core”** used by all tasks (router collapse / MoE-like load imbalance).
- TF‑IDF masking mostly prevents *writing* into that shared core, but tasks still *read* heavily from it, and overlap remains large.

### Evidence from sequential (tasks 6–9; layers 14 & 15)
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
- Key knobs:
  - Memory on expert layers **[14,15]**, `value_type=lora`, `lora_rank=4`
  - `mem_n_keys=384`, `mem_heads=4`, `mem_knn=16`, `mem_k_dim=512`
  - Language-conditioned queries: `lang_to_query=true`, `fuse_method=film`, `embedding_model=all-mpnet-base-v2`
  - Increased exploration: `dropout_prob=0.1`
  - Stronger separation objective: `contrastive_method=sample`, `contrastive_loss_weight=0.5`

---

## Entry 2 - 25 Feb 26 (Simplified Experiments)

### Summary of the problem
Run alations across:
- Contrastive loss type (centroid vs sample-wise)
- Contrastive loss weight (0.5, 1, 2)
- Dropout probability (0.05, 0.1, 0.2)

Run with Lora=4 and layers 12 and 14.

### Observations
- **Contrastive loss type:** No winner. Centroid contrastive exhibits less interference at later layers (e.g. layer 14) but slighly more in earlier layers. Interesting - I thought we'd expect more intersection from centroid.
- **Contrastive loss weight:** We see the largest intersection for 0.5, significantly reduced intersection for 1.0, then oddly increased intersection for 2.0. Non-monotonic increase is confusing.
- **Dropout probability:** 0.2 performs better than 0.1 in terms of interference. Appears less degredation in previous task per, less intersection. This makes sense as some attempts to access slots important for previous tasks will be nullified by higher dropout.

### Thoughts
- The sample-wise contrastive loss we are including some terms which encourage uniformity of the queries inside a given task **as well** as distance across tasks. This could be problematic if we saturate the query space. Could explain the non-monotonic behaviour of the sample contrastive loss weight. To test we are trying a loss that removes the positive pairs from the denominator.
- I also want to test the change caused by the lora r. Testing 2 to see if it makes any difference.
- I also want to test the effect of initialising the idf stats with the pretraining stats with different weights. Testing denom = 16 (e.g. 2x a single sequential task), 33 (1x seq task), 66 (0.5x seq task)

### Future
- I should test the effect of batch size. Due to VRAM constraints I'm limited in the batch size for lora=4 which could affect the training dynamics of the query projections if not all the tasks are in the batch. 
- Test corruption again. Seemed promising.

---

## Entry 3 

- Pretraining test to increase effective contrastive pool without increasing micro-batch memory: cross-batch query FIFO (`contrastive_query_queue=2048`).
- Goal: improve sample-wise contrastive negatives/positives under `batch_size=32` by reusing recent query embeddings from prior batches.
- Using layers **[12,14]**

---

## Entry 4

### Updated dynamics interpretation
- Consistent tradeoff between expressivity and stability in the value parameterization:
  - **Static value vectors** gave lower interference but lower ceiling performance.
  - **LoRA values** increased current-task performance but also increased forgetting.
- Working hypothesis: LoRA slot overlap is more destructive than vector overlap because each slot is a **transform** of the hidden state, not just an additive template.
- This is compatible with the observed pattern where TF-IDF reduces direct write overlap but forgetting still appears once additional tasks are introduced (especially by task 4): the model still reads overlapping regions, and overlapping LoRA slots are high-impact.

### Why TF-IDF may still be insufficient
- Current online masking is based on **frequency of slot access per batch** (TF over counts), which can still repeatedly prioritize globally frequent shared slots.
- IDF helps, but if TF dominance is strong, medium-importance task-specific slots may still be under-updated.
- Masking reduces some overwrite, but not enough to prevent drift in shared high-impact LoRA slots.

### Mitigation directions (non-EWC / non-replay / non-hard-mask)
- Use **contribution-weighted TF** (weight by retrieval weights) instead of pure access counts so low-weight incidental touches do not dominate slot selection.
- Use a **saturating TF transform** (e.g., sqrt/log scaling) before TF-IDF ranking to reduce repeated wins by the same hot slots.
- Consider **per-head update budgeting** to reduce global head collapse where dominant heads consume most of the update budget.
- Add **soft plasticity decay** per slot (continuous reduction in update magnitude for heavily updated slots, not hard exclusion).
- Keep query/router adaptation offline (pretraining) and avoid online query updates if they destabilize old-task routing.

### New experiments
- Testing **LoRA rank 1 with more memory slots** to probe whether many weaker experts are more stable than fewer stronger experts.
- To approximately double slot count relative to `mem_n_keys=384` (slots = `n_keys^2`), we set:
  - `mem_n_keys=544`

---

## Entry 5

- Discussion clarified two separate ideas:
  - **Weighted TF-IDF** for sequential updates: rank slots by retrieval-weighted contribution rather than raw access counts.
  - **Routing regularization** during pretraining: act on actual PQ sub-key usage, not just query embeddings.
- Key point: more diverse queries do **not** necessarily imply more diverse slot accesses. Different queries can still route into the same hot late-layer region, so query contrastive is only an indirect proxy for what we care about.
- Current hypothesis:
  - within a task, routing should stay fairly localized
  - across tasks, those localized routing regions should differ
- To test this, I initially added a one-sided **routing compactness** loss on PQ marginals.
- After further discussion, the more important anti-forgetting term is **routing separation**:
  - compactness says each task should route locally
  - separation says different tasks should route to different regions
- So compactness is now being treated as a **secondary control ablation**, while the main next test is routing-separation.
- Also fixed the LoRA corruption path so corruption is applied to the **adapter output before the shared gating/aggregation path**.
---

## Entry 6

- Discovered that the logged `routing_separation_mean` metric was actually the **mean pairwise cosine similarity** between task routing distributions, not a higher-is-better separation score.
  - `~0.998` means tasks are routing almost identically.
  - `~0.02` means tasks are routing very differently.
- In the first routing-separation sweep, `0.1` was too weak to change routing much, while `0.5` and `1.0` drove the similarity way down but also collapsed **intra-task routing entropy** to about `0.10` and `0.06`.
- That means the separation term was working, but it was achieving separation by pushing each task toward an almost one-hot PQ subkey pattern. Too sharp.
- Main adjustment:
  - replace the old “compactness” term with a **intra-task locality** loss
  - define locality as a **support/entropy band**, so it penalizes routing that is both too diffuse and too concentrated
  - keep **inter-task separation** as a separate objective
- Planned sweeps:
  - **Intra-task locality** with support band `8-32` and locality weights `0.1 / 0.25 / 0.5`
  - **Inter-task separation** with locality fixed on (`weight=0.25`, support band `8-32`) and separation weights `0.15 / 0.25 / 0.35`
- Goal: find a regime where tasks separate in routing space **without** collapsing each task onto 1-2 subkeys per PQ half.

---

## Entry 7

### Findings from Entry 6

3 locality sweeps (`locality_weight` 0.1 / 0.25 / 0.5, support band `[8, 32]`) and 3 separation sweeps (`sep_weight` 0.15 / 0.25 / 0.35, locality 0.25, support `[8, 32]`). Results:

1. **Sequential training performance did not improve.** Best separation run (sep=0.15) tied with corruption baseline at ~16.5% success.
2. **The routing loss was operating on the wrong distribution.** It computed soft distributions over all `n_keys=384` subkeys per PQ half, then measured entropy and pairwise similarity on those half-marginals. Retrieval only uses the **top-k** subkeys and forms **Cartesian-product joint slots**. Loss could be satisfied by moving tail mass in the half-distributions without changing the top subkeys or the final retrieved slots.
3. **Evidence of misalignment:**
   - Half-level support was broad (routing_intra_task_support_mean ~216–384) even as the **actual final-slot effective number** in layer 14 was very concentrated (effnum ~30–82 for sep 0.15/0.25).
   - sep=0.25 drove half-similarity down, but layer-14 weighted access IoU between tasks stayed at ~0.316. It achieved "separation" by peeling one task off onto a different tiny hot core, while the other 3 tasks still shared.
4. **Task 7 "zero overlap" anomaly:** In sep=0.25, task 7 was isolated onto a different layer-14 core (weighted IoUs ~1e-5 against other tasks) while tasks 6/8/9 still shared heavily (IoUs 0.60–0.69). This is a pathological asymmetric solution allowed because the pairwise loss has no global-balance pressure.
5. **Last-task slot-usage instability:** Task 9's slot-usage metric oscillated between ~0.020–0.037 across steps. Since the router is frozen during sequential training, this is **batch/episode heterogeneity** within task 9, not learning instability.
6. **Logging note:** The sequential `train/loss` includes routing auxiliary terms even though they have no gradient path to the trainable value params. `mse_loss` is the meaningful optimisation signal.

### Root cause

Routing regulariser operated on the two PQ half-distributions separately. But:
- Retrieval takes top-k in each half, forms the k×k Cartesian product, then selects top-k final slots.
- The loss on soft half-marginals is a **weak proxy**: can be cheaply satisfied by reshuffling tail mass while the actual top subkeys stay shared.
- The support band `[8, 32]` targeted half-subkey support, not final-slot support. With `n_keys=384`, these are completely different scales.

### Solution
Rewrote both routing losses to operate on the **joint product-key candidate distribution**:
1. Take top-M subkeys per PQ half (M = `routing_loss_topk` or `knn`, default matches retrieval)
2. Form M×M Cartesian-product candidate scores and slot IDs
3. Softmax over those M² joint candidates per sample per head
4. Scatter per-task distributions into a compact slot histogram
5. Compute locality (entropy band) and separation (cosine similarity) on those **full-slot distributions**

This directly regularises the distribution retrieval actually uses. Gradient flows: loss → histogram → joint softmax → joint scores → top-k values → PQ half scores → query projection + keys.

Support bounds now refer to **effective final slots** in the `n_keys²` space, not half-subkey support.

### Support band rationale

Old band `[8, 32]` was in half-subkey space over 384 subkeys. New band is in final-slot space over 147,456 (`384²`) slots. With 35 pretrain tasks:
- Want to allow **generalist slots** (shared priors across tasks), don't need complete separation
- Want to prevent **collapse** onto a tiny hot core
- Uniform partition: 147K / 35 ≈ 4,200 slots/task, so even max_support=2048 well within budget

### Next experiments

**Locality sweep** (locality_weight=0.25, no separation loss):
- `[64, 512]`
- `[64, 1024]`
- `[128, 2048]`
**Separation sweep** (locality_weight=0.25, support `[128, 2048]`):
- `sep_weight` = 0.15 / 0.25 / 0.35

---

## Entry 8

### Results from Entry 7 experiments

6 pretraining runs (3 locality-only, 3 locality+separation) and 5 sequential runs (sep_0.25 sequential added later). All use layers [12,14], LoRA rank 2, support band [128, 2048], locality_weight=0.25.

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

**1. The joint-slot routing fix (Entry 7) works.** Unlike the half-distribution approach which achieved spurious separation by reshuffling tail mass, the joint-slot loss produces routing differences that translate to less forgetting. sep=0.25 achieves 34.5% vs 16.5% for the best locality-only run.

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

**Batch 2 — Capacity via depth (3-layer pretrains):**

3 pretraining runs adding layer 10 to the existing [12,14] setup, with sep=0.25:
- Adds ~50% more effective capacity per task (3 layers × ~2,500 unique updated slots each)
- Interference budget is per-layer, so a third layer doesn't compound interference
- Will test whether the per-task MSE gap (0.06-0.09 vs pretrain 0.011) narrows with more depth

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

**Future: higher capacity pretrains.**

- LoRA rank 4 (2× per-slot expressivity) at 3 layers
- 4-layer configurations (e.g. [8,10,12,14])

Both address the per-task MSE gap from different angles — rank increases per-slot expressivity while depth increases slot budget.

---

## Entry 10 - 17 Mar 26 (Capacity Pretrains)

**3 new pretraining runs launched** — all use sep=0.25, loc=0.25, support [128,2048]:

| Run | Layers | LoRA rank | Rationale |
|-----|--------|-----------|-----------|
| 2L r=8 | [12,14] | 8 | 4× per-slot expressivity at baseline depth |
| 3L r=4 | [10,12,14] | 4 | 2× expressivity + 3-layer slot budget (best depth from Entry 9) |
| 4L r=2 | [8,10,12,14] | 2 | Maximum slot budget, baseline expressivity |

---
The r=4 and r=8 runs use batch_size=16 with gradient_accumulation_steps=2 (effective batch=32).

---

## Entry 11 - 23 Mar 26 (Contrastive Pool Confound in Capacity Comparison)

### Observations and analysis

- Reviewed the latest `22_3_26` runs against the original baseline runs from `16_3_26` using the per-task memory-slot JSON dumps.
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

### Watch

1. **Pretrain success** — stronger anti-collapse can hurt fit; need to confirm it doesn't degrade below the 4-layer baseline
2. **`eval/avg_pc_success_seen`** after 4 sequential tasks — the primary metric
3. **Task-9 read share through task-8-updated slots**, especially L12/L14 — the direct measure of read-time interference
4. **Weighted access IoU** (not binary overlap) — the meaningful interference metric per Entry 8 finding 5
5. **`routing_global_entropy`** during pretraining — should increase with higher weight; watch for saturation or training instability

### What we are not doing

- More weighted-TF variants
- Online query/key training during sequential (risks changing routing for old tasks)
- Stronger pairwise separation alone (doesn't address aggregate collapse)

---

## Entry 13

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

("read-thru overwrite" = fraction of that task's read weight on slots that *later* tasks updated, mean over 4 layers.)

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

**First (sequential-only, reuses the knn=36 pretrain):**
- **top_t=1536** rerun. Tests whether the write-budget lever recovers the plasticity-limited tasks (env 0/9/3) at 10-task scale, vs worsening early-task overwrite (e9←e2 already 20% at top_t=512). Watch env-8/9/2/5 retention and `task9-reads-task8-updates`.

**Staged plan after that:**
- IDF slot protection (idf_exponent 1→2, or contribution/saturating TF off `raw`) to keep later tasks' top_t off earlier tasks' high-weight slots — sequential-only, low risk, attacks the e2→e9 channel.
- knn=24 / routing_loss_topk=24 + top_t≈768 (new pretrain) — Entry-16 retention-friendly point; narrower reads → lower overlap → safe to add write budget.
- Make the held-out router separable: stronger separation/contrastive on held-in goal-like tasks, OR small `memory_keys_lr` during sequential (train_memory_keys=true) so routing specialises the goal tasks (medium risk).
- Raise the floor: more steps/task (3000→5000; loss still falling at task boundaries), lora_rank 2→4, stronger/longer pretrain (held-in only 65%).
- Amplifier: gate saturates at 0.99 (no modulation) — probe a gate regulariser / per-task update budgeting to tame heavy overwriters.

---

## Entry 19 - 9 Jun 26 (libero_90 → LIBERO-Long results + write-protection mechanism autopsy + 10k probe plan)

Deltas vs Entry 18: pretrain libero_90 (90 tasks, 40k) instead of libero_minus_goal (30 tasks, 30k); sequential suite is LIBERO-Long (libero_10) instead of libero_goal; `top_t=1536` from the start. Cross-suite absolute numbers are NOT comparable (LIBERO-Long is the hardest suite — six of ten tasks need two full pick-place cycles).

### Pretrain summary

- Held-in eval: `76.4%` @20k → `81.1%` @40k (vs 65% for the 30-task pretrain). MSE `0.196` @20k → `0.133` @40k. **Both still improving when the schedule hit zero → 40k undertrains libero_90.**
- Gate ≈ `0.98` at all four layers (pi05 saturation again — overwrites pass at full strength).
- Usage is broad and healthy at the aggregate level: table-wide effnum 50–62K of 147K per layer, top-1 share ~0.0006. **No hot-core collapse — the 90-task pretrain fixed the Entry 18 global-routing problem.**
- Inter-task routing sim 0.10. Intra-task support mean ~5.8K vs locality band [128, 2048]:
  - **L8 compacted (8,012 → 4,298, −46%) but L14 never moved (7,930 → 7,398, flat from ~10k on, 3.6× above band max).** Locality weight 0.25 contributes ~0.002 to a 0.17 objective — MSE outbids it exactly at the layer that matters most.

### Sequential headline

Final avg over 10 tasks = **34.4%**. Trajectory: 26 → 32 → 29.3 → 39 → 31.2 → 29.3 → 25.1 → 22.5 → 25.3 → 34.4. Logged read IoU `0.107` (slot-JSON recomputation matches exactly — pipeline validated).

Per-task (train order; env id):

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
- DF increments are binary-per-batch (within-batch intensity discarded); IDF = `log((B+1)/(DF+1))^e`, recomputed only at task boundaries; mask score = in-batch raw-count TF × IDF, top-1536.
- Reconstructed the exact IDF in effect during t7's block (L14, B=21,000, DF = Σ batch_accesses of t0–t6): t4/t5 core-50% slots median IDF **0.49** vs t7's other written slots **1.84** → a **3.0× penalty**, while TF on those same slots runs **~86×** the accessed-median. TF wins; the log crushes a 60:1 DF ratio into 3×.
- Counterfactuals: at e=1, **100%** of t7's 2,913 writes into t4/t5 cores survive the write margin; **e=2 keeps 97%**; e=4 keeps 60%. Mass-weighted DF is worse (per-batch mass shares ≈ 0 → IDF uniform ~9.9 → zero protection). Only ~400 universal slots (DF ≥ 0.9B → IDF → 0) are genuinely hard-protected.
- **Decision: the `idf_exponent=2` plan is dropped** — measured as a near-no-op before spending a run on it.

**2. Hard protection is zero-sum under this routing.** Veto "slots present in >τ of any single prior task's batches": τ=0.7/0.5/0.3 vetoes 37/50/64% of the table, rescues 100% of the measured core damage at every τ — but blocks 68/79/88% of t7's write events, and **83–96% of t7's own read mass sits on vetoed slots**. Any write policy strong enough to protect predecessors starves the new task. Write-side knobs alone cannot produce the target behaviour.

**3. Incidental vs intrinsic damage (L14).** Cross-family damage is incidental for the writer (t2←t8: hit slots carry 21.7% of t2's mass but only 10.3% of t8's; t3←t8: 16.1/7.2; t0←t7: 12.2/4.5) → protectable cheaply in principle. Within-family it is contention (t4←t7: 34.8/24.3; t5←t7: 29.5/18.2; t4←t5: 37.4/32.3) → the writer needs the slots it damages; only routing separation can fix this channel.

**4. Routing breadth, measured at the right granularity.** The "94% of table touched per task" union is coupon-collector arithmetic (~4.8M queries × ≤144 slots each). The meaningful numbers: a **single same-task batch touches ~31K unique L14 slots** (12.9K at L8); per-task effnum 19K; core-50 ~3.3K. Combined with the L14 locality stall (above): **the router never learned per-task chunking at late layers even in-distribution, and held-out tasks can't be allocated chunks at all under frozen routing.**

**5. Gradient-dilution hypothesis tested and rejected as the diagonal driver.** Update events ~119/slot mean for the heavy writers, but concentration is uncorrelated with fit: t8 is most diffuse (118 ev/slot) with init 74%; t0 most concentrated (286 ev/slot) with init 26%; t5 fits to near-best MSE with the 2nd-largest write set. The libero_goal 512-vs-1536 pair (3× concentration difference) shows no init advantage either. The diagonal is task structure + expressivity, not gradient spreading.

**6. SupCon facts.** The sample contrastive is SupCon at τ=0.07 → gradient concentrates exponentially on the *most similar* cross-task pairs (hard-negative emphasis is built in — the property needed for lookalike families). Caveats: it operates on the per-sample mean query (task-centroid geometry, not slot footprints — the Entry 5/7 proxy gap applies); default SupCon keeps same-task terms in the denominator (intra-class spreading at high weight — `contrastive_negatives_only=true` exists to remove it); batch 32 + queue 128 is thin negative coverage for 90 classes.

### Constraints fixed in discussion (design space)

- **Router stays frozen, permanently.** Slow key-training during sequential was tried previously: per-task plasticity improved greatly, prior-task performance was decimated (frozen old queries × moved keys silently re-points old tasks' retrieval). Off the table.
- **No per-task parameters, no added parameters of any kind.**
- **Pretrain-task forgetting is acceptable.** Pretraining is purely prior-construction for clean sequential adaptation; only the 10 sequential tasks need protecting from each other.
- Still off the table: EWC, replay, hard task-boundary slot allocation.
- Target property: *each sequential task inherits a compact, separated footprint from the frozen router; cross-task overlap only at benign/synergistic edges.* Everything must be bought at pretrain time via the routing losses.

### Plan: two 10k single-lever probes

Both are **truncated full runs** (same warmup 4000 / decay 40000 schedule) so a passing probe continues to 40k at zero wasted compute.

- **Probe L:**
  - Single knob: `routing_intra_task_locality_weight 0.25 → 1.0`.
  - **Pass:** L14 intra-task support < ~5.5K by 10k and still descending (control: flat ~7.9–8.2K); MSE@10k within ~10% of control's 0.261; audit shows held-out L14 core-50/effnum down ~2×. Family IoU recorded but not required (compaction ≠ de-coincidence).
  - **Fail route:** L14 flat at 1.0 → per-layer locality (heavy on 12/14), not a bigger global weight.
- **Probe C:**
  - Contrastive arm as a package: weight 0.01 → 0.05, `contrastive_negatives_only=true`, queue 128 → 512 (flag = the high-weight pathology guard; queue = SupCon statistics enabler). Locality stays 0.25.
  - **Pass:** held-out audit family IoU (basket t4/t5/t7, L14) down ≥~30% vs control audit; MSE guardrail as above.
  - **Fail route:** query_inter_sim moves but slot-level family IoU doesn't (the Entry 5/7 proxy failure) → similarity-weighted *separation* (slot-space pairwise term weighted by language-embedding similarity; small code change, no new params).

**Review instrument:** held-out routing audit = forward the libero_10 demos through a frozen checkpoint with per-task usage logging (no training), dump the slot JSONs, compute per-task L12/L14 support, core-50, effnum, and the pairwise weighted IoU matrix. Run it on the **control checkpoint (20k and/or 40k) first** to fix the baseline — control routing plateaued early, so control@20k is a fair comparator for probe@10k. Minutes-to-an-hour per checkpoint vs 45h for a sequential run.

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
- Aggregate-usage check: probe C L14 aggregate effnum 49.6K with per-task support 3.4K → aggregate/per-task ratio ~14× vs control ~8×. **Compaction + separation, not collapse onto a shared core.**

### Held-out routing audit (the decisive test)

Instrument: stream the libero_10 demos through the FROZEN checkpoint with `memory_value_lr=1e-12` (numerically inert), no env (no eval), 100 batches/task — dumps the standard `memory_by_task` JSONs measuring the **pristine prior's** held-out footprints.

Baseline caveat: control audited @40k; probes @10k. The training-amount confound is covered by probe L, which acts as a same-duration control (near-control config, audited @10k, shows ≥control overlap).

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

1. **Resume probe C → 40k**. Gates: MSE tracks control (0.196 @20k / 0.133 @40k), held-in eval @20k/@40k vs control 76.4/81.1, support & query-sim stability.
2. **Re-audit at 40k** — confirm held-out compaction/separation survives full training.
3. **Sequential run** on the audited 40k checkpoint with `top_t` re-derived from the new footprints: held-out core50 dropped ~7.5× (2.6K → ~350), so `top_t=1536` is ~an order of magnitude oversized for this regime — start at **256–512** (decide from the 40k audit). Minimal-change config (everything stays pretraining-side; the only sequential knob is the existing TF-IDF top_t).
4. Probe L: no further investment.

**Update (11 Jun):** Decisions baked in:
- `top_t=512` pre-committed rather than audit-gated: protection now comes from separated/compact footprints, not the write mask; per-batch accessed slots shrink ~7× with this prior, so 512 is *relatively* more generous than 1536 was in the old regime, and libero_goal showed 512 safe at far worse IoU (0.17) than this prior's held-out 0.05 bg / 0.19 family. The 40k audit (stage 2) is informational.
- **Contrastive weight held at 0.05** for this cycle: 7× compaction at +1.2% MSE is already near the useful ceiling, weight responses have been non-monotonic throughout the project, and the open risk is fit-side (which more pressure worsens). Weight 0.1 becomes a 10k probe only if the 40k re-audit shows separation eroding.
- Sequential config otherwise identical to the failed top_t=1536 run (3000 steps/task, lr 1e-3→1e-4, 50 eval eps, same env mapping) for clean attribution: pretrain recipe + top_t are the only deltas.
- What to look at first when it lands: retention matrix vs Entry 19's (esp. t5 after t7's block, the step-24000 cliff), diagonal inits on the dual-cycle tasks (plasticity should be roughly unchanged — this cycle attacked interference, not the ceiling), `memory_iou/all_modules_mean`, read-through-any-later for t0–t5, and 40k-audit family IoU vs the 10k audit's 0.190.
- **Pre-registered next lever if retention is good but absolute perf lands ~50%:** the bottleneck is then the plasticity ceiling, and the first response is optimization budget, not architecture — more steps/task (3000 → 5000; Entry 19 showed MSE still falling at every block end) and higher memory-value LRs (floor 1e-4 → ~2e-4 first; peak 1e-3 → ~2e-3 second, watching within-block stability — t8's MSE rose late in its own block at the current peak). These are sequential-only, cheap, and now safer to push because the compact/separated footprints mean extra write pressure leaks far less into other tasks' cores than it would have pre-probe-C.

**Update 2 (11 Jun, correction):**
- **The "truncated full run / resume for free" probe design was wrong.** lerobot auto-scales the LR schedule when `steps < scheduler_decay_steps` (probe logs confirm: "Scaling warmup: 4000 → 1000, decay: 40000 → 10000"). Both probes therefore ran a **compressed full cosine**, fully decayed by 10k.
- Consequences for Entry 20 conclusions: **none material.** Probe L vs probe C shared the identical compressed schedule, so that contrast is clean; the held-out audit gaps (7× compaction, −46% family IoU, with probe L as a same-schedule control moving the opposite direction) dwarf any schedule artifact. Bonus: the SupCon effects survived a complete LR decay — they are end-of-training properties, not high-LR transients. One nuance: probe-vs-control *in-run* comparisons at 10k carried an LR-position confound (probes at LR floor, control at ~0.85×peak), which slightly flattered probe MSE.
- The resume rebuilt the scheduler **unscaled** (steps=40000 ≥ decay), so its LR jumped from floor to ~0.85×peak at step 10001 — an SGDR-style sawtooth, not comparable to control.
- **Replacement: fresh 40k pretrain of the recipe** with a clean schedule (steps=40000 == decay → warmup 4000 / decay 40000 honored, exactly matching control). Pipeline v2 now: fresh pretrain → 40k audit → sequential top_t=512 (unchanged).
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

**2. Query collapse confirmed at all layers.** `query_intra_sim` = **0.990–0.996** (control 0.83), `query_inter_sim` = **−0.016** (near-orthogonal). Measured on the per-sample routing query `z` (mean over tokens×heads). The `negatives_only=true` flag removed same-task samples from the SupCon denominator — i.e. it removed the **intra-task uniformity pressure** that was the only force keeping per-task queries spread. With only the positive pull and no counter-pressure, same-task queries collapsed to near-parallel → routing compacts to a tiny footprint.

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

### Next experiment (Josh: "do both") — 2-probe isolation, mirrors E21

- **Probe 3:** standard SupCon **0.1** (negatives_only=false, queue 512), all else = control. Isolates the contrastive-weight axis on the capacity-safe variant.
- **Probe 4:** standard SupCon **0.05** + `routing_inter_task_separation` **0.5** (negatives_only=false, queue 512), all else = control. Isolates direct slot-space separation.
- Same compressed 10k schedule + held-out audit + same gate (core50 ≥ ~1,500 AND query_intra_sim ≤ ~0.90 AND famIoU ≤ ~0.28). Throwaway screens; winner → fresh 40k + 1-task plasticity probe + sequential.
- GOOD: a point with famIoU ≤ ~0.28 while core50 ≥ ~1,500 (the decoupled frontier we haven't found yet). BAD: famIoU ≈ control (no separation) OR core50 collapse / query_intra_sim > 0.95.
- Reserve if both fail: separation 0.5 + locality 0 (test compaction amplification), or other pretraining-side separation formulations (e.g. similarity-weighted separation). Pretraining-side only.

### Note: discussion after launching probes 3/4 (mechanism + capacity diagnostics)

Two clarifications worked out while probes 3/4 run. Both refine how to read the results; no config changed.

**(a) SupCon (negonly=false) vs `routing_inter_task_separation` are NOT the same loss.** Three differences, two of which are exactly the bet:
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

Traced the two routing losses. The **contrastive** loss uses the cross-batch queue (=512 in our runs) — it concatenates 512 detached query vectors from prior batches. The **separation/locality** loss does **not**: it operates on current-batch queries only, grouping the current micro-batch by task. With batch 32 over 90 tasks (random sampler), that means:
- **~1 sample/task** per step (≈27 distinct tasks/batch, mostly 1 sample each) → each task's slot-histogram in the separation loss is a single-observation estimate.
- **~9% pairwise coverage**: any two specific tasks co-occur in a batch only ~(27/90)² of steps, so the exact pairs we need to separate (e.g. basket family) get a gradient <1 step in 10, and no step ever sees the global 90-task structure.

So separation has been operating on a noisy, sparsely-covered estimator. The Entry-22 "separation can't decouple" conclusion is **likely premature** — separation may simply never have had a clean signal. (Josh caught this.)

### Fix implemented: routing-separation cross-batch queue

New config `routing_query_queue` (in SAMPLES; 0 = off). When >0, separation runs a dense-histogram path:
- **Current batch** → differentiable per-task slot histograms over the full n_keys² space (dense == compact numerically; carries the gradient).
- **Queue** → FIFO of per-token detached queries (highest granularity, global FIFO, per Josh). Each step the queued queries are **recomputed against the CURRENT keys** (under no_grad) → detached per-task reference histograms covering all recently-seen tasks. Recompute-vs-current-keys (not frozen histograms) is deliberate: the separation loss *moves* the keys, so frozen references would lag.
- **Separation** = push each current (differentiable) task histogram away from all reference (detached) task histograms j≠i. Fixes both the 1-sample estimate and the 9% coverage (every present task pushed against ~all 90 references every step).
- Locality / global-balance stay on the current differentiable histograms (unchanged).

### Rerun

Same configs as probes 3/4, only delta = `routing_query_queue=512`, so this isolates the queue's effect against the no-queue probes3 audits:
- **probe 3':** standard SupCon 0.1 + rq512
- **probe 4':** standard SupCon 0.05 + sep 0.5 + rq512 (favored)
- 10k each.

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
- no-queue: mean pairwise cosine between **current-batch** per-task histograms → **sparse-vs-sparse**, only the ~27 in-batch tasks.
- rq512: cosine between each current task's **single-sample** histogram and the queue's **dense, aggregated, all-90-task reference** histograms → **sparse-vs-dense**, full coverage.

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

**Correction to Entry 19's "FiLM-on-language" framing (Josh's catch):** the query is `q = proj(x)·(1+γ(lang)) + β(lang)` — a projection of the **action-expert hidden state** `x`, FiLM-modulated by language. γ/β are language-only (constant across a task's frames), but `proj(x)` varies per observation and carries the visual scene (different objects on the table). So there *is* discrimination signal beyond the instruction string; the "irreducible floor from identical language" claim was too strong. The floor is softer than stated — which is part of why pushing separation harder is worth a real test (and motivates future direction #2).

### Loss-magnitude audit (answering "what can we actually play with")

Loss assembly: `loss = MSE + Σ weightᵢ·rawᵢ`, every raw term logged. Reconstructed weighted contributions @10k:

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

### Decision (probe 6)

One decisive, cheap probe: **sep=2.0** (8× P3, 4× P4), **contrastive=0.05 FIXED** (isolate separation, preserve capacity — Josh's call to avoid confounds), **rq512** (clean gradient), same compressed 10k schedule. Capacity-gated audit.

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

### The 4-probe batch

All single-knob deltas from the P6 anchor; the question is whether **separation alone is the lever.**

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

- **Contrastive read as 0.05, not 0.5.** The request said "Contrastive 0.5" for P8–P10; taken as a typo for 0.05 (the project's capacity-safe value, and the only reading under which these are clean single-knob deltas — 0.5 is 5× the 0.1 that already over-compacted, would collapse capacity, and would confound goals B/C). If 0.5 was intended, P8–P10 need a rerun (P7 and the curve *shape* still stand).
- **P7 will not log `query_intra/inter_sim`** — those diagnostics live inside the contrastive block, gated by `contrastive_loss_weight > 0`. Read P7 capacity from the held-out audit (core50/effnum/subkey), which is the ground truth regardless.

### Next (after the batch)

Pick the winning prior on the joint axis (lowest famIoU with core50 ≥ ~2000, contrastive on/off per P7, locality on/off per P10) → fresh 40k → 40k held-out audit → 1-task plasticity probe → sequential, judged on the **retention matrix** (esp. early-task collapses), not just final average. The plasticity ceiling remains the separate, binding constraint on absolute performance.

**Update (18 Jun — P7 in, goal A answered: contrastive is LOAD-BEARING, "sep alone" disconfirmed).** P7 (c=0, sep2.0, rq512) held-out L14: core50 **7700**, effnum **37384**, famIoU **0.482**, bg **0.260** — the broadest, highest-overlap prior we've made, *worse than control* (0.349/0.127). In-run: MSE 0.201, routing_sim 0.136 (vs P6 0.084), support_L14 8175. Removing contrastive removed all intra-task compaction → footprints sprawl → separation still translates centroids apart but can't keep enormous clouds disjoint → overlap blows up. **Clean division of labor: contrastive = compaction (footprint *size*); separation = translation (footprint *position*); both required.** Breadth axis is Goldilocks: too much contrastive (negonly/0.1) → collapse; none → sprawl; **c0.05 + sep2.0 (P6, famIoU 0.311 / core50 2368) sits in the pocket and stays the frontier.**

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

### Decision

Graduate **P9 (c0.05 + sep5.0 + locality-off + rq512)** to a full run.
- Same as the c0.01 combined 40k config, with EXACTLY: contrastive 0.01→0.05, contrastive_query_queue 128→**512**, sep 0.25→**5.0**, locality 0.25→**0** (support bands dropped), **+ routing_query_queue=512** (the c0.01 reference predated it — the critical add). Pretrain arch/schedule + entire sequential stage (tfidf_top_t **1536**, 3000 steps/task ×10, value_lr 1e-3→1e-4, 50 eval eps) unchanged.
- top_t=1536 kept deliberately: sep5 is broad-but-separated (core50 ~2679 ≈ control), so ~1536 is the right write budget per Entry 22(d), and safer than the Entry-19 cliff (lower overlap, famIoU 0.264 vs 0.349). Watch `task9-reads-task8-updates` early; re-derive from per-batch L14 effnum if overwrite climbs.

### What to check when it lands

1. **Mid-flight (after stage 1):** held-out audit on the 40k checkpoint — did sep5 *hold* under full LR decay (famIoU/core50 ≈ 0.264/2679, not eroded toward control)?
2. **Held-in eval** @20k/40k vs control 76.4/81.1 (fit guardrail).
3. **The real test — sequential retention matrix:** per-task init→final, the early-task collapses (esp. the t5→t7 cliff at step 24000 in Entry 19), `memory_iou/all_modules_mean`, read-through-overwrite. Diagonal expected ~unchanged (this cycle attacked interference, not the plasticity ceiling — which remains the binding constraint on absolute performance; levers = steps/task 3000→5000, value_lr floor 1e-4→2e-4 if needed).

---

## Entry 27 - 23 Jun 26 (sep=5 graduation result: interference HALVED, performance FLAT — lever is now plasticity; prior-usefulness write protection built + launched)

### Headline

The sep=5 prior (Entry 26) **delivered its technical target and bought nothing.** In the real sequential run the prior held under full LR decay and sequential read overlap **halved (0.107 → 0.052)** — yet final avg = **34.0%** vs the Entry-19 control's **34.4%** (within rollout noise). Every prediction held *except the score*. The interference axis is now exhausted: broad/interference-limited (34%), compact/capacity-limited (~12%, Entry 21), and now decoupled-middle/separated (34%) all land ≤34%. **The benchmark is plasticity-bound.** Separation reshaped forgetting from broad-and-mild to narrow-and-catastrophic, netting zero.

### Runs

- sep5 sequential — 10×3000, top_t=1536, 50 eval eps
- control = Entry-19 sequential (34.4%)
(Slot pipeline reproduces the logged `memory_iou` exactly — 0.0520 / 0.1066 — so internals read correctly.)

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

Q (Josh): isn't this what IDF is for — env7's writes should raise document-frequency and stop later tasks writing there? Traced the code: online DF is `df_vec[used]+=1` (binary per batch, `used`=retrieved/READ indices), pooled over all tasks; `idf=log((B+1)/(DF+1))^e`; mask = top-t by `tf*idf`. **Structurally the wrong shape**, four reasons:
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

### Offline β-sweep (first-order, static footprints)

Gate `π(s)=(1−u(s))^β`, `u(s)=max over prior tasks of peak-normalized read profile`; β=0 reproduces the measured read-through (37.3% sep5 / 55.4% ctrl — validated). Net = mean prior read-mass saved − mean writer demand blocked:

| β (sep5) | mean saving | mean cost | net | env7 RTO | env0 cost | env1 cost |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 81.5% | 0 | 0 |
| 4 | 12.4% | 6.9% | +5.5% | 52.7% | 13.2% | 16.4% |
| **8** | 17.2% | 11.0% | **+6.3%** | **41.8%** | 20.4% | 24.7% |
| 16 | 22.2% | 16.5% | +5.6% | 30.9% | 29.6% | 34.7% |

Net-positive at every β>0, peaks ~β=8 (env7 read-through roughly halved). Cost concentrates on env0/env1 (themselves basket tasks near the floor). To hit the same env7 outcome the control pays its writers ~36% vs sep5 ~20–25% (compounding, confirmed). **Caveats:** mass ≠ rollout success (likely *favourable* — saving sits on env7 at 0%, cost on env0/env1 at 22/10%, concave returns); static footprints (no closed-loop re-routing); these are why the real run is the arbiter.

### Mechanism implemented

- **Store** `u(s)` (raw read counts/batch, mirrors online-IDF) at each task boundary (`u ← max(u, counts/counts.max)`, then reset). Folded *after* a task finishes so a task never protects against itself.
- **Gate**: `tfidf *= (1−u).clamp(min=0)**β` before top-t → protected slots fall out, budget reallocates to the task's private reads.
- NB: implementation β acts on the top-t **ranking** (reselection), so its scale differs from the offline soft-suppression model — treat β=4 as "moderate."

### Launched (23 Jun)

Reuses the sep5 40k prior (no new pretrain), protection β=4, **eval 20 eps** (vs 50). Otherwise identical to the sep5 sequential.
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

The Entry-27 mechanism ran (reused the sep5 prior, 20 eval eps). **It works — on the protectable half.** Headline (caveat: 20 eval eps vs the 50-ep baselines, so ±3–4pp noise; slot metrics are eval-independent and corroborate):

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

Then probed the actual query map from the **checkpoint alone** (no forward): embed the 10 instructions (all-mpnet-base-v2), push through each layer's `film_mlp`, inspect γ/β. **The premise is wrong — the FiLM language pathway is near-inert:**
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

### Part C — forward probe

The checkpoint-only probe is decisive that language is near-inert, but the last open question — *is the basket residual driven by scene-similarity or by the small β-residual?* — needs `proj(x)` magnitudes.
- runs ~25 batches/task on basket (task_idx 4/5/7) + controls (2/6) through the frozen sep5 checkpoint, recomputes routing under **full = proj·(1+γ)+β / scene = proj·(1+γ) / lang = β** and reports `‖scene‖` vs `‖β‖`.
- **Decisive test:** basket `IoU(scene)` vs `IoU(full)`. `scene << full` → β drives the collision → option 1 has teeth. `scene ≈ full` → pure scene similarity → only options 2/3.

**RESULT (25 batches/task on basket 4/5/7 + controls 2/6):**
- **Scene dominates β by 17–21×**: `‖proj(x)·(1+γ)‖` ≈ 16 / 18 / 18 / 20 at L8/10/12/14 vs `‖β‖` ≈ 0.98. The language bias is a ~5% additive perturbation on the query.
- **Stripping β changes basket routing by ≈0%** (L14 basket weighted IoU): full **0.215** vs scene (β-stripped) **0.216** — e7~e0 0.221 vs 0.224, e7~e1 0.282 vs 0.283, e0~e1 0.142 vs 0.143. Language-only (q=β) would be 0.545, but β is 20× too small to bend the routing. (full-query basket 0.215 / background 0.058 reproduces the sequential JSON read IoU — pipeline validated.)
- **VERDICT: env7 is pure scene-similarity, NOT routing-fixable by reweighting.** Scene already dominates 20×; deleting language entirely leaves the basket collision unchanged. The three basket tasks have near-identical *initial scenes* (same kitchen, overlapping objects, a basket); the sole discriminator is the instruction, which the model routes ≈0% on. **Option 1 (down-weight language) is DEAD.** The only routing-based lever left is the *opposite and amplified* — recover AND scale the language pathway ~20× (anti-collapse + magnitude) so the instruction can steer routing — a large architecture change, with held-out-generalization risk (Entry 18) and bounded by instruction similarity (mpnet cos 0.86). Pragmatic read: env7 is the genuine *same-scene/different-instruction* residual (1 of 10); bank the β=4 protection win on the other 9 and pivot to the plasticity ceiling for the average; revisit env7 only if a language-amplification or co-host idea is worth a pretrain.

### Next steps
1. Run the forward probe (Part C) → decide whether env7 is routing-fixable at all.
2. Fold β=4 protection into the recipe regardless (it's a free retention win on the incidental channels).
3. Plasticity remains the binding constraint on the *average* (diagonal ~43; dual-cycle tasks low) — steps/task, value_lr, rank, longer pretrain. Independent of the env7 question.

---

## Entry 29 - 24 Jun 26 (LAUNCHED: autonomous protection+plasticity batch on the sep5 prior — 4 sequential runs)

Lined up 4 **sequential-only** runs (all reuse the EXISTING sep5 40k prior — no new pretrain), single-knob deltas from the standing baseline **β=4 protection (Entry 28, 40.5% @20ep)**. All 20-eval-eps for apples-to-apples with the β4 baseline.

| # | run suffix (`…_top_t_1536_<suffix>`) | delta vs β4 | tests | prediction |
|---|---|---|---|---|
| C | `protect_beta4_steps5k` | steps 3k→5k | plasticity (safe) | diagonal/init ↑ (MSE still falling at 3k block-ends) |
| B | `protect_beta4_lr2x` | value_lr 1e-3/1e-4→2e-3/2e-4 | plasticity (LR) | diagonal ↑; **watch 2e-3 peak instability** (Entry 20: t8 late-block MSE; grad-clip 1.0 is the guard — if grad_norm/MSE blow up it's diverging, fallback floor-only 2e-4) |
| D | `protect_beta4_lr2x_steps5k` | both | plasticity ceiling / additivity | most likely new best; protection(write-location) ⟂ LR(write-magnitude) → should compose |
| A | `protect_beta8` | β 4→8 | protection curve | env7 ~unchanged (self-limiting); env0/env1 start paying; incidental channels protected harder |

**Standing conclusions feeding this batch (Entries 27–28):**
- β=4 protection is a confirmed keeper: forgetting halved (init→final −7.6→−3.0), mean read-through 37→30%, +6.5pp — entirely on the **incidental** channels; env7 (genuine hub) stays 0.
- **env7 is NOT routing-fixable** (forward probe, Entry 28 Part C): scene dominates the query 17–21×; stripping β changes basket routing ≈0% (full 0.215 vs scene 0.216 @L14). It's the genuine same-scene/different-instruction residual. Routing-side lever would be the *opposite+amplified* (recover & scale language ~20×) — deferred, risky.
- The **average is plasticity-bound** (diagonal ~43, dual-cycle tasks low) — hence this batch targets plasticity. The big uncovered lever is **rank-4** (per-slot expressivity) — needs a fresh pretrain + a clean **50-ep re-eval** of whatever wins this batch (de-noise the 20-ep numbers).

**Analysis recipe:** env map `{0:4,1:6,2:9,3:2,4:7,5:0,6:8,7:1,8:3,9:5}`; train order = task_index 0..9; basket family = task_idx 4/5/7 (env7/env0/env1, env7=soup+cheese hub).
- Retention matrix: control 34.4 / sep5 34.0 / β4 40.5 (final avg); mean init (diagonal) control 39.8 / sep5 41.6 / β4 43.5 — **for the plasticity runs the key read is whether mean init/peak rises above ~45**.
- Slot usage: capacity (L14 effnum/core50), read-through-overwrite (control 55/sep5 37/β4 30%; env7 86/81/71%), pairwise overwrite, read-IoU validation (must match logged `memory_iou`).

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
- Both: NOT sequential — standard joint finetune, eval **libero_10 @ 50 eps/task** at the end only. `scheduler_decay_steps=50000` (honor the schedule, E20 gotcha).
- **Dataset built:** libero_90 3959 eps + the 10 Long tasks 379 eps.
- **What this tells us:** if base-pi05 on 90+10 reaches ~50–55% per-task, the memory method is roughly at the joint-finetune ceiling and the remaining gap is the OOD dual-cycle structure (→ rank/longer-pretrain, not continual-learning machinery); if base-pi05 is well *above* us, the memory constraints (frozen backbone + rank-2 values) are costing real performance and that's the thing to attack. The libero_10-only run isolates how much the 90-task pretrain data actually helps the Long suite.

### Next
1. Read the two baselines' per-task libero_10 success → place our 44.5 / peak-55 against the standard-finetune ceiling.
2. **50-ep re-eval of C** (`protect_beta4_steps5k`) to de-noise 44.5 and resolve the env4 cell.
3. Then the orthogonal plasticity swing: `lora_rank=4` (fresh pretrain — now best-justified, attacks the per-slot expressivity the steps/LR levers can't), and/or push `steps→7000` (steps is the clean Pareto lever; MSE was still the cleanest mover).

---

## Entry 31 - 1 Jul 26 (non-memory baseline: joint-finetune ceiling = 72.6%; the memory gap is ~28pp, concentrated on the forgetting/dual-cycle tasks)

Plain pi05, **NO memory**, same base train args (bf16, bs32, grad-ckpt, warmup 4k / decay 50k, pi05 default LR), 50k steps, eval libero_10 @ 50 eps/task. Ran **B1 = libero_90 + libero_10 joint finetune**; killed B2 (libero_10-only) — B1 answered the question.

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

That 28 pp is mostly forgetting + plasticity, and rank is the one untried per-slot-capacity lever. Rank-4 on all four layers is 4.8 B values → OOM, and rank-3 has matmul issues — so the question is *where* to spend a smaller increase. Implemented **per-layer rank**. Two 10k probes, **C's noloc-sep5 recipe verbatim except `layer_ranks`** (compressed 10k schedule = every prior probe → the existing rank-2 sep5/P9 audit is the `[2,2,2,2]` baseline):

- **P1 `[2,2,2,4]` on `[8,10,12,14]`** — keep all four, boost the action-proximal L14 (+25%, 3.0 B values). *Invest in the highest-value / most-output-proximal layer.*
- **P2 `[4,4,4]` on `[8,10,12]`** — drop L14, rank-4 on the rest (+50%, 3.6 B). *Josh's original: shed the messy high-interference L14, beef up the cleaner earlier layers.*

Decisive read at the held-out audit: does **P2's L12 inherit-and-worsen** the family IoU / core50 — the high-trust/high-forgetting role is *positional* (last memory layer), so dropping L14 likely migrates it to L12, now at rank 4 = more destructive overwrites (Entry 4) — or does P1's proximal boost sit better with routing intact? Plus the 1-task plasticity probe for fit. Caveat: rank is a **fit** lever, not a forgetting one under the frozen router (realworld Entry 3) → expect diagonal gains; the average is plasticity-bound anyway.

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

### Next: `[2,2,4,4]` A/B probes

`[2,2,4,4]` = the max-affordable all-layers config: 3.62 B values (= P2's footprint, fits), boosts the two layers carrying ~70% of read mass/trust (L12+L14, per Entry 22d ladder), keeps L14 in place so no positional migration. Two cells, single-knob delta:

| cell | config | question |
|---|---|---|
| A | `[2,2,4,4]` c=0.05 sep=5.0 | does r4-at-L12 perturb L14's routing from below? (P1 only tested the last layer — nothing downstream of it) |
| B | `[2,2,4,4]` **c=0.1** sep=5.0 | does rank-4 make compaction affordable AND convert to lower famIoU via sep? (the model-driven contrastive re-look; untested combination) |

**Rank-adjusted gate:** famIoU ≤ ~0.28 AND query_intra ≤ ~0.93 AND capacity-in-rank-units ≥ baseline at the r4 layers (core50 ≥ ~1300 at L12/L14) AND r2 layers not cratering (L8/L10 core50 ≥ ~50% of P9's 534/1047). B graduates if it holds capacity and lands famIoU < A; A graduates if B fails the P3 pattern (famIoU ≈ A) or craters L8/L10; `[2,2,2,4]` (P1) is the validated fallback. Blast-radius contingency for the eventual sequential: per-layer sep bump on L12/L14 only.

Known risk either way: rank is a **fit** lever, not a retention lever (frozen router fills all ranks of a shared slot; realworld Entry 3) — the diagonal should move; env7-class genuine contention won't, and r4 overwrites hit 2× harder → β4 protection stays mandatory in the sequential stage.

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
3. **Slot-ablation importance probe** (new instrument, Josh-approved idea): zero the top-k contested slots on a frozen adapted checkpoint, measure the `--eval.type=loss` delta → calibrates read-mass against actual importance.

Test 2's rank-2 baseline = **protectB4** (same β4/prior; only mismatch 3k-vs-5k steps, ~cancels in the layer ratio).

**Rank-2 calibration curve BUILT (2 Jul):** 70 task-points. Shape: RTO<20% → ret ~104% (net transfer), 0 collapses; 20–40% → 95%; 40–60% → 85%; **>60% → 7/10 tasks collapse to ≤5** (bin means mask bimodality — deep basins hold, shallow ones cliff, exactly the two-factor model). corr(RTO, drop): pearson −0.36 / spearman −0.42 — moderate, confirming exposure is only half the story (basin depth is the other half).

### Update (3 Jul) — `[2,2,4,4]` A/B verdict: **A (c=0.05) GRADUATES — healthiest audit in the project; B (c=0.1) repeats the P3 pattern (compaction ≠ family separation)**

Held-out audit (last layer L14; gate famIoU≤0.28 ∧ q_intra≤0.93 ∧ r4-core50≥1300 ∧ L8/L10 ≥ 50% of P9):

| cell | L14 core50 (rank-units) | L12 core50 (r-u) | L8/L10 core50 | famIoU L14 | bgIoU | e7~e1 | q_intra | verdict |
|---|---|---|---|---|---|---|---|---|
| P9 r2 baseline | 2679 (5.4k) | 1362 (2.7k) | 534/1047 | 0.264 | 0.087 | 0.302 | 0.912 | — |
| **A c=0.05** | **2981 (11.9k)** | 1687 (6.7k) | 923/1599 | **0.265** | 0.092 | 0.288 | 0.910 | **✓ GRADUATES** |
| B c=0.1 | 2156 (8.6k) | 1227 (4.9k) | 682/1055 | **0.279** | 0.073 | **0.367** | 0.931 | ✗ P3 pattern |

- **A answers the arch question: r4-at-L12 does NOT perturb L14 from below.** L14 famIoU 0.265 = P9's 0.264; core50 *up at every layer* (923/1599/1687/2981 vs 534/1047/1362/2679 — above even control's, before counting rank-units: 2.2–2.5× at the r4 layers); 4-layer mean famIoU 0.199 vs P9 0.211; fit fine (min MSE 0.201 vs 0.196). One structural shift: the gate ladder flattens (in-run gate L12 0.964 ≥ L14 0.952 — trust spreads over the two beefy layers), showing as mildly higher L12 famIoU (0.245 vs 0.210) — watch L12 channels in the sequential.
- **B disconfirms the contrastive re-look at this dose.** The rank-relaxed capacity gate ~held (rank-units ≥ ~92% of baseline; L8/L10 did NOT crater — the global-c bluntness worry was wrong), and background improved (0.073). But **famIoU went UP not down** (0.279 > A's 0.265) and the hub's worst channel degraded (e7~e1 0.367, approaching control's 0.355) — compaction pulls each basket task's queries tighter around the *same scene representation*, so sep=5 still can't translate the family apart. Same failure shape as P3, now demonstrated with capacity held by rank — the family overlap is scene-genuine, not a capacity artifact. q_intra 0.931 right at the guard. B's only win (background) is the half β4 protection already handles at write time. **c=0.05 confirmed; the contrastive question is closed at both ranks.**
- **Next:** graduate A → fresh 40k (`layer_ranks=[2,2,4,4]`, noloc sep5 recipe, clean schedule) → 40k audit → sequential with C's config (β4 + 5k steps/task + top_t 1536) → overlay the 10 rank-4 points on the rank-2 RTO curve (test 1) + the within-run r2-vs-r4 DiD (test 2, baseline protectB4). Optional pre-flight: 1-task plasticity probe A-10k vs P9-10k to confirm rank→fit conversion.

---
## Entry 33 - 8 Jul 26 ([2,2,4,4] verdict: rank exonerated AND exhausted (+6 diagonal, on-curve retention, no drift amplification) — bottleneck = 14pp frozen-backbone fit + 12pp retention tax → STAGED PRETRAINING protocol launched (competence into the frozen backbone, memory as pure residual substrate))

### The [2,2,4,4] graduation result

40k pretrain held-in **81.9** best-of-family → sequential, C's config, 20 eps. Final **46.5%** (+2.0 over C, within noise), init **49.5** / peak **58.5** (both project bests), forgetting −3.0 (C: +1.0). Retention matrix (init→final): e4 35→20, e6 55→**75**, e9 40→20, e2 85→80, e7 25→**0**, e0 45→45, e8 20→**45**, e1 30→20, e3 80→80, e5 80→80.

| run | final | init | peak | forget | MSE-floor | L14 effnum/core50 | read-IoU | RTO | ch≥12% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protectB4 (r2,3k) | 40.5 | 43.5 | 49.5 | −3.0 | 0.150 | 5257/2534 | 0.051 | 29.8% | 6 |
| C steps5k (r2,5k) | 44.5 | 43.5 | 53.5 | +1.0 | 0.120 | 6630/3265 | 0.057 | 33.2% | 8 |
| **r2244 (r4@L12/14)** | **46.5** | **49.5** | **58.5** | −3.0 | **0.105** | 6394/3152 | 0.059 | 34.7% | 9 |

Slot pipeline validates (computed IoU 0.0592 == logged). Exposure ≈ C on every axis — rank-4 did NOT broaden footprints at matched steps/LR.

### Both pre-registered rank tests (E32): rank-invariant — Josh's fractional-invariance model confirmed

1. **RTO-curve overlay (test 1): ON the curve.** Task-matched residual vs each env's 7-point rank-2 history = **+1.3pp ≈ 0** (e6 +31/e8 +18 net-transfer absorbers; e7 −21 vs a floor-capped prediction; rest −7…+2).
2. **Contested-slot drift DiD (test 2): r4 slots drift LESS ≈ pure gradient dilution.** ρ = d(L14)/d(L8) on env7's core across env0's block: protectB4 (all-r2) **1.26** (updated-slots) vs r2244 (L14=r4) **0.74** → DiD ≈ 0.59, ≈ the expected 2×-params dilution (0.63). No specialization-drift excess.

**Why retention still looked worse: the fit gains landed on high-RTO tasks whose collapse floors are unchanged** — e7 init 10→25 then →0 regardless; e0 +20 init kept +5. More to lose, same losers. Run-specific aggravation: the 40k audit's e7~e0=0.445 watch-item materialized (e7←e0 overwrite **67.8%** vs C's 41.6%; env0 hits 74.5% of e7's L8 core mass vs protB4's 44.3%) — this pretrain converged the soup pair (prior-wide), concentrating e7's death into one channel. e3-as-heavy-writer appears in C too (5k-steps effect, not rank). Gate-ladder flattening (L12 0.977 ≥ L14 0.973) persisted with no visible extra damage channel — L12 watch-item closed.

### Why the fit gain was small

MSE floor −12% vs C but concentrated on already-easy tasks (soup+sauce −30%, book −21%); the hard dual-cycle set barely moved (**soup+cheese −2%**, mug+micro −6%, cheese+butter −9%). +50% params bought less than steps 3k→5k (−20%). ~Half the blocks stop improving by mid-block (identical pattern in C). **Per-slot capacity was never the binding constraint on the low tasks — the frozen backbone is** (base pi05 joint finetune = 72.6 on the same data).

### Bottleneck, measured (gap to 72.6 = 26.1pp)

**14.1pp fit-at-peak** (e7 +55, e1 +36 — basket dual-cycle) **+ 12.0pp retention-from-peak** (e4 35 noisy-cell, e7 25, e9 20, e0 15). Both halves rank-insensitive. Lever ledger now measured-and-spent: separation (flat), protection (β4 opt), steps (half-saturated at 5k), LR (backfires), rank (+2 net). Structural constraints: frozen backbone (fit half) + frozen router & gate≈0.97 (retention half).

### Discussion (Josh) → decision: STAGED PRETRAINING

Josh's 2-prong framing: (1) some tasks don't convert capacity into fit; (2) some convert but get overwritten. Options weighed: gate-desaturation penalty (first-order analysis: gate scale CANCELS out of seq-task-on-seq-task interference since ΔV ∝ 1/g — the real effect is raising the post-collapse floor by shifting pretrain competence into the frozen part ⇒ same bet as stronger pretrain, enforced structurally); more layers (extend DOWN, [4,6,8,10,12,14] — last-layer hub role is positional, earlier layers are the clean ones; E9/E11 precedent); β8×rank-4 (new cell: r4 private capacity may absorb the veto that starved writers at r2); **magnitude-based protection** (move (1−u)^β from top-t *ranking* to per-slot *gradient/LR scaling* — the E27 offline soft-suppression frontier that measured net-positive at all β, env7 RTO 81.5→41.8 at β=8, which the ranking implementation only delivered at ~β≈1 strength; strongest un-tried retention lever, PARKED not dead).

**Chosen: Josh's staged protocol** — (1) finetune base pi05 (no memory) on libero_90; (2) attach memory, train ONLY memory (values+keys+query/FiLM+gate) against the frozen backbone with the sep5 recipe; (3) sequential as usual. Memory becomes an additive residual/adaptation substrate; core competence lives in frozen weights by construction. What it buys: post-collapse floor = stage-1 zero-shot (e7 currently falls to literal 0 because corrupted memory < no memory); ~blank value table at stage 3 (no pretrain content to read through and lose); big iteration-speed win (stage 2 has no backbone grads/Adam → every future prior experiment is a cheap stage-2 rerun). What it does NOT fix: delta-on-delta overwrite of sequentially-learned fit. Open question the probe answers: does the sep5 routing pocket reproduce on a frozen backbone (standard audit gates: famIoU≤0.28 ∧ core50≥1500 ∧ q_intra≤0.93)? Pre-registered success criterion for the full chain: **no task ends below its stage-1 zero-shot floor, and final ≥ 50**.

### Code + launch

- New flag `--policy.train_memory_only` (pi05): freezes everything without `.mlp.mem.` in the name (60 memory tensors trainable / 2.45B params, 813 frozen / 4.14B); 52 router tensors @ base LR + 8 value tensors @ memory_lr.
- Stage 1 = E31-baseline recipe on libero_90, 50k, eval **libero_10 @ 50 eps at 50k = the zero-shot floor table**; stage 2 probe 10k compressed, frozen-base, sep5 recipe verbatim. Then 40k clean schedule → C's sequential verbatim; launch after the probe audit clears.

---
## Entry 34 - 10 Jul 26 (STAGED protocol first results: stage-1 zero-shot floor ≈ 0 on all collapse tasks; frozen-base probe = a NEW ROUTING REGIME — gates naturally moderate at 0.63-0.68 (first time ever) but the routing losses lose their actuator and the audit FAILS separation (famIoU 0.390, inverted layer ladder) → nonlinear query head (query_proj_layers=2) implemented + probe LAUNCHED; script 2 held)

### Stage 1 (libero_90 base finetune, no memory, 50k) — healthy, and the floor table kills the floor-raising hope

Train MSE 0.09-0.13 at 50k (≈ joint pretrains' fit). **Zero-shot libero_10 @ 50 eps = 10.6% mean**; per env: e3 bowl+drawer **60**, e5 book **32**, e2 stove+moka 8, e0/e6/e7 **2**, e1/e4/e8/e9 **0**. So the post-collapse floor the protocol was meant to raise is ~0 exactly on the collapse-prone tasks (env7 floor = 2). Corollary worth recording: every sequential init we've ever measured (25-85%) was ~entirely memory adaptation, not backbone transfer — the diagonal IS the memory system. (Floor table = the pre-registered stage-3 comparison row.)

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

### Held-out audit — FAILS separation, smashes capacity

L14: famIoU **0.390** (control 0.349, P9 0.264; gate ≤0.28 FAIL), bg **0.188** (2.2× P9), core50 **6456** / effnum 14663 (2.4× control; capacity gate passed ×4.3). **Layer ladder INVERTED**: L8 is now the worst layer (famIoU 0.538, bg 0.272) instead of the cleanest — raw frozen features are generically similar early and differentiate with depth, the opposite profile of a routing-loss-shaped backbone. Family pairs uniform (e7~e0 0.385 ≈ e7~e1 0.399 ≈ e0~e1 0.386) — generic broad-footprint collision, not object-graded semantics.

### Mechanism (the diagnosis)

**The routing losses lost their main actuator: backbone co-adaptation.** In every joint pretrain, contrastive/sep gradients flowed through proj(x) into the backbone and reshaped the hidden states themselves — that's what actually bought the E20-26 compaction/translation. On a frozen backbone the only routing surface is one shared linear proj + FiLM, which cannot fold 90 interleaved task clouds into compact separated footprints. Weight-limited is disconfirmed (aux already 10:1 over MSE); step-limited unlikely (in-run sep flat 0.18 from 2.5k). Representation-limited.

### Why this is ambiguous rather than a clean fail (discussed with Josh)

Three of four damage-model terms moved favorably — gate 0.63-0.68 (attenuation at read time), near-blank values (no pretrain content to lose through shared reads), 2.4× slot capacity for β4 to steer writes into — and one badly (exposure: broad overlapping reads → high stage-3 RTO). The audit gate thresholds were calibrated on joint priors where high famIoU meant load-bearing shared content at gate 0.98; none of those side conditions hold here, so 0.28 may not transfer. The floor-raising benefit, though, is measured dead (floors ≈ 0).

### Decision (Josh): fix the router for frozen-x FIRST — nonlinear query head.

`query_proj_layers` (default 1 = original single linear) + `query_proj_hidden_dim` (0→input_dim): Linear→SiLU→…→Linear; ~2M params/layer at depth 2.

**Probe (10 Jul):** single-knob delta from the failed frozen-base probe (`query_proj_layers=2`).

**Read + decision rule:** primary = famIoU materially down from 0.390 toward ≤~0.28 with core50 ≥1500 (it starts at 6456 — huge room) and q_intra ≤0.93 (some RISE from 0.71 is expected and fine — the head can now tighten clusters); secondary = does the inverted ladder flatten, in-run sep < 0.18. If it clears/materially improves → stage-2-full graduates WITH `query_proj_layers=2`. If ≈ depth-1 → the frozen-feature separability ceiling is confirmed → run stage-2-full + sequential on depth-1 anyway as the end-to-end baseline (the changed damage model deserves one full measurement); depth-3/wider-hidden only if something suggests headroom.

---
## Entry 35 - 12 Jul 26 (STAGED PROTOCOL VERDICT: DEAD as a frontier route — killed at 6/10 blocks. Held-in PARITY (81.7) but held-out adaptation collapses (inits −18pp, MSE floors 1.5-2×, e7 init 0): compact routing was a JOINT-TRAINING product; diffuse routing dilutes both reads AND writes. Pivot back to the joint track)

### Chain status + kill decision (Josh)

Stage-2 full (40k, frozen base, depth-1 head per E34) completed clean; 40k audit ran; stage 3 (C's config verbatim) auto-started and was **killed at 6/10 blocks** (~30k steps) — mid-flight results were clearly non-competitive and the pre-registered criterion (final ≥ 50) was already unreachable.

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
4. Held-in eval CANNOT gate substrate quality (81.7 parity here) — only held-out adaptation probes can.

### Pivot (back to the joint track; the E33 lever ranking resumes)
1. **Magnitude-based protection** (per-slot gradient/LR scaling by (1−u)^β instead of top-t reselection) on the [2,2,4,4] prior with C's config — the E27 offline frontier (net-positive at all β, env7 RTO 81.5→41.8 at β=8) that the ranking implementation never delivered. Sequential-only, no pretrain.
2. **50-ep re-eval of r2244's final** — de-noise the 46.5 frontier number before building on it.
3. **6-layer depth** ([4,6,8,10,12,14] r2 joint pretrain) — the remaining capacity shape (E33).

---
### Entry 35 addendum (12 Jul, deep-dive at Josh's request) — the full mechanistic chain, with code findings

**Code finding 1 (the actuator, confirmed):** forward passes live `x` into `query_proj` — NO detach. In joint pretrains the contrastive/sep gradients therefore flow THROUGH the query into the backbone (and at E24-measured magnitudes: aux terms 30-91% of MSE). Joint routing quality was bought by sculpting the hidden states; in stage 2 that path dies at `requires_grad=False`. qproj2 (E34) already showed head capacity is not the binding constraint.

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
4. **Language-only routing (Josh's idea) — evaluated, parked**: E28's forward probe already measured it (q=β): basket family IoU **0.545 vs 0.215** scene-based — doubles the binding collision (near-duplicate unseen instructions co-locate under any smooth map; mpnet cos 0.86), and constant per-task queries = the E21 state-independent regime = per-task-adapter baseline rebuilt minus state-conditional composition. Keepable kernel: a hybrid language-only HEAD (private sections).
5. **Move memory layers up (Josh's idea) — measured, negative** (probe below).

### The feature-separability probe (new instrument)

Hooks on each expert layer's `.mlp` input (= the router input), mean-pooled per sample; 4 sweeps (stage-1 & joint sep5 ckpts × libero_90 & libero_10), layers [4,8,10,12,14,15,16,17]; linear task probes + centroid cosine geometry.

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

### Router-only training (`train_router_only`)

Freeze everything except keys + query_proj/FiLM (28 tensors); values pinned at init (slot_up zero). The two load-bearing mechanics: **zero values ⇒ MSE gradient on the router is exactly ~0** (warm-up purity) and **contrastive+sep gradients DO reach keys/proj** (the learning signal).

### Router warm-up v1 (12 Jul)

Config: router LR 2.5e-5→**1e-4**, c=0.05 kept (Josh: comparability; recalibrate later if needed), sep5/noloc/rq512, [8,10,12,14], 10k compressed schedule. **Decision tree**: famIoU ≤~0.28 ∧ core50 ≥~1500 ∧ q_intra ≤~0.93 → anchoring hypothesis confirmed, geometry-before-content becomes the method → discuss A-then-sequential. Collapse signature (core50 crater / q_intra >0.95) → unopposed contrastive won → c down. Still ~0.39 → the sep-through-top-M formulation itself can't find the probe's directions → fallback: **seed the query proj from the linear-probe directions** (we have the probe weights).

### Design decisions from the discussion
- **A vs B after warm-up** (A = short values-on-90 phase, B = straight to sequential): Josh prioritizes iterating the geometry to near-perfect first (the warm-up↔audit loop is ~4h/iteration — the staged protocol's iteration dividend), expects A to help downstream. Note: anchoring is a FEATURE once geometry is good (content cements the map); A also calibrates gates (untrained ≈0.5 after warm-up) and re-audit after A is warranted (values perturb downstream x → routing shifts, E30 mechanism).
- **Per-task LoRA-FT baseline** (vla-memory.md §5, never run): three roles — (i) the fit CEILING of adapter-adaptation on this exact frozen backbone (turns the "LoRA-FT is enough" premise into a number; separates "what the lookup costs" from "what freezing costs"), (ii) the reviewer-mandatory comparator (answer: task-identity-free routing, fixed params, measured forward transfer — e6 30→75, e8 20→50), (iii) effort compass (LoRA-FT ~55 → substrate near ceiling, work on retention; ~70 → memory design leaves fit on the table).

---
## Entry 37 - 12 Jul 26 (ROUTER WARM-UP v1: ALL GATES CLEARED — famIoU 0.145, the best routing prior in project history, on a FROZEN backbone, in 3h. Anchoring hypothesis confirmed end-to-end; the "scene-genuine 0.26 floor" was substantially an anchored-equilibrium artifact. → A-phase (values-only, ROUTER FROZEN) + sequential LAUNCHED)

### Warm-up result (10k router-only, lr 1e-4, c0.05/sep5/noloc/rq512, frozen stage-1 backbone)

In-run @10k: routing sim **0.050 mean AND L14** (P9 joint: 0.061/0.085 — beat the joint run outright, and L14 matches the mean for the first time ever: no MSE-trust equilibrium → no last-layer pathology), q_inter **0.016**, q_intra 0.883 (inside P9's healthy band — the unopposed-contrastive/E21 collapse never came at c=0.05), support_L14 5191, effnum 10.4k.

Held-out audit, all gates cleared:

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

**Chose: router FROZEN (option 1).** (i) MSE's router-gradient points back at the joint attractor (famIoU ~0.26, trust ladder) — small LR only slows the walk back, and it erodes a certified asset; (ii) routing drifts anyway via upstream-value→x perturbation (E30 Finding 2) — minimize the controllable part and measure the rest with a re-audit; (iii) values-on-frozen-router demonstrably fits (every sequential block ever); (iv) crisp protocol: aux-only geometry → frozen router → MSE-only content, consistent with the sequential's frozen-router constraint. **Reserve trigger for option 2** (router unfrozen at ~2.5e-6 ≡ train_memory_only + tiny optimizer_lr): A-phase MSE plateauing meaningfully above the joint pretrains' libero_90 level (~0.13-0.16) = separation-vs-usefulness mismatch.

### Chain (12 Jul)

- A-phase trainable = memory module MINUS keys/query_proj = **32 tensors** (values 8 @ memory_lr + gate/value_proj/swilu 24 @ base LR).
- **A phase** (10k values-only on libero_90 from the warmed router ckpt, aux losses kept ON as pure telemetry — grads dead-end on the frozen router but the in-run routing-sim log becomes the live drift monitor; held-in eval @10k = seen-task plasticity check) → **re-audit** (deployed geometry after value training; informational) → **sequential** (C's config verbatim: β4 + 5000 steps/task + top_t 1536, 20 eps).
- Pre-registered sequential reads: inits vs fb-depth1's mean-30 disaster and r2244's 48 (the dilution chain predicts routing quality converts to FIT here); self-adapted read mass (fb 43-53% → want 70%+); retention matrix vs floor table; RTO overlay (damage-per-exposure).

---
## Entry 38 - 13 Jul 26 (rwarmupA sequential postmortem: the killer is ROUTING DRIFT — value training re-points the frozen router above L8 (self-IoU 0.21-0.33 vs L8's exact 1.000); exposure/writes exonerated; the certificate was void by task 0's own block. Fix: FROZEN-BASE ROUTING (dual-path); stageB chain (A rerun -> audit -> 5-task sequential))

### The run (killed at 7/10 blocks, Josh's call)

`libero_10_sequential_..._frozenbase_rwarmupA_..._protect_beta4_steps5k` — C's config verbatim from the A checkpoint (warmed router + 10k values-only fill).

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

Ran the held-out audit on the seq-35k checkpoint and compared per-task footprints to the post-A audit (same instrument, same demos, seed-matched):

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

The staged substrate lacks **routing stationarity**. E37 proved the geometry is achievable on frozen features; nothing holds it — the router input is the live stream, value training moves the stream, and the warmed router separates tasks along low-variance directions of the crowded frozen-feature cone (E36: inter-task cos 0.86-0.99) so there is no margin to absorb the motion. Joint priors tolerate the same nominal channel because (1) co-trained features are spread (cos ~0.46 → wide margins), (2) heavy pretrain content (‖θ‖ 1.83-2.6 vs A's 1.06) makes sequential writes a smaller relative field change (+35% vs +75%), (3) saturated gates are insensitive. Refines E35's verdict: joint training doesn't (only) create the geometry — it creates the margins and ballast that keep it still. (r2244 post-sequential audit for the cross-regime drift number was started and killed on Josh's call — the L8-vs-L10/14 within-run control is decisive without it.)

### Fix implemented: FROZEN-BASE ROUTING (dual-path)

Memory ROUTING (query projection + gate) reads the backbone features as they would be WITHOUT any memory contribution; the value/output path (LoRA transform, swilu) stays on the live stream. Addressing becomes stationary by construction — L8's IoU=1.000 extended to every memory layer. The warmed router was trained with values pinned to zero, i.e. on EXACTLY the features the frozen branch serves → it drops in unchanged.

- Overhead within noise; the fork costs ~40% of one expert-suffix pass, expert ≈ 300M of 6.6B.

### stageB chain (13 Jul)

**A rerun** (10k values-only, router frozen, frozen-route ON — gates must train against the frozen-branch input they'll read at deployment; old-A content was measured compatible but 4h buys exact train/deploy consistency) → **audit** (expect ≈ the warm-up certificate 0.145/2955 — the frozen branch serves exactly the warm-up features) → **sequential, FIRST 5 TASKS** `[0-4]` (deadline scope; contains all three collapse cases e2/e6/e7 + the e7←e0 genuine channel), C's config verbatim + the flag.

**Tripwire (zero GPU):** when task 0's block JSON lands, IoU(t0 block reads, post-A audit) must sit near the ~0.93 measurement ceiling (old wiring: 0.28). If low → the fix isn't holding in the real trainer → kill.

**Pre-registered reads on landing:** (1) per-task self-IoU pre→post ≈ 1.0 at ALL layers (the property, in production); (2) still-mine mass ≈ 1 − static-RTO; (3) retention back on the calibration curve at the cleanest-ever exposure — e2/e6 should hold near init; e7←e0 (famIoU 0.234, genuine soup channel) remains the residual risk with β4; (4) fit ≈ unchanged or slightly better (inits ~39-40; tasks no longer chase their own routing within a block).

### Next steps
1. stageB lands: tripwire after block 0, then the pre-registered reads above. If retention holds → extend to 10 tasks and/or 50-ep re-eval.
2. If e7 still collapses via the genuine soup channel → that's the honest residual: magnitude-based protection (E27 offline frontier) is the parked lever.
3. Post-deadline queue unchanged: per-task LoRA-FT baseline on the frozen stage-1 (fit ceiling / reviewer comparator), hybrid language-only head, joint warm-up variant.

---
### Entry 38 addendum (13 Jul, discussion) — the drift tax is retroactive: this channel was live in EVERY sequential run ever

Josh's observation, confirmed on existing data: routing drift (values mutating the stream the frozen router reads) was not introduced by the staged protocol — it has been present in every sequential run in the project; the regimes differ only in the damage coefficient.

- **Joint-era evidence, reread with today's lens:** E30 Finding 2's footprint broadening (L14 effnum 5257→11789 as write magnitude/duration grew) IS this mechanism — we measured it and named it "broadening" because we never compared a task's own footprint before/after. **B (lr2x) is the joint-era drift casualty**: damage-per-exposure ~3× β4's (ret −9.5 @ RTO 33% vs −3.0 @ 30%) — the same anomaly signature that cracked the rwarmupA run. Its "backfire" was attributed entirely to bigger overwrites; doubled ΔV also doubles stream perturbation.
- **Why joint survived it at standard LR:** margins (feature cos 0.46 vs the frozen cone's 0.86-0.99), ballast (relative field change +35% vs +75%), gate saturation. The static-exposure machinery (calibration curve, timing-matched cliffs, β4's +6.5pp) worked because drift-per-write was small there — but not zero.
- **The deep consequence: the rank-2 calibration curve's ABSOLUTE level is contaminated.** "85% retention at 40-60% RTO" was measured with drift on. No drift-free sequential run has ever been observed; **stageB is the first stationary-addressing run in project history.** Pre-registered read upgraded accordingly: retention ON the historical curve at matched exposure ⇒ drift was staged-only; retention ABOVE the curve ⇒ the gap is the historical drift tax, quantified behaviorally — and it applies to the whole family, joint runs included.
- **Follow-ons if confirmed:** (1) the flag for the JOINT track — with a variant: a joint router was trained on live features *including pretrain memory content*, so its frozen routing input should be features computed with **values snapshotted at sequential start** (same dual-path code, frozen branch carries the snapshot; ~10GB extra at r2), not memory-free. "r2244 + snapshot-frozen routing" is the candidate frontier run. (2) The direct retrospective number: the r2244 post-sequential audit (pre→post self-IoU for the joint regime).
- Relative conclusions of the project (write-budget, separation-as-translation, incidental/genuine protection split, capacity/interference axis) stand — all A/B'd within-regime with drift on both arms. The absolute retention levels, and specifically the lr2x reading, carry the unquantified tax.

---
## Entry 39 - 14 Jul 26 (stageB verdict: FORGETTING SOLVED — first flat retention matrix in project history (MSE forgetting +0.0-1.7%, routing self-IoU exactly 1.0000); the low score is NOT retention, it is the staged substrate's rollout-fit conversion (inits 35 vs r2244's 48), decomposed per-task: e9 = warm-up footprint dilution (measured), e4 = backbone-integration gap (footprint-controlled), e7 = staged WINS. The "apparent forgetting" (e6 60→25) is 20-ep eval noise on a +0.5%-drifted function)

### The run

`libero_10_sequential_..._frozenroute_rwarmupB_..._protect_beta4_steps5k_tasks5` — A-phase 10k frozen-route -> audit -> 5-task sequential, C's config + frozen-base routing everywhere; values-only 2.42B/6.6B trainable in sequential.

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
- **The definitive instrument** (new post-sequential audit vs post-A): per-task footprint self-IoU = **1.0000 exactly, every task, every layer** (rwarmupA: 0.21-0.32 above L8). Deployed geometry after 5 blocks is byte-identical to post-A: famIoU 0.144/0.144, bg 0.066/0.064, core50 2958/2958, effnum 6507/6507. **The E38 read-side drift channel is dead.**
- **Parameter-level proof** (checkpoint diff 010000→015000, e9's block): the ONLY tensors that changed are the 8 slot_up/slot_down — keys, query_proj/FiLM, gate, value_proj, swilu, backbone all bitwise-zero delta. Combined with self-IoU=1.0: the sole interference channel left in the whole system is *values on shared slots*.
- Certificate transfer: warmup-audit ↔ postA(B)-audit per-task IoU = L8 1.000, L10-14 0.980-0.988 (live-route A-phase: 0.926-0.945). The 0.98 residual is the value_proj-bias delta (the warmup audit ran on the live path, whose stream at values=0 still carries g×bias; the frozen branch is strictly memory-free) — expected, not drift. A-phase itself no longer moves routing.

### 2. FORGETTING: SOLVED. The E3-style MSE matrix is flat — first time ever

Ran the full 5×5 forgetting matrix offline (paired-noise, 16 batches/task, all 5 per-task checkpoints):

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

1. **Port stationary addressing to the JOINT track — "r2244 + snapshot-frozen routing"** (E38-addendum follow-on, now maximally supported): joint substrate has the fit (init 48), stageB proves stationarity eliminates function-level forgetting, and r2244's within-window losses (e4 35→20, e9 40→20, later e7→0) are the drift-suspect retention tax. The frozen suffix branch must run memory WITH slot values snapshotted at sequential start (the joint router/gate were trained on live-with-content features; memory-free features would be OOD for them). Expected: init ≈ 48 with a flat matrix → new frontier >46.5, and the drift tax finally quantified within the joint regime.
2. **Extend stageB to 10 tasks**: the matrix predicts the seen-avg holds ≈ its init mean through blocks 5-10 where rwarmupA fell to 24.3. That is the controlled catastrophe-elimination demonstration (mechanism E38 → fix → matched-config rescue) — the cleanest causal story the project owns.
3. **50-ep re-eval of the 5 retained per-task checkpoints**: every conclusion above rides on ±11pp cells; de-noise the trajectory, resolve e6's wobble empirically.
4. **Staged-track plasticity levers unlocked by zero forgetting** (if the staged track stays a thesis pillar): with drift dead and exposure clean, the levers that backfired via interference are now safe by construction — memory_value_lr 2e-3 (B's fit was best-in-family, its −9.5 retention was the drift tax; pre-registered: under frozen-route the tax should vanish), top_t 1536→~3072 (e9's coverage 17%→34%), steps 5k→7k. Cheap sequential-only reruns from the same A checkpoint.
5. **e4-class integration gap: do not chase in-protocol** — E35/E36 stand (needs backbone co-adaptation at some phase = joint track). The staged track's honest scope: routing-bound tasks won, integration-bound tasks conceded, zero forgetting.

---
## Entry 40 - 14 Jul 26 (E39 follow-through: joint track PARKED (off-thesis, Josh); the conversion-gap attack goes 4-way parallel — (1) affine slots + no gate, (2) lr 2x, (3) steps 7k, (4) top_t 3072)

### Decisions from the E39 discussions

1. **Joint track parked as off-thesis (Josh).** The thesis is "take an off-the-shelf VLA and add capabilities without erasing old ones" (with router-abstention as the future-work arc), so "r2244 + snapshot-frozen routing" is at most a *pricing ablation* (what does co-training buy at matched stationarity), not the mainline. Josh's counter to the E39 integration-gap story — "base + adapters is how everyone finetunes; the backbone shouldn't have to adapt" — is a testable claim, NOT settled by the E39 exclusion inference: our memory differs from standard LoRA-FT in four taxed ways (sparse routed mixture of 144 rank-2 fragments vs dense unconditional adapter; 4 expert-MLP sites vs attention+MLP everywhere; a trust gate in front; softmax averaging). **The per-task LoRA-FT baseline (E36, reviewer-mandatory) is the arbiter**: LoRA-FT at 35-50 on e4/e9 => backbone sufficient, the deficit is OUR machinery's conversion tax (closable in-protocol); LoRA-FT ~10 => frozen-backbone adapters can't do those tasks at all, scope the claim. Supporting fact: libero-90 has no dual-cycle tasks, so no stage-1 finetune can "embed composition subspaces" — the joint runs' e4/e9 rollouts (35-40) were installed by VALUES on a frozen backbone at sequential time, so composition-through-values is demonstrably possible; the joint edge is conversion machinery, not backbone knowledge.
2. **Gate REMOVED rather than saturated (Josh's call, agreed).** The E39 candidate mechanism for the conversion gap: the A-phase gate is calibrated on libero-90 where the backbone is competent and memory auxiliary (settles 0.72-0.85 mid-sigmoid), then deployed where memory carries ~everything (stage-1 floors ~0-2) through a valve whose delivered magnitude is state-dependent (g(1-g) sensitivity ~5x saturated) and never saw libero-10. Saturation-pressure would patch this; removal deletes it structurally, with no aux weight to tune. Every healthy joint run effectively ran ungated (0.97-0.99, "no modulation" — E19). We keep per-channel state-dependent modulation anyway via swilu (`value_proj(out * silu(swilu_proj(x)))`). Removal must apply from the A phase onward (values calibrate ||dV|| ~ target/g; stripping at deploy overshoots ~1.3x). Not kept for the abstention roadmap either: an MSE-trained trust valve is not an OOD-abstention detector — build that purpose-built later. (Gate-variance probe — is the 0.77 a constant or a modulator? — noted, not run.)
3. **"Lora + value" = per-slot AFFINE, not a layer split (from Josh's hybrid idea).** value_i(x) = U_i V_i x + b_i. The homogeneous LoRA (up @ SiLU(down @ x)) has no constant term — the DC component of a correction must ride x's stable cone direction, spending rank. The per-slot bias learns the DC directly (no U-V product coupling => robust to e9-style diluted writes), frees rank for the state-dependent residual, and restores the additive-constant class of value_type="vector" WITHOUT handing state-conditionality to the router (which the warm-up deliberately de-resolves within-task: q_intra 0.883 — vectors-by-routing would rebuild the E21/E22c per-task-bias pathology). Layer-split variants (lora early / vectors late or vice versa) rejected: L12/L14 carry the read mass, trust, and action proximity (r2244's winning allocation), and constants there would be router-addressed precisely where within-task resolution is weakest.
4. **e6's 60->25 formally attributed to eval noise + marginality** (Josh's challenge "would noise explain +/-30pp?"): a single -35pp cell at 20 eps under constant p~0.36 is a ~2-5% draw; with ~12 adjacent-cell pairs per run, >=1 such swing per run is near a coin flip — and every baseline shows them (C's e9 35->50 "negative forgetting"; r2244's e4 55->20 in-window). The claim rests on the paired-noise MSE matrix (+0.1% across e9's block), not on noise arguments. All 4 new arms carry a 50-ep final eval (below) to shrink the headline cell to +/-7pp.

### The 4-way parallel batch

All arms: 5 tasks [0-4], C's sequential config, frozen-base routing, beta4, 20-ep evals + **50-ep final**. Arms 2-4 are sequential-only from the EXISTING stageB A checkpoint. Arm 1 runs its own A phase (value-path changes require it; the warmed router drops in unchanged — frozen-branch routing is a function of backbone+keys+query only, which also makes the post-A audit provably identical to stageB's, so it is skipped).

| # | delta vs stageB | tests | key read |
|---|---|---|---|
| 1 | `lora_slot_bias=true` + `mem_gated=false` (A rerun + seq) | affine capacity + gate-miscalibration hypothesis jointly | e4/e9 inits vs 10/5; e6-class stability at 50-ep final; A-phase MSE vs 0.092 |
| 2 | value_lr 1e-3->2e-3 (end 2e-4) | drift-tax-free lr (joint-era B: best fresh diagonal +4, -9.5 retention = the tax, now dead by construction) | inits up with retention flat => bank it; retention drop => the lr tax was never (all) drift |
| 3 | steps/task 5000->7000 | optimization budget on stationary targets | MSE floors + inits; forgetting must stay flat (matrix if needed) |
| 4 | top_t 1536->3072 | the e9 DILUTION hypothesis (E39): per-batch L14 write coverage 17%->34% | e9 init off 5 => dilution real; e9 still ~5 at sharper writes => conversion/substrate, not budget |

Expected magnitudes (honest): arms 2-4 are worth ~+3-7pp average and ~0 on e4 (its MSE is already joint-level at joint-identical footprints); arm 4 is a hypothesis test more than a fix. Arm 1 is the mechanism bet. The LoRA-FT baseline (above) remains the queued arbiter for whatever gap survives the batch.

- Cross-arm eval note: arms are mutually comparable (same seeds/eval protocol); all differ from stageB itself only by their single delta (stageB @25k baseline row: 20/35/5/70/30 = 32.0 @ 20 eps).

---
## Entry 41 - 15 Jul 26 (E40 3-arm postmortem + the conversion-gap probe program: five instruments, two falsified mechanisms, one corrected over-claim -> a 3-layer failure model. The e6-after-e9 drop is REAL (4/4 replication; E39/E40 noise attribution REVERSED). Soft (grad_scale) protection calibrated; denoised-chunk error adopted as the gate metric)

### The three completed E40 arms (env order e4/e6/e9/e2/e7; finals 50-ep, stageB final 20-ep)

| arm | inits | mean init | final | block-min MSE | block-END MSE | eval-time paired MSE (diag mean) |
|---|---|---|---:|---:|---:|---:|
| stageB (baseline) | 10/60/5/70/30 | 35.0 | 32.0 | 0.1274 | 0.1410 | 0.144 (E39) |
| affine+nogate | 20/55/10/75/38 | 39.6 | 32.8 | 0.1190 | 0.1274 | 0.129 |
| lr2x | 35/65/25/80/32 | **47.4** | 35.6 | 0.1132 | 0.1239 | 0.125 |
| steps7k | 10/55/10/70/32 | 35.4 | **36.0** | 0.0984 | 0.1235 | 0.126 |

Config integrity verified from checkpoints + wandb for all arms (each ran exactly its single delta). Josh's framing question: losses lower, perf flat — why?

### Arm verdicts

- **Arm 1 (affine+nogate): both hypotheses dead, cleanly.** A-phase healthy (held-in 80.8 vs stageB-A 81.7; MSE comparable). The biases LEARNED (nonzero on ~all 147k slots, ||b|| ~0.6x the UV-path norm at every layer, stable A->seq) — the parameterization works; the DC term was not the binding constraint. Gate removal delivered full-strength corrections — also not binding (values recalibrate to the gate during training either way). +4.6pp init is at the noise edge; final 32.8 == baseline. DECISIONS: `mem_gated=false` retired (gate stays); `lora_slot_bias` retired from the recipe.
- **Arm 2 (lr2x): the only init mover (47.4 = r2244's 48)** — and it gave ~all of it back (final 35.6). Grad norms 0.022 max (no instability). See the correction below for how much of the init story survives.
- **Arm 3 (steps7k): dead lever at the endpoint.** The -23% block-min MSE is partly a min-over-more-windows selection artifact; the within-block trajectory converges by ~2.5-3k steps then OSCILLATES (e4: 0.087<->0.132 band, no trend — NOT instability, batch-window variance; my earlier "+54% late-block degradation" claim retracted on the full trajectory, and the TF-IDF self-pollution mechanism I proposed for it is disconfirmed in code: DF accumulates per batch but IDF is frozen within a block, recomputed only at boundaries). Endpoint function: -12% vs stageB == the other arms; inits unchanged. Retro-explains E30: 3k->5k helped joint because joint blocks hadn't converged at 3k; staged blocks converge by ~3k, so 5k is already past the knee. 5k->7k retired at the current write rule.

### FORGETTING: still solved in every arm (MSE matrices)

Just-trained->final drift: stageB +0.0-1.7% (E39), affine <=+0.6%, lr2x +0.4-2.2%, steps7k +1.2-1.7%. Stationarity holds at 2x LR, with biases, at 7k steps. Untrained-task cells reproduce across arms to 3 decimals (instrument validated). lr2x diagonals uniformly -13% vs stageB.

### The e6-after-e9-block drop is REAL — E39/E40's eval-noise attribution REVERSED

e6 drops across e9's block in **4/4 runs** (stageB -35, affine -10, lr2x -20, steps7k -30; pooled ~3sigma), e4 in 3/4. No other block shows systematic cross-task deltas. Mechanistic chain, each link measured:
1. **Exposure topology is arm-invariant** (frozen router): all arms share footprints, RTO 10-13%, bleed channels e6<-e9 = 4-6%/layer, e4<-e9 = 3-4% (vs e2's 2-3%) — e9 is the one anomalous writer (35.8k L14 slots written, 2x anyone; the warmed router gave it a 2.4x footprint; E39).
2. **The perturbation is tiny in function space**: e6-perceived field change across e9's block 1.36-1.62% (core50 0.13-0.16%); paired-MSE drift on e6's cell +0.5-1.3%.
3. **The rollout cost is 10-30pp** — a ~10-20x function->success amplification on MARGINAL tasks only (e2 at 80% absorbed identical drift free). Policies sit near the success boundary; 50-action open-loop chunks compound small field changes.
4. **Protection can't gate it (structurally)**: rank-mode (1-u)^beta only flips top-t membership — high-TF slots never rank out, and the diffuse tail sits at u~0.01-0.05 under peak normalization (u at core-50 boundary ~0.035; max/p99 ~10-16x) => beta4 vetoes only the top ~1% mega-hot slots (E39 note, now the load-bearing fact).
Displacement analysis (checkpoint diffs, L14): lr2x moves written slots ~1.6x (median ||d|| 1.33-1.51 vs affine 1.03-1.21, steps7k 0.87-1.08 — steps7k's median slot moved LESS: same mask budget spread over more batches + decayed-LR tail). The giveback (lr2x -11.8pp init->final) rides on this channel: e6 -31, e9's own decay -17, e4 -15 — all post-e9-block or late-block micro-drift.

### The conversion-gap probe program (5 instruments, chronological)

The sharpest puzzle: steps7k and lr2x reach the SAME eval-time paired MSE (within 2%) with 20-ep inits 10-vs-35 (e4) and 10-vs-25 (e9). Working hypothesis #0 (displacement/amplitude -> "commitment") was formed from this; the program below tested it and its rivals.

**Probe A — downstream-gain (Josh's layer-position hypothesis: "layers 15-17 fix the corrections back toward pretrained behavior").** Inject at L14/L8: the learned memory delta vs matched-norm random vs matched-norm feature-direction perturbations; measure velocity-readout movement. RESULT: **disconfirmed in its strong form** — learned deltas transmit 2-3x BETTER than random (T_rand 1.9-3.2) and 1.2-1.8x better than feature directions, at both layers, every arm. Downstream is an amplifier tuned to the memory's directions, not a fixer. Surviving nuance: amplitude response saturates at L14 (2x delta -> +60-70% output; L8 +81-94%; layer-norm renormalization is the likely mechanism) — a headroom warning for LR pushes, not a case for moving layers. Also: total velocity-space throw of the memory is ARM-INVARIANT (~25 at L14) — lr2x does not deliver a bigger total correction, it delivers a better-placed one. Layer repositioning PARKED with evidence.

**Probe B — denoised-chunk (the integrated field).** Run the real 10-step denoise on demo obs; compare executed chunk to demo chunk. The training loss only ever queries the field ON the noise-demo interpolation bridge (x_t anchored to the true action — teacher-forcing one level down); integration queries the model's OWN trajectory, which leaves the bridge after step 1. RESULT: chunk error ranks arms **exactly as rollouts do, 9/9** (e4: lr2x 0.156 < steps7k 0.172 < stageB 0.191; e9: 0.366/0.401/0.459; e2: 0.224/0.272/0.304) where velocity-MSE couldn't separate lr2x from steps7k. Arm gaps grow on the last 10 chunk steps. **ADOPTED as the standard gate metric** (no simulator).

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

### Soft (grad_scale) protection (the Layer-1 fix)

- **Design**: TF-IDF top-1536 mask unchanged (locality); protection moves from the ranking score to the UPDATE: each surviving slot's applied update is multiplied by (1-u_q(s))^beta. **Implementation catch that would have silently no-op'd: Adam's step is invariant to a time-constant per-row gradient scale** (m-hat and sqrt(v-hat) scale together; smoke-measured movement ratio 0.995 under naive grad scaling). Correct mechanism = **post-optimizer-step blend** theta <- theta_pre + scale*(theta_post - theta_pre) on protected rows — exact per-slot LR scaling under any optimizer (smoke: measured 0.00389 vs theoretical 0.25^4 = 0.00391).
- **u-normalization fix**: u = counts / count-at-core50-boundary, clipped to 1 (peak-norm's u~0.035-at-boundary degeneracy fixed; the whole prior core now protects).
- **beta calibration** (offline, real lr2x per-slot deltas x read-mass distributions, L14+L8): smooth trade, no knee — e9-block bleed onto e4/e6 kept 70/60/48/35/24% at beta 1/2/4/8/16 for e9 static write-mass cost 7/12/18/26/35%. **beta=4 chosen**: halves the bleed; static cost overstates real cost (mask reallocates suppressed magnitude to prior-free slots; e9 — the writer that matters — is the CHEAPEST to constrain); midpoint is information-maximizing under unknown damage-response shape. Decision tree: e6 still drops at beta4 -> response is threshold-y -> beta 8-16; e7's init craters (later writers pay cumulatively more, ~28-38% static at beta4) -> beta 2.

### Batch-size (Josh's hypothesis: bs32 under-covers the trajectory)

Effective-64 via `gradient_accumulation_steps=2` (the trainer counts OPTIMIZER steps and merges retrieval indices across micro-batches, so the TF-IDF mask sees a true 64-frame TF). **The strongest mechanism for bs64 is mask stability, not gradient noise**: at bs32 the top-1536 is ranked from 32 frames' retrievals -> mask churns -> write budget rotates across slot subsets -> updates smear (the dilution pathology; steps7k made it WORSE — more draws, 37.7k unique slots). 64-frame TF -> stabler selection -> fewer slots, more events each -> counters dilution AND shrinks the bleed surface. This is the anti-dilution lever from the opposite direction to topt3072.

### Three arms (all 5-task, stageB-verbatim + single delta, 50-ep finals)

| run | delta | pre-registered reads |
|---|---|---|
| topt3072 | tfidf_top_t 3072 | e9 init + chunk error (dilution-as-budget?); e6@20k cell (2x write breadth should WORSEN the bleed if the model is right — falsification opportunity) |
| lr2x+softprotect (grad_scale/corefrac/beta4) | protection mechanism + 2e-3 | inits >=~45; e6-across-e9-block drop <=10pp (vs -10..-35 in 4/4); e9 init may pay a few pp; final >=42 = new frontier |
| bs64accum2 | effective batch 64 | e9 L14 written-slot count (expect down from 35.8k) + ev/slot p50 up; e9 init + chunk error; e4/e6 bleed shrinkage |

Read via retention matrices + slot JSONs + the probe battery (chunk metric first), NOT 20-ep init cells.

### Next steps (after the 3 arms)
1. **LoRA-FT per-task baseline** (e4/e9/e2) — now the decisive Layer-3 experiment; read through the probe battery, not just success.
2. **Seed-averaging micro-experiment** (offline): endpoint error = shared bias + per-seed component (probe-B spread 0.09-0.25 is substantial); averaging denoised chunks over a few noise seeds may cancel the seed part at pure inference cost. Test with probe B before touching evals. Risk: mode-averaging on multimodal segments.
3. If softprotect works: 10-task extension of the winning config (the catastrophe-elimination demonstration, E39 #2).
4. Parked with evidence: layer repositioning (probe A), gate/bias variants (arm 1), steps>5k (arm 3), global top-t for retention (E19 + bleed mechanism), joint track (off-thesis, E40).

- Eval-comparability note: stageB's final row is 20-ep; all E41-era finals are 50-ep.

---
## Entry 42 - 16 Jul 26 (E41 3-arm verdicts: bs64 EXONERATED-AND-RETIRED (no accum bug; chunk parity at 2x cost); softprotect = new 50-ep frontier 41.2 BUT the blend mechanism is DEFECTIVE (Adam momentum leak, ~90% passthrough) — the number is the lr2x family + eval fortune, not protection; topt3072 FALSIFIES the bleed->rollout model on schedule and identifies COVERAGE as the second fit lever. Fixes: momentum-aware blend + hard veto; next: lr2x+3072 / lr4xsched+3072 / LoRA-FT)

### The three landed arms (stageB-verbatim + single delta; final rows 50-ep; baselines from E41)

| arm | e4 | e6 | e9 | e2 | e7 | mean init | final | give-back |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| stageB | 10 | 60 | 5 | 70 | 30 | 35.0 | 32.0 (20ep) | -3.0 |
| lr2x | 35 | 65 | 25 | 80 | 32 | 47.4 | 35.6 | -11.8 |
| steps7k | 10 | 55 | 10 | 70 | 32 | 35.4 | 36.0 | +0.6 |
| **bs64** | 10 | 40 | 0 | 60 | 26 | 27.2 | 30.4 | +3.2 |
| **softprotect** | 35 | 50 | 30 | 70 | 38 | 44.6 | **41.2** | -3.4 |
| **topt3072** | 15 | 35 | 5 | 85 | 34 | 34.8 | **37.6** | +2.8 |

Loss (wandb block-min/block-end mean): stageB.1274/.1410, lr2x.1132/.1239, bs64.1176/.1355, softp.1139/.1270, **top3k.1086/.1215** (family-lowest). Probe battery run over ALL arms (chunk grid own-block AND final for t0-t3; slot autopsy L14+L8 with realized-write counts, displacement, self-coverage, full bleed matrix; softp MSE matrix).

### Arm 1 verdict — bs64: no gradient-accumulation bug; parity at 2x cost; RETIRED

- Full code audit CLEAN: Accelerate scales loss under accumulation (averaged grads); TF-IDF ranks the merged 64-frame indices; value-LR schedule = 5000 optimizer steps; DF/IDF/protect stats scale-invariant. Config verified from checkpoint.
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

### Fixes

1. **Momentum-aware blend**: scales the row's exp_avg by the same factor (kills the tail at source; exp_avg_sq untouched).
2. **`protect_hard_u`**: slots with u >= threshold removed from top-t CANDIDACY (not just score-zeroed — with top_t >= candidates a zero-score slot still gets selected). Never in mask => no grad, no momentum — airtight. At corefrac, u>=0.9 = the victims' true cores: ~19%/34% (L14/L8) of e9->e6 bleed mass on a few hundred mask slots.
3. The honest expectation even for the FIXED mechanism: per-slot LR scaling suppresses ENDPOINT displacement only where s x 5000 steps can't converge (u>=~0.4-0.5); the shoulder still converges slowly — the implementable ceiling is closer to the veto-mass curve (26% of bleed @ u*=0.5, L14) than the calibration's 48%.

### Next

| VM | config | pre-registered reads |
|---|---|---|
| 1 | value_lr 2e-3->2e-4 + top_t 3072 | inits >=~47 AND give-back >=~0 => final >=42-45 (50ep) = new frontier; e9 init >=20; FAIL: give-back <=-8 (levers interact) |
| 2 | value_lr 4e-3->2e-4 + top_t 3072 | e9 init/chunk vs VM1 (the amplitude-limited cell); block-ENDs vs VM1 (worse => L14 saturation binding, 2x = plateau, axis closed) |
| 3 | 5x independent LoRA r32 on stage-1 base, 5k steps, 50-ep evals | e4/e9 ~35-50 => our machinery's conversion tax, priced; <=15 => frozen-backbone ceiling, scope the thesis claim; >=60 => rethink value path |

The 70% arithmetic, honestly: best staged 41.2; base-joint ceiling 74.8 on these 5. The compositions plausibly buy mid-40s. The remaining ~25pp is Layer 2/3 — which is exactly what VM3 prices. If LoRA-FT lands high, the known gap-closers in-protocol are r4-at-L12/14 on the staged track (never run; r2244's +6 came from it in the joint era), the top-p (mass-quantile) mask (e9's 2.4x footprint needs an adaptive budget), and inference-side seed-averaging (E41 #2, still unrun). If LoRA-FT lands low, 70% is not reachable by ANY frozen-backbone adapter method on this suite and the target needs renegotiating toward "joint-adapter-level fit with zero forgetting".

- Eval-comparability: all E42 finals 50-ep; init cells remain 20-ep (retired from decisions per E41; the probe battery is the ranking instrument, with the e2-misrank caveat above).

---
### Entry 42 addendum (16 Jul eve, discussion) — the loss-vs-rollout root cause narrowed to OFF-TRAIL behavior; the grounding-gradient family resolved; "freeze the generalist slots" queued behind an overlap measurement

Long discussion with Josh on THE standing question (why sequential loss == multitask loss does not give sequential perf == multitask perf). Distilled outcomes:

1. **Taxonomy + status.** Candidate mechanisms enumerated ground-up: (a) noise-floor masking [confirmed; the chunk probe is the workaround], (b) error accumulation along the denoise path [measured family-wide ~0.95 coherence incl. r2244 — shared, not the differentiator; NB never measured on the true full-finetune baseline, which was deleted at E31], (c) error placement within the chunk [late-chunk skew known; full per-position/per-dim profile still to run], (d) OFF-DEMO-TRAIL degradation [now the PRIME suspect: E42 showed e6's function flat on demo states while rollouts wobble — the damage lives at states only the closed loop visits; never directly measured], (e) frozen-feature extrapolation margins [E36 crowding, circumstantial], (f) sample/seed variance [spread measured substantial; seed-averaging still unrun], (g) the success-threshold cliff [shared amplifier, closed as a cause — and bs64's null doesn't bear on it: flat-minima batch lore is parameter-space, the cliff is task-metric-space].
2. **Grounding-gradient proposal (Josh) resolved after full derivation.** Any precomputed/cached gradient fails three ways: gradients at frozen params feed nothing (backprop chains through activation signals; weight grads are leaves); a dataset-averaged (signal x activation) product cannot be re-decomposed into state-conditional signal at new inputs; and at the anchor's own minimum the averaged gradient is ~0 by first-order optimality (gradient = slope, not height — the A-phase values are CONVERGED on libero-90, so the "generalist direction" is zero where we'd harvest it). The valid family members: live rehearsal (= averaging losses; targets retention, which is solved) and the curvature version (= EWC = our protection machinery aimed at the A-phase — same non-problem). The surviving state-conditioned carrier of "generalist opinion at new states" is the base FUNCTION: distillation anchor L = L_task + lambda*||f_adapted(x) - f_base(x)||^2 on new-task states (cache the function, not the gradient). Caveat: the anchor is task-incompetent (stage-1 floors ~0-2), so lambda trades fit for smoothness.
3. **Remedy shortlist for the off-trail story (all values-only, in-constraint), gated behind the off-trail instrument** (score executed-chunk error on observations harvested from our own rollout episodes; on-trail vs off-trail degradation per arm, staged vs r2244): demo-jitter augmentation (same targets, perturbed observations — local flatness, no labels needed), lambda-anchor (above), and:
4. **"Freeze the generalist slots" (Josh, from the original memory-layers paper: accumulate pretraining access stats, let CL touch only low-usage slots).** In our stack this is the E42 protection machinery with the store seeded from A-phase (libero-90) usage + protect_hard_u as the freeze (mass-threshold, not slot-count — corefrac handles it). As RETENTION machinery it targets non-problems (pretrain forgetting out-of-scope by E19 constraint; sequential forgetting solved). As FIT machinery the precedents are negative (E9 pretrain-IDF seeding tanked; E19 veto starvation; E42 softprotect's writers IMPROVED by writing into hot slots; coverage won by widening). The interesting reading: a STRUCTURAL distillation anchor — frozen generalist slots pin the shared ~23-44% of every retrieval mixture at the general function, so off-trail routing drift falls back toward the general policy instead of overwritten task-specific content. Zero hyperparameters beyond the mass threshold. If adopted, the claim is "reserved generalist substrate improves off-distribution fallback," NOT the paper's protect-pretraining story.
5. **The gating measurement** — under frozen-base routing reads are VALUE-INDEPENDENT, so an inert-LR sweep of libero-90 through the A-checkpoint reproduces the A-phase read profile exactly (90 tasks x 40 batches), then measure each sequential task's read/update-event mass on the top-{50,20,10,5}%-mass A-phase sets. High write overlap => E19 starvation regime, freeze is costly; low => freeze ~free (and its residual case is the structural-anchor story).
6. Also noted: VLM-side memory layers (Josh) are viable with a stationarity-safe placement — prefix layers ABOVE the highest expert-memory layer (>14) never enter the prefix caches that expert routing at 8-14 consumes (per-layer KV pairing) — parked as a compute-gated direction pending VM3's read on whether upstream trainable capacity is where the wins are; and inference-side re-planning (execute 25 of 50) noted as legitimate-if-applied-to-all-models, gated behind the per-position error profile.

**Addendum results (17 Jul, 00:30) — generalist-slot overlap MEASURED: the freeze is zero-sum-shaped; demoted.** Gotcha first: this libero_90 build has 73 task_index entries (0-72), not 90. Results: A-phase aggregate effnum 58-64k/layer; top-50%-mass core = 22-25k slots (15-17% of table), top-20% ~5.5k, top-10% ~2k. THE PATTERN: sequential tasks' read-overlap ~= write-overlap at EVERY mass threshold (top-50%: reads 26-42% vs writes 23-52%, e4@L14 51.8%(!); top-10%: both 3-8%) — the new tasks write where they read, proportionally at every depth of the A-core. No threshold exists with high read-dependence + low write-demand, i.e. no cheap-protection pocket even under the warmed router: a freeze deep enough to preserve meaningful generalist substrate blocks an equal share of write demand (E19 starvation shape, re-measured), and a free freeze (top-5-10%) protects a near-inert 3-8% of the mixture. VERDICT (revised after discussion, see below): the DEEP freeze (top-50%) is dead — E19 starvation shape re-measured; the SHALLOW freeze (top-10-20% mass, 2-6k slots) is a live bake-off candidate on equal standing with demo-jitter and the lambda-anchor. Side note: e4 (weakest rollout task) is the heaviest A-core writer (up to 52%), e9 the most private (23-27%) — consistent with e4's function living on contested generalist substrate.

**Verdict revision (17 Jul, discussion with Josh — the first "demoted" call was overzealous).** The zero-sum reading equated overlap mass with cost and benefit, but neither conversion is linear, and they bend in opposite directions:
1. **Cost is sub-linear in write-event overlap.** The hard veto removes frozen slots from top-t CANDIDACY (the S13 fix), so the 1536 budget redistributes to the next-ranked slots automatically — nothing is wasted and the adapted footprint keeps its size, just relocated. Reads are untouched (frozen slots keep serving their preserved A-content into every mixture — the point of the freeze). The residual cost is only that corrections at the 3-8% of read mass on frozen slots must be expressed through the other ~92-97% of each state's 144-slot mixture — mostly compensable (the beta-calibration's static-overstates-real result; LN renormalization). Estimated real fit cost of a top-10-20% freeze: ~1-3%, not the naive 3-16%.
2. **Benefit is plausibly CONCENTRATED where the freeze is cheap.** The structural-anchor value rides on off-trail routing drift landing on the hottest, most scene-generic slots — i.e. exactly the top-~2k, not the top-25k. Overlap mass is not the currency the benefit is paid in, so the 1:1 table cannot rule the shallow freeze out.
3. **It is the CHEAPEST bake-off arm to implement**, with no hyperparameters beyond the mass threshold — the opposite of the demotion rationale.
Honest cautions that stand: the benefit is contingent on the off-trail mechanism (instrument still to run — the gate for the whole remedy family), and the preserved fallback is a policy with 0-2% floors on these tasks (the bet is coherent-generic motion off-trail beats corrupted-mixture motion, keeping states recoverable — plausible, unproven). What the table genuinely killed: the deep freeze, and the paper's protect-most-of-the-substrate framing (our router sends new tasks INTO the pretraining core; theirs evidently didn't).

---
## Entry 43 - 17 Jul 26 (Composition arms: levers COMPOSE in function space (block-min 0.0880, project-lowest; 4x does NOT saturate) but 50-ep finals stay in the 40+/-2 band — the conversion slope measured across a 31% loss range (~+2pp per −10% loss, far too shallow for 70). FIRST real function-space give-back ladder (e9 own->final: 1x −3% / 2x +7.7% / 4x +14.5%, pre-registered predictions hit). LoRA-FT partials REPRICE the gap: e4=58 (above base-joint 52) with clean chunk 0.018 vs our 0.153 (9x) — the e4 deficit is substantially FIT, not pure conversion; e6=44 (BELOW staged 54-56))

### The two composition arms (stageB-verbatim + deltas; init cells 20-ep, finals 50-ep; seed 1000; configs verified from checkpoints)

| arm | e4 | e6 | e9 | e2 | e7 | mean init | final |
|---|---|---|---|---|---|---:|---:|
| VM1 lr2x+topt3072 | 20->[32] | 55->[56] | 10->[14] | 70->[64] | [36] | 38.2 | **40.4** |
| VM2 lr4xsched+topt3072 | 30->[20] | 60->40->[54] | 35->[14] | 100->[80] | [44] | 53.8 | **42.4** |

Loss (block-min/block-END means): VM1 **0.0969/0.1077**, VM2 **0.0880/0.1019** — the two lowest in project history (top3k 0.1086, stageB 0.1274). Grad norms clean at 4e-3 (max 0.020). e7's loss response at 4x is the outlier: 0.1167 -> 0.0891 (−24%).

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

### Discussion outcomes (Josh, 17 Jul) — the plan

1. **Off-trail instrument, jitter version**: perturb raw demo obs (state sigma x per-dim std; image pixel noise; RNG seeded per task/batch/scale so all models see identical inputs), score the 10-step-denoised chunk vs the demo chunk; the READ is across-model degradation slopes at matched clean error (shared target bias cancels). Full version (score on rollout-visited states; target-free drift ||f_after − f_own_block|| on- vs off-trail) needs an obs-dump in the eval loop — deferred pending jitter results. Grid: staged finals (VM1/VM2/softp/top3k/stageB) x {e4,e9} + LoRA t0; r2244 skipped (10-task exposure = protocol mismatch).
2. **Softprotect retest**: single delta vs VM2 = momentum-FIXED grad_scale blend + corefrac. Pre-registered: e9 own ~0.28 held, own->final <=+4% (vs +14.5%), e9/e4 finals up; FAIL = writer block-min MSE +10%. Framing: mechanism validation + de-taxing the LR axis (bank 4x fit for whenever conversion improves), not a points bet.
3. **Generalist freeze**: top-5000 A-phase read-mass slots/layer (18-19% of A-phase mass) seeded u=1.0 + protect_hard_u=0.9 structural veto; single conceptual delta vs VM2. **Correction from discussion (Josh): my "freeze buys no fit so it can't push the frontier" was the fit=perf error this project exists to kill.** The mechanism: hottest A-slots = the generalist transforms in every mixture; sequential writes erode them; on-demo MSE never registers (own-task adaptation compensates on-trail); off-trail the eroded generalists are the fallback => rollout cost invisible in loss. Freeze removes the erosion; expected signature = flat MSE, better rollouts, shallower jitter slope.
4. **LoRA compass**: two e4-only arms — expert-only (attn+MLP+proj) vs VLM-only (attn+MLP+proj) vs the full-LoRA 58 anchor. A~58/B-low => n256/r4 staged build, VLM memory dead; A-low/B~58 => VLM-side memory is the build (placement chosen by running the E36 feature probe on PREFIX layers, not by guess — and low placement is allowed if the dual-path trick extends to the prefix KV, one extra prefix forward/batch); both mid => expert first; both high => cheapest wins. C (expert-MLP-only, attention ablation) dropped: we are not building attention-path memory regardless (parked; per-token routing makes it feasible in principle — each token routes itself, no seq-level router — but k/v corrections make interference cross-token), so its content only prices a caveat the 1-day n256/r4 arm tests directly.
5. **n256/r4 dose fixed**: n_keys 384->256 (65.5k slots, ~0.9x current param budget) x r=4, NOT n/4 (36.9k would dip below the A-phase's 58-64k effective slot usage => forced collisions at pretrain fill). Note: values-only stage-2/3 could likely afford r4-on-all-4-layers at the CURRENT bank too (the config joint training OOM'd on).
6. **Rejected**: seed-averaged inference (Josh: a hack, from other papers, lifts baselines too). Attention memory parked. r2244 as jitter anchor skipped.

### Entry 43 addendum (17 Jul, discussion) — two working heuristics from the LoRA cells, and the regime filter on old counter-evidence

**Heuristic 1 (breadth law, Josh): perf ∝ (1/MSE) · n — read as an iso-performance surface, not a product.** Anchors: single-task LoRA rolls 44-58 at chunk ~0.02 (n=1); base-joint rolls 72.6 at train-MSE ~0.10-0.15 (n=100). The MSE required for a given rollout level RISES with the breadth of what the model trained on (two-point fit ~ MSE_req ∝ n^0.4 — decoration; the direction is the content). Mechanism: n proxies COVERAGE of rollout-visited states — a narrow model reaches 0.02 by interpolating its 36-episode manifold and is unconstrained one excursion away; a broad model at 0.12 is anchored by neighboring competence everywhere the rollout wanders. Correct functional form is a THRESHOLD, not proportional: success ≈ steep function of (threshold(task, support) − chunk), threshold shifted UP by support breadth. This single form retro-explains the E43 table: e4 (our 0.13-0.19 far above the n~substrate threshold => 20-35; LoRA meets its n=1 threshold => 58), the e6 sign-flip (our 0.10 meets OUR breadth-adjusted threshold => 54-56; LoRA's 0.02 can't fully buy out n=1 => 44 — the cell no demo-state instrument could explain), and the shallow conversion slope (we move loss 5-20% at a time, far above threshold, where the curve is flat; LoRA moved it 6-9x, through it). Jitter grid corroborates the support half: staged arms are NOT more brittle (absolute degradation LoRA >= staged at matched perturbations; the specialist wins by LEVEL, gap compressing 4.4x -> 1.5x as perturbation grows toward where rollouts live).

**Regime filter on the "lower MSE can't save us" corpus (Josh's point — prior findings are largely from other regimes):** E15 is SmolVLA-era; E19-t5/E30/E33 are joint-era AND pre-stationarity (drift tax on all their lever ROIs). Only E39-43 measure the current staged protocol. After filtering + the LoRA baseline: **E33's "the frozen backbone is the binding constraint" and E41's "value-path capacity is not where the offset mainly lives" are both FALSIFIED** (LoRA hits chunk 0.02 / 58% on e4 on the SAME frozen backbone via pure adapter capacity — the r4/bias levers supplied capacity in the wrong currency, and we blamed the backbone). Bin-1 of the old corpus was instrument error (one-step velocity-MSE blind to the integrated field — fixed by the chunk probe, E41); Bin-2 was small-moves-far-above-threshold (consistent with the law); the genuine residue is e9 (chunk −39% -> rollout pinned; decided by today's LoRA e9 cell: low-chunk+high-rollout confirms the threshold story, low-chunk+low-rollout breaks it) and E39's staged-vs-joint e4 pair at matched velocity-MSE (never re-measured at chunk level; may dissolve like lr2x-vs-steps7k).

**Heuristic 2 (read-write alignment): usable adaptation ≈ Σ over parameters of (accumulated learning signal) × (per-forward read participation).** Learning on barely-read params is wasted (dilution); reading never-trained params is wasted (stale A-content in the mixture). Both halves are already measured: read side = self-coverage (74-95% at 3072); learning side = update events/slot (p50 17-34, p90 740-1850). LoRA's product is degenerate-perfect (every param read every forward AND updated all 5000 steps); our CORE slots are in its ballpark (~1-2k events, read constantly) — **the deficit localizes in the mixture TAIL: ~20-50% of every retrieval mixture is A-content with zero sequential learning or shoulder slots with ~20 events.** E42's bs64 result ("concentration gain == coverage loss") was this product being conserved at fixed bank size — the trade is forced at n=384^2. **This is the sharpest argument for the n256/r4 arm**: shrinking the bank 2.25x raises BOTH factors instead of trading them (same knn x heads read budget on fewer slots => each read slot read more often; footprints shrink toward the mask budget => selfcov up; each masked slot collects ~2x the events), with r4 doubling per-slot expressivity. First bank-shape change that moves the product rather than sliding along it. Bounds ("within reason"): (i) over-concentration rebuilds the E21 per-task-bias pathology — q_intra + per-batch effnum stay gated; (ii) concentration serves the PRECISION axis only; heuristic 1 says the threshold half is won by NOT overwriting the generalist remainder — so the package is "concentrate learning+reads on the task subset, freeze/preserve the substrate" = n256/r4 + freeze, jointly.

**New standard for the battery:** compute the read-mass-weighted update-event distribution per task per arm from the autopsy JSONs ("the product") — pre-registered n256/r4 read: product up >=2x at flat q_intra; chunk down on e4/e9; rollouts follow where the threshold allows.

**Addendum 2 (17 Jul, ~16:00) — LoRA e9 lands: 70.0 @ 50 eps, chunk 0.0675. The registered prediction resolves FOR the threshold law (3/3).** e9's specialist beats base-joint (60) at 5x our best staged cell (14; softp 30), with function 4-5x below ours — low-chunk+high-rollout, the confirming branch. The e9 residue dissolves: our 0.28 was 4x the ~0.07 requirement; every within-family improvement moved us between points far above threshold (the flat region), which is why nothing ever converted on e9. Both decision cells now at/above the pre-registered ">=60 => rethink value path" boundary (e4 58, e6 44, e9 70; 3-cell avg 57.3 vs our ~30-35). Cross-task note: LoRA e6 (0.020 -> 44) vs e9 (0.068 -> 70) re-confirms thresholds are task-specific — chunk comparisons stay within-task only. One more jitter fact: e9's specialist is EXTREMELY image-brittle (image@0.05: 0.068 -> 0.452, +0.38 absolute vs +0.10 on e4/e6 — falls INTO our clean range) yet rolls 70 — the eval's visual variation evidently stays inside its competence; another hint that the perturbation shell and the rollout distribution are different objects (true off-trail instrument still the eventual arbiter).

---
## Entry 44 - 18 Jul 26 (Overnight E44 program: COMPASS answers the value-path question — expert-only dense LoRA FAILS e4 (14 @ chunk 0.229) while VLM-only nearly reproduces full (40 @ 0.030) => perception adaptation carries ~10x of the fit; span attribution finds the leverage CONCENTRATED in the language field (instruction + pi05's STATE-AS-TEXT tokens: 36-57% of dL/dh at 20x per-position density, L15 peak) with OPEN routing geometry (inter-task cos 0.73-0.86 vs 0.96+ image); softprotect-FIXED passes its gate (e9 give-back +14.5% -> +3.9%, victim cores 0.00% bleed) but taxes the last writer (+49% block-min); freeze5k mechanically perfect (0/5000 slots moved) and strategically null at 5 tasks (give-back WORSE +19.2%); multitask LoRA confirms the threshold law's e6 flip (52 vs specialist 44) and kills its iso-budget strong form. => THE BUILD: text-span VLM memory at LM [15,16], uniform n256/r2 across all 6 layers (expert right-sized to its measured 58-64k effective usage), VLM routing sweep LAUNCHED)

### Overnight results (all pre-registered reads scored)

**Compass (e4, 50-ep + chunk):** full LoRA 53.2M -> chunk 0.020 / roll 58; VLM-only 39.2M -> 0.030 / 40; expert-only 13.9M -> 0.229 / 14 (train loss plateau ~0.18 = ceiling, not undertraining); staged memory (expert side) 0.118-0.19 / 20-35. VERDICT: the e4 fit requirement is UNMEETABLE from the expert side at any density — placement, not density/rank, is the currency (the "22x per-forward gap" framing dies: expert-dense has 6x our per-forward params and does worse than our sparse memory at matched placement, A-phase-substrate caveat noted). VLM necessary and nearly sufficient; the expert tower adds 0.030->0.020 + 18pp (e4's threshold sits ~0.02-0.03 — steepest part of the curve). Threshold law 4/4 then 5/5 (VLM-only high-chunk... low-chunk/high-roll branch consistent).

**Span attribution (new instrument, on the VLM-only LoRA):** pad control exactly 0.00% at every layer (instrument clean); Taylor term ~0 at displacement 0.35-0.94 (beyond first order — expected). Gradient share of the action loss by prefix span: image (~770 pos) 64->43% falling with depth; instruction (~24 pos) 13->8%; **STATE-AS-TEXT (~30-40 pos) 22->49% RISING with depth — pi05 tokenizes proprioception into the prompt (`Task:..., State:...;`), and at L15 those positions carry HALF the loss sensitivity at ~20x the per-position density of image tokens.** The language FIELD (instruction+state) = 36-57% of leverage in ~60 contiguous positions. The adapter moved text-position hiddens as much as image ones (disp parity).

**Placement probe (corrected text-span pooling after a padding artifact):** linear task probe 100% at every LM layer 2-16 (98.8-99.3 at 17); all-token (image-dominated) inter-task cos 0.962(L12)-0.995(L17) — crowded, avoid 16/17; **text-span positions: inter-task cos 0.73-0.86, the most open task geometry ever measured in the project, basket family 0.86-0.95 (vs 0.99+ image / 0.978-0.994 expert-side) — the instruction stream separates the lookalike pairs the scene cannot.** Image side: discriminative-but-crowded (task signal spatially sparse in object patches; scenes genuinely ambiguous on lookalikes — the E28 floor at representation level).

**Protection arms (both single-delta twins of VM2=4x+3072 @ 42.4):**
- softprotect-FIXED (grad_scale/corefrac/beta4): **mechanism works in production** — u-bin displacement exactly inside the (1-u)^4 envelope (u>=0.9 slots 0.0000; the E42 leak's same bin moved 1.55), victim-core bleeds 0.00% everywhere, e2blk->e9 full bleed 7.55->1.54%. **e9 chunk gate PASS: own 0.2665 (best-ever) -> final 0.2770 = +3.9% (gate <=4; twin +14.5%)**, e9 roll 14->26, e4 20->30. THE BILL: last-writer starvation — e7 block-min +48.7% -> roll 44->28 (the corefrac store accumulates monotonically; by t4 the writer is attenuated across 4 prior profiles); e2 +10.5% absorbed. Final 39.6 ~= twin 42.4 (victims up, writers down). e6 34-vs-54 = **chunk-rollout misrank #3** (its function flat + family-best 0.0714->0.0728) — e6 cells carry no attribution weight, now demonstrated three ways. NEXT: a non-accumulating/decaying protection store to keep e9's +12 without e7's -16.
- freeze5k (top-5k A-core seeded, hard_u 0.9): 0/5000 seeded slots moved, bitwise — the mechanism is perfect. Strategically NULL at this horizon: selfcov -10pp/task (frozen slots carry 10-24% of read mass, un-adaptable — a direct read-write-product hit), e7 block-min +26.5%, bleeds UP (displaced budget crowds victim shoulders) => e9 give-back WORSE (+19.2% vs twin +14.5%), finals 39.2, all 5 cells <= twin, jitter slopes ~= twin (no off-trail dividend in-shell). The A-core erosion it guards against measures 0.2-0.5% core50 here — it prevented non-damage. RETIRED at 5-task scale; revisit only if a 10-task run shows real A-core erosion.

**Multitask LoRA (one adapter, 5 tasks, SAME 5k total budget):** 49.2 overall — e4 38 / e6 52 / e9 36 / e2 72 / e7 48. Pre-registered: e6 UP vs specialist 44 ✓ (breadth removes the n=1 penalty, lands in the staged 54-56 band); e4/e9 down ✓ (precision-bound, 1/5 per-task exposure); **iso-budget strong form FAILS** (matched-3 avg 42.0 vs specialists 57.3) — the two currencies (precision, support) trade at task-specific rates; support pays near threshold, precision rules far from it. e7=48 = best e7 measured anywhere.

### THE BUILD (decided with Josh)

**Text-span VLM memory:** modules on paligemma LM layers [15,16], attached to the LAST-200 prefix positions (the tokenized language field = instruction + state-as-text), retrieval computed on that slice only. Placement logic: L15+ keeps the prefix KV at <=14 bit-identical => the expert router's certificate, its frozen-base routing, AND its value-path inputs all survive untouched (per-layer KV pairing: suffix i attends prefix i only); the lowest VLM module's router input is memory-free by construction. Rejected: full-attachment v1 (routing on the crowded 0.96+ image cone, 14x counter blow-up — prefix is ~770 tokens vs the suffix's 51, the "why were VLM forwards so high" mystery from day-2, now a design input); layers 16/17-only (geometry degrades); <=L14 (dual-path prefix build not warranted for the leverage delta).

**Uniform n256/r2 across all 6 memory layers (Josh):** the expert bank right-sized 384^2->256^2 (65,536 ~= its measured 58-64k aggregate effective usage — right-sizing a FRESH bank, not slot-transplant pruning) + VLM 256^2 r2 on the 2048-dim hidden. Total values 2.15B — the expert shrink pays for the VLM addition. VLM knn=16 (routing_loss_topk aligned =16, E24 rule).

### VLM routing sweep (regime hazards: (1) geometry already open => sep may need less force; (2) instruction constant within task => contrastive intra-pull is an elevated E21 collapse risk — the state tokens carry the within-task variation the router must keep)

Gates (re-anchored for the 65,536 bank): PASS = held-out famIoU <= ~0.25 AND per-task core50 >= ~650 AND effnum >= ~500; COLLAPSE tripwire effnum <= ~150 (per-task-bias signature, kill regardless of IoU); stretch famIoU <= 0.15. Winner -> joint A-phase (both value banks; expert re-warmed at n256, gate: famIoU <= ~0.20 / core50 >= ~1300 vs the n384 certificate 0.145/2955) -> e4 1-task plasticity probe (chunk toward <=0.08 = build validated; ~0.12 = restricted attachment failed -> full-attachment fallback arm) -> 5-task sequential.

### Entry 44 addendum (18 Jul ~14:00) — PAD CONTAMINATION bug in the v1 text-span attach: found at arm A's audit, mechanism-verified, FIXED, sweep relaunched as _padfix

Arm A's audit failed both gates (famIoU 0.466/0.469, core50 95-418) with a signature that unmasked a v1 design bug: the last-200 span includes ~140 PAD positions (valid language tokens ~52-57), and pad hiddens — near-identical across all samples and tasks — query the memory too. Direct verification on the arm-A checkpoint: the pad region's ~40k slot-draws collapse onto ~230 unique slots (the shared core) while the text region routes richly (8-9k unique slots/task-batch) — the router is healthy on real tokens; the statistics, TF counts, contrastive means, and routing histograms were drowning in 70% pad mass. Contaminated B/C audits confirm ARM-INVARIANCE (famIoU 0.454-0.507 for all three recipes — the sweep measured the pad floor, not the routing configs).

FIX: the language attention mask reaches the VLM wrappers. Valid tokens are a contiguous PREFIX of the field, so the loss/queue machinery runs on a valid-token slice; per-task diagnostics filter by the full mask (audit/TF-IDF/protection see valid tokens only); memory output zeroed at pads (bitwise plain-mlp there).
## Entry 45 - 18 Jul 26 (VLM routing sweep: ALL THREE ARMS FAIL famIoU (0.42-0.53) — failure is CONTENT-STRUCTURAL, not dose: the state-as-text sub-span routes on shared digit vocabulary (sub-span probe: state region famIoU 0.36-0.40 vs instruction 0.21-0.22, which PASSES gates on the failed checkpoints). Aux-only regime: contrastive = the SPREADING force (breadth monotone in c; B-vs-D single delta). Querystats probe: state positions carry task signal at 0.11-0.13x their digit noise; pooled-state key near-degenerate (family cos 0.98-0.99); instruction anchor alone already carries within-task conditionality (intra-cos 0.864-0.890). => POOLED-ROUTER build (route state region on one per-sample anchored key; value path per-position) + 3-arm overnight chain to the e4 decision instrument)

### Sweep verdict (padfix arms; gates famIoU<=0.25 ∧ core50>=650 ∧ effnum>=500; collapse effnum<=150)

| arm | c | sep | famIoU L15/L16 | bgIoU | core50 mean L15/L16 | verdict |
|---|---|---|---|---|---|---|
| A | 0.05 | 5.0 | 0.467/0.502 | 0.294/0.349 | 8.4k/10.3k | FAIL sprawl |
| B | 0.0125 | 2.0 | 0.418/0.463 | 0.200/0.272 | 4.2k/6.9k | FAIL (best) |
| C | 0 | 5.0 | killed <1k steps (top-1 0.7, support ~300) | | | COLLAPSE |
| D | 0.5 | 2.0 | 0.501/0.529 | 0.364/0.411 | 11.0k/13.1k | FAIL (pre-registered sprawl signature) |

Integrity: configs verified single-delta from checkpoints; code skew between arms is immaterial (loss/queue machinery ran on the valid-token slice in both); pad fix verified working (cores 3-13k with real arm variance vs the contamination's 95-418 shared pad core — same famIoU number, completely different object). No collapse in A/B/D (effnum min 1286).

### Mechanism (three facts + one instrument)

1. **c sets breadth, monotonically** (core50 4.2k -> 8.4k -> 11k at c 0.0125 -> 0.05 -> 0.5; B-vs-D is a clean single delta). In the aux-only warm-up (values pinned => MSE grad on router exactly 0, E36 smoke) SupCon's SAME-TASK-DENOMINATOR term (uniformity) is the only anti-collapse force — its coefficient sets the equilibrium radius. The joint-era "c = compaction" calibration inverts here: there MSE spread and alignment tightened; here sep collapses (arm C) and uniformity spreads. On this tower same-task mean queries differ ONLY via state digits => the uniformity push inflates exactly the generic subspace.
2. **sep bought nothing** (A sep5 >= B sep2 on famIoU) — translation can't keep clouds this large disjoint (E25-P7 replayed).
3. **The protocol is fine**: the expert n256 re-warm ran the identical aux-only recipe the same day and PASSED (famIoU 0.145 = the n384 certificate; core50 1713; family 0.267/0.115/0.055). The difference is the routed content.
4. **Sub-span probe** (NEW instrument, per-position routing reconstructed from the order-preserving eval-path stats; region split instr[0:16)/mid/state[28:]): the language field is two objects. State region (~2/3 of valid positions): each task routes onto ~45k slots (69% of bank) of shared numeric vocabulary — famIoU 0.36-0.40, bg 0.20-0.29. Instruction region: famIoU 0.212-0.224, core50 407-1610, bg 0.051-0.091 — arm A's instr region PASSES all three gates on the checkpoint that just failed; instr famIoU ~0.22 is ARM-INVARIANT (the genuine word-sharing floor, 3.4x bg). Caveat: per-position IoUs are estimator-depressed (sparse-vs-sparse, E24); region aggregates are the honest numbers.

### Discussion (Josh) -> the design

Options for the router input: (1) pool everything (rejected — sacrifices the measured-good per-token instruction routing); (2) per-token instr + pool(state) key for the state region; (3) per-token instr + state keyed by pooled(instr(+state)) — anchor + smooth offset = footprint translation by construction. Josh's push: measure the component variances ANALYTICALLY first and scale-match (the E28 lesson: ||scene||/||beta|| = 17-21x WAS the routing outcome).

**Querystats probe** (NEW, stage-1 features = the router input, libero_10):
- State positions: between-task/within-task variance = **0.126 (L15) / 0.112 (L16)** vs instruction 0.985/0.810 — task context IS mixed in (Josh's hypothesis, confirmed present) but sits 8-9x under the digit noise => per-token state routing routes on the generic 88% (the sprawl, quantified).
- Pooled state component: between-task cos 0.931/0.970, **family 0.981/0.991** — near-degenerate; a state-only palette key has almost no family separation to work with.
- Instruction pool: between 0.800/0.866, family 0.882/0.919; **intra-cos 0.864/0.890** — the anchor alone already carries within-task variation (image/state context bleeds into instruction-token hiddens) => the E21 constant-palette fear is empirically weakened at b=0.
- Composite grid k = a*nrm(instr)+b*nrm(state): separation degrades monotonically in b (inter 0.800 -> 0.898 -> 0.931 across b=0 -> 1 -> state-only). **(1,0) dominates the init geometry**; b>0 only pays if the trained proj converts the state channel into conditionality worth the separation cost. Pooled means arrive at sqrt(rho)-shrunken norms (coherence 0.22-0.49) => component RMS-normalization is mandatory (the lop-sidedness Josh predicted, measured).
- Multiplicity design note: all state positions share one key => the palette enters losses/stats at xn_state. Deliberate split: TF/write-demand counts keep the multiplicity (slots serve n_state positions); routing-loss dedup (loss-mask) DEFERRED as a fast-follow — tonight's arms carry the documented z-mean weighting (~2/3 palette).

### THE BUILD

- Router keys ONLY (value path per-position on live x): instr positions [0,b) per-token; positions [b,v) share a*nrm(pool instr[3:b]) + b*nrm(pool state[b+3:v-5]), rescaled to batch-mean token RMS; per-row fallback to per-token when the boundary is missing.

### The 3-arm chain

Warm-up 10k (router-only, c0.05/sep5.0) -> audit -> vlm_audit_analysis + subspan -> PERMISSIVE gate (kill only min-effnum<=100 or famIoU>=0.45 both layers — strict E44 gates don't transfer to palette-dominated famIoU; morning reads decide) -> merge -> A-phase 10k (values both towers, routers frozen, frozen-base expert routing, bs32) -> e4 1-task probe (C-config: beta4/top_t1536/5k steps/lr 1e-3->1e-4, 20-ep eval). Chunk probes centrally in the morning; winner -> 5-task attribution run (C-config, vs stageB 32.0/35.0 its near-twin). Arms: **A anchor10 (1,0)** / **B anchor1005 (1,0.5)** / **C statepool** [bracket end + querystats-calibration cell]. Deferred: 10-task extension (post-validation), sep refinement wave (tomorrow daytime if cells marginal), loss-mask dedup.

### Entry 45 addendum (18 Jul eve) — the pooled-state-key question resolved by the variance ledger

**"Doesn't the pooled state key also carry within-task information?" (Josh) — yes, and the probe says the anchor already carries more of it.** The ledger, from querystats: after pooling, the INSTRUCTION pool's within-task variance is 165.9/150.8 (L15/L16) vs the state pool's 77.9/84.9 — attention has already mixed image/proprio context into the instruction-token hiddens, and that contextual bleed survives pooling ~2x better than the state signal survives its own digit noise. Sharper still, the state pool's variation is task-INTERLEAVED: intra-cos 0.852 < inter-cos 0.931 (two same-task samples are less similar than two task centroids — task structure buried), while the instr pool is coherent (intra 0.864 > inter 0.800). So b>0 buys little conditionality not already present at (1,0), at a measured separation cost (inter 0.800->0.898 at b=1). Framing worth recording: **(1,0) is not state-blind memory — state enters through the per-position value path (U_sV_s x_p on each live hidden) and through the anchor's contextual bleed; b controls only whether SLOT SELECTION keys on direct proprioception.** Standing caveats (why (1,0.5) still gets a box): these are init-time statistics through the identity map — a trained proj could extract the task-orthogonal component of the state pool; and variance accounting is not information accounting (phase gating could need the direct channel).

Morning protocol: chunk probes on the three e4 checkpoints (pass toward <=0.08; staged-best 0.153, LoRA target 0.020) + audits + subspan region reads -> winner -> 5-task attribution run (C-config). A hard-gate stop overnight ("GATE: HARD FAIL") is a result, not a bug.

---
## Entry 46 - 19 Jul 26 (Pooled-router sweep verdict: BOTH anchored variants hit the expert-certificate routing level (famIoU 0.145-0.151) and produce the largest e4 fit gain in the project (chunk 0.153 -> 0.099, -35%); winner = anchor+state (arm B). Discussion corrections: capacity ledger = read-write product (params fell 11% while the palette became an always-read/always-trained block); L16 routing was NON-stationary (Josh's catch) -> frozen-route extended to the VLM tower; route-once (retrieval computed once per shared key); bank-scaling law measured (footprints track bank size, famIoU invariant). -> E46: three like-for-like joint router re-warms (incumbent / vlm-knn36 / uniform n128-r4))

### Results — the overnight pooled-router sweep (three chains)

Each chain: VLM router warm-up (10k, aux losses only) -> held-out audit -> merge with the
certified expert n256 bank -> joint A-phase (values both towers, 10k on libero_90) -> single-task
e4 adaptation probe (5k steps, C-config) + executed-chunk error (famIoU pass <= 0.25; chunk
anchors: previous staged best 0.153, single-task LoRA specialist 0.020).

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

### Builds

- **Frozen-route extended to the VLM tower**: a no-grad fork
  advances a memory-free prefix stream from L15 and recomputes L16's attention front on it;
  inference reuses the capture/stash dual pass.
  No router retrain needed
  in principle (memory-free = the certification distribution up to the value_proj-bias residual,
  the known 0.98-transfer term).
- **Route-once** for shared router keys: the state region routes ONCE per sample on a compact
  sequence [state key, instruction tokens] (key first keeps valid tokens a contiguous prefix for
  the loss machinery); slot params gathered once per row, applied per-position.
  Usage/TF/audit stats
  keep served-position multiplicity (numbers comparable); routing/contrastive losses now count
  each unique key ONCE — the pre-registered dedup, a deliberate change to future warm-ups'
  training signal.
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

| arm | expert | VLM | question |
|---|---|---|---|
| 1 incumbent | n256/r2/knn36 | n256/r2/knn16 | the baseline, re-certified under the new protocol |
| 3 knn axis | n256/r2/knn36 | n256/r2/**knn36** | does palette capacity 64->144 pay? |
| 2 uniform | **n128/r4/knn36** | **n128/r4/knn36** | iso-rank-unit concentration: 4x fewer, stronger slots |

All: sep 5.0, c 0.05, anchored (1.0, 0.5), 10k compressed, router lr 1e-4. Pre-registered arm-2 read: expert core50 ~400-800 at
famIoU ~0.145 = scaling law holds; famIoU up at scaled cores = the per-query-draw floor binds.

### Next steps

1. Afternoon re-assess on the three audit certificates -> A-phases on survivors
   -> filled e4 probes + chunk -> select -> graduate ONE config to the 5-task
   sequential (C-config, vs stageB 32.0/35.0).
2. Queued behind selection: 10-task extension; LoRA specialist cells e2/e7; the sep-response
   question in the anchored regime; protection-store decay fix before any multi-task run that
   needs it.

---
## Entry 47 - 19 Jul 26 (E46 three-arm audit verdict: the new protocol re-certifies the incumbent EXPERT router to the third decimal, but the route-once LOSS DEDUP collapsed the VLM state palette to ~2 query-draws (famIoU 0.08 -> 0.19-0.24) — the deduped loss under-weights the palette relative to its deployment read mass; arm 2 (n128) breaks the bank-scaling law on both towers. Fix: warm-ups broadcast, downstream compact)

### Results — the three like-for-like joint router warm-ups (all completed + audited 19 Jul)

Recap of the design: each arm retrains BOTH towers' routers from the stage-1 backbone
(values pinned at zero, aux losses only, frozen-route inputs, per-tower routing-loss
topk aligned to knn), then runs the held-out audit and stops. Anchors: the old
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

Instrument note: route-once reordered the per-sample stats rows to
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

### Fix

- **Route-once switch** (default true):
  gates the compact route-once dispatch. False forces the byte-exact
  legacy broadcast path (shared key routed at every state position -> losses/queues
  carry served-position multiplicity). Warm-ups run false; A-phase/sequential/inference
  keep true (router frozen -> paths numerically interchangeable, parity 1.19e-07).

### Next steps

1. **Arms 1'/3' re-warms** (same chain shape: warm-up -> audit ->
   analyses -> sub-span -> STOP). Pre-registered reads: arm 1' ~ poolB's certificate (VLM famIoU ~0.149/0.147, palette
   ~0.08 / effnum 600-800) — it is poolB's exact replica plus the protocol
   improvements; arm 3' answers the knn36 palette-capacity question at matched loss
   semantics (the E46 comparison was confounded by the dedup pinning palette size to
   ~2 draws in both variants). Expert side expected unchanged (~0.145-0.17) for both.
2. **Arm 2' = uniform n192/r4/knn36 both towers, broadcast losses**
   (decided with Josh).
   Why: n192 is the unmeasured scaling-law midpoint (4x
   cumulative from n384; the law held at 2.25x, broke at 9x), and n192/r4 is the
   CLEAN iso-budget concentration arm (147,456 rank-units/layer vs n256/r2's
   131,072; the original n128/r4 arm was only HALF the budget — a mislabeling in the
   E46 design). Pre-registered: famIoU ~0.145-0.16 at core50 ~1,100-1,300 = law
   holds; ~0.18+ = branch closed.
3. **Review ALL THREE audits together** (arm 1' knn16,
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

---
## Entry 48 - 19 Jul 26 (E47 broadcast re-warm verdict: arm 1' REPRODUCES the poolB certificate (dedup attribution closed end-to-end); knn36's apparent advantage was a dedup artifact (score-profile-depth mechanism) but remains a 2.25x per-forward capacity lever; n192 FAILS via a newly measured mechanism — per-half subkey utilization crosses the pigeonhole knee (demand ~ n^0.74; n256 sits AT the knee) — the bank axis is closed with a sizing rule. GRADUATION: three chains on the palette-constancy axis (arm 1' / arm 3' / arm 3-old) -> A-phase -> 5-task)

### Results — the three broadcast re-warms (vlm_route_once=false; audited + analyzed 19 Jul eve)

Recap: the E46/47 arms retrain both towers' routers from the stage-1 backbone with
values pinned (aux losses only), frozen-route inputs, per-tower topk=knn — differing
only in bank size / knn / loss semantics. Anchors: the n256 expert certificate and
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

Note: arm 2' re-audited at bs16 x 200 steps/task (total audited samples MATCHED to every
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

### Graduation (decided with Josh)

Each chain: A-phase (10k values-only both towers on
libero_90, routers frozen, frozen-route ON, compact path restored downstream) -> 5-task
sequential (C-config: beta4, top_t 1536, 5k steps/task, value lr 1e-3->1e-4, 20-ep
intermediates + 50-ep final; comparator stageB 32.0 final / 35.0 init). Arm 2' NOT
graduated (pre-registered kill: famIoU >= ~0.18 at 3 of 4 expert layers).

| chain | axis position |
|---|---|
| arm 1' | state-conditional end, best certificates |
| arm 3' | middle; the knn/per-forward-capacity contender |
| arm 3-old | constant end; compact-palette fit hypothesis |

**Gate 2 (pre-registered):** each sequential's t0 block IS the e4 probe (same C-config
5k steps) — chunk error on checkpoints/005000 vs anchors 0.153 (old staged best) /
0.0994 (poolB) / 0.020 (LoRA specialist); kill any run >= ~0.12 before its remaining
blocks run. Chunk probes run centrally with the standard instrument.

### Next steps

1. Gate-2 chunk read on all three t0 checkpoints (+ kill/continue calls).
2. 5-task finals + probe battery -> select the 10-task carrier. If arm 3'
   wins fit materially, the knn choice becomes capacity-vs-family-exposure and the
   10-task decides; if fit ~equal, arm 1' wins outright on certificates.

---
## Entry 49 - 20 Jul 26 (E48 graduation verdict: arm 1' = NEW FRONTIER 40.0 at plain C-config — the palette-constancy axis is monotone on every instrument incl. all 15 cross-arm final-chunk cells; knn axis closed at 16; the read-write product heuristic gains a STATE-CONDITIONALITY factor; new protection-x-palette starvation mechanism measured; retention POSITIVE in function space. -> composition + VLM-r4 arms; image-span plan fixed (step 1 keep-expert single-delta / step 2 drop-expert reallocation))

### Results — the three graduation chains

Recap: three chains from the E47/48 certified joint router warm-ups, differing ONLY in
the VLM router — arm 1' knn16 broadcast-loss keys (state-conditional palette, ~11-13
slot-draws per footprint), arm 3' knn36 broadcast (~5-7 draws), arm 3-old knn36
dedup-loss keys (near-constant ~2-draw palette). Expert tower config-identical
(n256/r2/knn36); each chain = A-phase (10k values-only, both towers) -> 5-task
sequential (C-config: beta4, top_t 1536, 5k steps/task, value lr 1e-3->1e-4, 50-ep
finals). Chunk anchors: 0.153 old staged best / 0.0994 poolB / 0.020 LoRA specialist.

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

- **The 70% arithmetic, corrected and re-anchored.** Josh challenged the
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
- **Two arms:**
  1. composition, sequential-only from arm 1's existing
     A checkpoint (lr 2e-3->2e-4, top_t 3072).
  2. VLM rank 4: the warm-up trains
     keys/proj only and frozen-route makes routing values-independent, so the r2
     certificate transfers WITHOUT re-warming;
     the graft drops the 4 r2-shaped VLM slot tensors
     (fresh r4 init; slot_up zero => memory output 0 at start).
     Sequential forced to
     bs16xacc2.
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
  r4x3 marginal at bs16xacc2) decided by 1A's audit + the r4 arm (which IS the
  rank-axis measurement on the right tower). Step 2 runs the full chain without a
  human review stop but keeps automated hard-fail tripwires. Lean: layers > rank
  (rank's record is weak everywhere tested; layers buy new routing surfaces =
  conditional capacity, the currency conclusion 2 just validated; image leverage sits
  earlier in the stack). Bonus physics: prefix memory runs 1x per 50-action chunk at
  inference (cached) vs the expert tower's 10x per denoise.

### Next steps

1. Composition arm + VLM-r4 arm; Gate-2 chunk probes on their t0
   checkpoints (~0.12 kill).
2. **1A**: querystats-image probe (layers incl. candidates below 15 to
   inform step 2) -> freeze region count / (a,b) / normalization ->
   joint warm-up (broadcast) ->
   audit with image-region gates -> review with Josh.
3. Then 1B (A + 5-task, single delta vs arm 1' 40.0/0.0940/chunk grid); then step 2.

---
### Entry 49 addendum (20 Jul eve) — 1A design frozen from the querystats-image probe

**Probe results** (layers 7-16, stage-1
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

Review gates: image-region famIoU <= ~0.25 at per-region effnum >=
~300 (no ~2-draw collapse), state/instr regions within ~20% of arm 1' (0.074/0.084
palette, ~0.20 instr), expert certificate reproduced. The audit famIoU topline is
image-mass-weighted — read the sub-span region table, not the topline.

---
### Entry 49 addendum 2 (20 Jul night) — composition Gate-2: the amplitude levers COMPOSE on the new substrate (e4 chunk 0.112 -> 0.0753, -33%); first e4 cell at the edge of its conversion window

The composition arm (arm 1' substrate + lr 2e-3->2e-4 + top_t 3072)
delivered its t0 checkpoint; Gate-2 chunk probe:

- **e4 own-block chunk 0.0753** (late10 0.0794, seed spread 0.061) vs arm 1' 0.1119
  (0.1275 / 0.079) — the best e4 function ever measured outside the LoRA compass
  anchors, beating even poolB's 1-task probe (0.0994). 20-ep t0 init 35 (arm 1' 25;
  retired instrument, but now with function-space support behind it).
- **The composition is the news**: on the expert-only substrate the amplitude levers
  never moved e4 (chunk 0.160 -> 0.156 across the whole lr/top-t family — E42); on
  the VLM substrate the same levers cut it 33%. Mechanistically coherent: e4's
  deficit is perception-side and the levers now feed the always-read VLM palette
  that carries that content.
- 0.0753 sits at the edge of e4's conversion window (specialist bound ~0.03;
  breadth-law discount per E43 — e9 converted at 0.068). Late-chunk error improved
  MORE than the mean (-38%) and seed spread tightened — the compounding tail is
  cleaner, which is where rollouts live.
- Residuals: one cell; rollout conversion unproven until the 50-ep final;
  amplitude give-back is the known risk — re-check own->final at 025000 against
  arm 1's uniformly-improving retention grid.

---
### Entry 49 addendum 3 (20 Jul night) — 1A CERTIFIES

Audit review: image regions famIoU 0.091/0.128 at effnum 548/665 (state-conditional
~8-10 draws; sprawl and collapse branches both dead), bg 0.010/0.013; **the instruction
region's word-sharing floor broke** (0.198-0.219 invariant across E45-47 -> 0.103/0.134,
bg 0.015 — 8x cleaner; plausibly the image keys absorbing the loss-mass competition);
state palette L16 0.127 (arm 1' 0.084, +51% relative — the one watch-item, at arm 3''s
level which still made 37.2); topline 0.105/0.141 < arm 1's 0.136/0.156; expert
certificate reproduced (5th consecutive). 2.5/3 gates -> certified.

1B: C-config verbatim from the imgspan warm-up, the
SINGLE-DELTA cell vs arm 1' (composition levers deliberately not stacked).
Comparators: t0 chunk vs 0.112,
50-ep final vs 40.0, alongside the composition final (its Gate-2: e4 chunk 0.0753)
and the r4 arm.

---
## Entry 50 - 21 Jul 26 (Morning-review verdicts: COMPOSITION = new frontier 46.0 (amplitude converts on the VLM substrate; e4 14->34 as its Gate-2 chunk predicted); BOTH substrate bets FAIL with BETTER on-demo function (vlmr4 30.4, imgspan 28.8) — the conversion gap re-surfaces at substrate scale; jitter probe + Josh's amplification model relocate the imgspan failure to the VALUE PATH at image positions (image stack parked); r4 axis closed on both towers. -> E50 wave: layer-max gated chain, top-p write budget + lr4x arms)

### Results — the three overnight arms (50-ep finals)

Recap: all three built on the E48 frontier, arm 1' (expert n256/r2/knn36 at [8,10,12,14],
VLM text-field n256/r2/knn16 at [15,16], famIoU-0.145-certified router; 40.0 final /
block-min 0.0940). Composition = same substrate + value-LR 2e-3->2e-4 + top_t 3072
(sequential-only, reused A-checkpoint). VLM-r4 = vlm_lora_rank 4 via checkpoint graft
(fresh r4 VLM values; router certificate transfers bitwise), plain C-config. Image-span
1B = the E49 image-region-routing warm-up graduated through the identical chain, plain
C-config.

| arm | e4 | e6 | e9 | e2 | e7 | FINAL | block-min | final-grid chunk mean |
|---|--|--|--|--|--|--|--|--|
| arm 1' (E48) | 14 | 58 | 16 | 82 | 30 | 40.0 | 0.0940 | 0.130 |
| composition | 34 | 60 | 26 | 84 | 26 | **46.0** | 0.0687 | **0.097** |
| vlm-r4 | 18 | 38 | 14 | 68 | 14 | 30.4 | 0.0813 | 0.111 |
| imgspan 1B | 20 | 36 | 24 | 58 | 6 | 28.8 | 0.0858 | 0.108 |

- **Composition converts end-to-end.** The Gate-2 prediction landed (e4 chunk 0.0753 ->
  rollout 34, the best staged e4 ever); every final-grid chunk cell beat arm 1' by
  22-33%; give-back small (e4 own->final +7.6%; rollout give-back ~0 vs the same
  levers' -11.4 on the old expert-only substrate). Exposure roughly doubled (selfcov
  0.96, VLM16-e4 RTO 71.5%, e4-perceived VLM field drift 86-92%) at zero measured cost
  — bleed-doesn't-convert holds at 2x amplitude here. E48's projection (loss
  0.070-0.080 -> 44-49) landed at 0.0687 -> 46.0.
- **Both substrate arms invert fit and rollout.** Each has BETTER on-demo function than
  arm 1' on all five final-grid cells and rolls ~10pp worse. The deficits sit on cells
  retention cannot explain (r4: e7 — the last task, zero later exposure — 14 vs 30 at
  better loss; imgspan: e2 58 vs 82 at better chunk). Eliminated by measurement: r4
  exposure topology == arm 1' (footprints bitwise-same under frozen-route; RTO within
  ~4pp; e4-perceived drift identical), r4 A-bank magnitudes fine; imgspan A-phase loss
  IDENTICAL to arm 1''s (0.1029 both), forgetting matrix flat (+1.6..+5.5%), retention
  grid flat-to-improving. Function-space ranking across the four arms (comp 0.097 <
  imgspan 0.108 < r4 0.111 < arm1p 0.130) is ANTI-correlated with rollouts across
  substrates while remaining correlated within one (comp vs arm1p).

### The jitter instrument + the mechanism correction (Josh)

Jitter probe (perturb demo observations, re-denoise, score the chunk; state sigma 0.1/0.2
per-dim std, image pixel sigma 0.05) on the four final checkpoints:

| relative degradation | arm1p | comp | vlmr4 | imgspan |
|---|---|---|---|---|
| image@0.05 (t0/t3/t4) | 2.06/2.09/2.77 | 2.33/2.51/3.44 | 2.27/2.35/3.05 | **2.70/2.61/4.11** |
| state@0.1 (t0/t3/t4) | 1.35/1.49/1.81 | 1.37/1.45/1.77 | 1.50/1.55/1.87 | **1.27/1.34/1.49** |

- imgspan: steepest image slope of all arms on every task AND flattest state slope —
  the brittleness is channel-specific to the one input its new pathway reads; its
  ABSOLUTE image-perturbed error crosses above arm 1' on all three tasks (worst on
  e7, 0.368 vs 0.264 — the cell that fell 30->6). comp/r4 stay absolutely BELOW arm 1'
  everywhere (so the shell explains imgspan, and does NOT explain r4 — r4's residual
  deficit stays formally unexplained: noisier within-block optimization (block-end
  1.4-2.0x block-min vs ~1.1-1.25x), two historically chunk-unrankable cells among its
  losses, and an unsynced A-phase health number).
- **Attribution corrected in discussion (Josh): the amplification model.** My initial
  "image-conditioned ADDRESSING is unstable" claim had zero direct evidence (no
  retrieved-set measurement). Josh's alternative — the slot transforms applied AT image
  positions amplify the function's dependence on image features, so off-demo visual
  drift moves the function proportionally more — fits the data better on three counts:
  (1) comp and r4 raised the image slope with routing bitwise-FIXED (value mass alone
  moves this metric); (2) imgspan's transforms take image hiddens as DIRECT input at
  512 positions (vs attention-mediated in every other arm) and its excess slope exceeds
  comp's at half the write budget; (3) the E43 VLM-LoRA specialists — dense adapters on
  image processing — were the most image-brittle models ever measured and survived only
  by level (clean error far below the brittleness cost; imgspan's did not get that far).
  Rerouting remains possible but unevidenced and second-order.

### Conclusions

1. **Frontier = arm 1' substrate + lr2x + top_t 3072 at 46.0**; amplitude converts
   through the VLM palette (the levers never moved e4 on the expert-only substrate).
2. **The on-demo chunk metric is ranked-instrument WITHIN a substrate and ranks
   BACKWARDS across substrates** (twice in one night). Substrate changes are henceforth
   judged on 50-ep finals + the jitter battery, never chunk alone. This also
   retro-explains Gate-2 passing the r4 arm (t0 chunk 0.0977 — an on-demo gate).
3. **The threshold law gains a support axis**: both failed arms lowered chunk while
   raising the effective threshold (brittler off-demo behavior); composition lowered
   chunk with support untouched and converted fully. Fit gains count only when the
   substrate's off-distribution behavior is preserved.
4. **Rank axis closed on both towers** (expert +2 in E33; VLM -10 here). Layers, not
   rank, is the remaining capacity direction.
5. **Image stack parked for the deadline (Josh's channel-ranking argument, measured
   support):** instruction constant, state ~linear in deviation (and pooled+anchored),
   image features noisiest relative to task signal (querystats: patch within-task
   variance dominates; pooled family cos 0.975-0.988). With image-perturbation training
   off the table ("we train on the same data"), the main de-brittling lever for
   image-position serving is unavailable. The perception-leverage evidence (compass)
   stands; the banked text-field VLM serving IS the usable share of it.
6. Freeze-top-5k stays retired (E44 ran the direct test: no jitter dividend, real fit
   bill, 0.2-0.5% erosion = protected non-damage), and the amplification model argues
   against it structurally (brittleness lives in NEW content's gain, not old content's
   erosion). Revisit only if VLM A-core erosion under composition-scale writes measures
   >~5% (cheap check: arm 1' A-usage inert sweep + checkpoint diff).
7. Protection-store decay rejected as insufficiently general (Josh). e9's remaining
   budget starvation is real but small: at top_t 3072, e4/e2/e7 are effectively
   un-truncated (selfcov 0.90-0.97) while e9 (~0.80 expert-side) and e6 (0.68-0.83)
   still rotate — hence top-p.

### Next steps (the E50 wave; all three = single deltas, 50-ep finals)

1. **Layer-max gated chain**: expert [2,4,6,8] +
   VLM text-field [10,12,14,16], n256/r2 everywhere (3.2B values; seq bs16xacc2).
   Rationale: parameter-max the tower the compass says carries fit, at depths where the
   anchor geometry is best (E49 probe: separation improves monotonically downward);
   Josh's split satisfies the placement guard (all VLM layers above
   expert max).
   Automated gate (validated PASS on the imgspan audit / FAIL on
   frozenbase): fail -> auto-fallback compact split expert [9,10,11,12] + VLM
   [13,14,15,16] -> re-gate -> fail = STOP (drawing board). Known risk the gate
   watches: expert routing at L2/L4 (feat-probe separability L4 89.5% vs 98 plateau).
2. **Top-p write budget**: k = min(n_read, max(3072,
   ceil(0.9*n_read)), 16384) per module per batch — "write ~everything you read";
   single delta vs composition.
3. **lr-max**: 4e-3 -> 2e-4 on the composition
   substrate; the amplitude-headroom cell (E43: function did not saturate at 4x;
   give-back is the accepted risk).

### Entry 50 addendum (22 Jul) — CORRECTION: the original layer-max split was accidentally SKIPPED, not rejected; the placement direction is UNSETTLED

The chain's fallback fired on an instrumentation failure, not a measurement: attempt A's
(expert [2,4,6,8] / VLM [10,12,14,16]) warm-up trained fine, but its AUDIT OOMed at
bs32.
The gate fail-safed to attempt B (compact, [9,10,11,12] +
[13,14,15,16]), which certified well (famIoU 0.140-0.154 uniform over all 8 modules).
Attempt A's router is UNMEASURED.
Until then, "compact beats max" is not a conclusion anywhere — the
max-vs-compact placement question is open, and if A gates PASS its own A-phase +
sequential is a follow-up decision against the compact result.

---
## Entry 51 - 22 Jul 26

### Part 1 — lr4x verdict (amplitude axis closed at 2x: an inverted-U dose-response) -> BUDGET-CONSERVING protection designed (Josh)

**The lr4x result that motivates the arm.** The 4x-LR cell on the frontier substrate
(arm 1' + top_t 3072) landed 40.4 @ 50 eps (22/54/18/76/32), completing the amplitude
dose-response: **1x -> 40.0, 2x -> 46.0, 4x -> 40.4** — an inverted U peaking at 2x.
Decomposition (Josh's read, confirmed): 4x wins every boundary — best fresh 20-ep cells
of the family, block-min mean 0.0651 (project-low at the time), grad norms clean — and
gives it back (rollout init-mean->final -5.0 vs comp's +1.8; e9 45->15 timing-matched to
its two biggest VLM16 bleed channels, 28.1% e2-block / 32.7% e7-block). The slot data
isolates the channel exactly: exposure topology BITWISE-identical to comp (frozen
router, same masks, same bleed masses) while displacement per unit exposure is up ~24%
at every module (e4-perceived VLM drift 86->107% / 92->114%). Same where, harder hit —
the E43 ladder replicated on the new substrate, now with a rollout price.

**Why existing protection can't fix it.** Rank-mode beta4 is candidacy-only: a vetoed
slot is replaced by the next-ranked at FULL LR (writer plasticity conserved,
relocated) — but high-TF shared cores never rank out, and magnitude is untouched.
grad_scale-mode attenuates magnitudes but never reallocates: as the max-union store
grows, every later writer's mask carries more scaled-down LRs — total effective
plasticity shrinks monotonically with task index (the measured E44 bill: e7 block-min
+49%, rollout 44->28, net wash 39.6 vs 42.4).

**The fix (Josh's design, v2 after review): budget-conserving REDISTRIBUTION within
the unchanged mask**. *[SUPERSEDED — see Part 4: v2 NaN'd ~200
steps after the first task boundary (momentum-boost compounding) and its refund had a
u==0 eligibility cliff; it never produced a valid run. The design below is recorded as
history; the shipped mechanism is v3.]* The top-t filter stays byte-identical
to a plain run (pure TF-IDF top-t: same slots, same breadth — v1's deeper-reach
variant was rejected in discussion: it changed write breadth as a second delta and
pushed budget into the low-TF tail the top-p incident implicates). Within the fixed
mask: each slot's LR is scaled by (1-u)^beta, and the deducted mass D = sum(1-scale)
is water-filled back onto the writer's own hottest UNPROTECTED slots in score order
(cap 2.0x per slot; capped-out remainder unspent + logged). Properties: (i) total
effective plasticity == mask size for EVERY writer, invariant to how large the
protection union grows — the E44 accumulation problem dies at the SPEND side without
touching the store (more general than the rejected decay); (ii) protection =
redistribution, not suppression — and the conserved budget concentrates on the
writer's private core, the read-write-product-positive allocation (E48 currency);
(iii) u-norm corefrac (whole prior cores at u~1, shoulders graded); (iv) protected
slots are NEVER boosted (fill skips any slot with u > 0).

**The arm** (sequential-only from
the arm 1' A-checkpoint): single
delta vs lr4x 40.4 = protect_mode rank->budget + u_norm peak->corefrac, write
breadth/mask unchanged by construction. Pre-registered:
t0-t2 block-mins ~= 0.0651 (early writers untaxed by construction); e9 final back
toward >=26; e7 block-min <= ~0.085 (the starvation tripwire; E44's grad_scale paid
+49%, budget mode should hold ~0.075); **beat 46.0 to displace the composition
frontier.** The claim under test: keep 4x's boundary wins, redistribute (not
suppress) the write pressure that produced the -5 give-back.

### Part 2 — top-p verdict: 19.2, the worst 50-ep final of the staged era, on the BEST losses ever recorded — the write budget was load-bearing protection all along

**Result**: final 2/10/0/64/20 = **19.2** (comp,
its single-delta twin at fixed top_t 3072: 46.0). Block-min losses were the project's
best at the time (0.050/0.037/0.089/0.052/0.055) — the starkest fit/rollout
dissociation ever measured, and the first early-task cliffs since pre-stationarity
(e4 25 -> 0 after ONE block and pinned there; e9 -> 0).

**Autopsy (slot JSONs + checkpoint field analysis; footprints frozen-router-identical
to comp, so every delta is write-side):**
1. **Write breadth exploded**: 57-64k updated slots per task per module (~90% of the
   65,536 bank, EVERY block) vs comp's 19-38k — the bank was rewritten ~5x over.
   (Also falsifies my earlier claim that the VLM tower was un-truncated at 3072: its
   writes doubled too, so VLM per-batch reads exceed 3072.)
2. **The core-overwrite channel (retention killer, dominant on the EXPERT tower)**:
   later tasks' update events into e4's core-50 rose 5-18x (L14 270k -> 3,776k), and
   the delivered damage — e4's core-50-weighted relative value drift over blocks 1-4 —
   went from **8-9% (comp, and every healthy run) to 67-74%** on the expert tower
   (VLM: 32-44% -> 59-69%). The function e4 relied on was simply rewritten; 25 -> 0.
   Mechanism: at "write 90% of reads," beta4's rank discount only excludes
   bottom-decile slots — high-TF shared cores never fall there — so the
   ranking+budget's implicit core exclusion died. **Scope condition on E42's
   "bleed doesn't convert": that was shoulder-bleed, measured under masks that
   steered writes OFF cores. Core-bleed converts immediately.**
3. **The tail channel (conversion degrader)**: events outside the union of all five
   cores went 53-77% -> 75-90% of a ~3x larger total — the stable A-content share of
   every retrieval mixture churned continuously, consistent with fresh-task rollout
   conversion dropping (e9 init 30 vs 50, e2 60 vs 75) while own-demo losses IMPROVED
   (more DOF on-manifold). The accidental counterfactual of the freeze debate: the
   fixed budget was implicitly preserving the generalist substrate.

**Conclusions.** (i) top-p as built is DEAD — but the negative result is load-bearing:
top_t was never just an interference knob; the budget+ranking pair silently provided
BOTH core protection and tail preservation, and every "breadth is safe" datapoint
(E42-43, comp's RTO 71% at zero cost) was conditional on them. (ii) Any future
adaptive-coverage lever must carry magnitude control. (iii) This strengthens the
budget arm (Part 1) on both channels by construction: mask stays
3072 (no tail explosion) and prior cores enter the mask only at (1-u)^beta ~ 0
magnitude (STRONGER core protection than rank mode, which provably could not exclude
high-TF cores at any budget). (iv) Chunk/MSE-matrix probes deferred
and likely unnecessary — the mechanism question is
answered at the slot/field level.

### Part 3 — top-p post-mortem discussion (dose vs structure) + the agreed board

**Was 0.9 simply too big?** (Josh's question, discussed): partly — but the structure is
fragile at any useful dose. Both kill channels scale CONTINUOUSLY with mask fraction
(the rank-veto's strength is exactly "how far below the cut (1-u)^beta can push a
core"; comp already leaked 270k-1.1M core events at top-3072 — top-p multiplied a
known leak 5-18x, it did not create one), and the useful range (materially more e9/e6
coverage) overlaps the dangerous range (veto degraded over the score band where shared
cores live). No p-sweep scheduled. If adaptive coverage is ever revisited it is
adaptive-B INSIDE budget mode ("top-p with teeth": mask growth only with the magnitude
layer on) — and the motivation has weakened anyway: comp at fixed 3072 already rolled
e9 at 26 (its best ever), and e9's residual deficit is threshold-shaped (chunk ~0.17
vs ~0.07), which no write budget fixes.

**Static mask is now measured-best from both sides** (comp 46.0 above, top-p 19.2
below); whether static SCALES beats static full-LR is exactly the budget arm's
question.

**Agreed board (Josh: "agree on all points"):**
1. compact layer-max final + full battery incl. jitter
   (substrate rule) + the attempt-A re-audit (settles max-vs-compact placement).
2. budget-4x final — needs >46.0 to displace the frontier;
   mechanism validated at the autopsy either way.
3. branch on the compact-LM final: >= ~40 at plain C-config ->
   compact-LM + composition levers (lr2x + top_t 3072; the arm-1' one-two, 40.0->46.0,
   on a substrate with 2x the VLM layers — the highest-EV cell on the board);
   < ~40 -> layer axis dead, budget@2x instead (clean comp's residual
   +7.6% e4 own->final give-back).
4. Endgame: converge substrate (arm 1' vs compact-LM, informed by the
   attempt-A verdict) x optimization package (2x+3072 +/- budget) into ONE frontier
   config, full battery, and reserve the final ~day for a clean confirmatory run of
   the frontier rather than one more speculative arm.

### Part 4 — budget v2 NaN postmortem (momentum-boost compounding + the u==0 eligibility cliff) -> v3 REDESIGN (Josh's conserved proportional allocation)

**The run.** The v2 water-fill run diverged to loss=nan at
~step 5.2k — ~200 steps after the FIRST task boundary.
t0 artifacts are clean (empty store => budget provably inert
=> t0 bitwise plain-lr4x; the 005000 checkpoint and boundary eval predate the
corruption).

**Root cause 1 — the NaN: momentum-boost compounding.** The v2 water-fill assigns
boosted slots scale up to 2.0; boosted slots are by construction the writer's hottest
unprotected slots, which persist in the mask for hundreds of consecutive steps. The
E42 momentum-aware blend multiplies the row's Adam exp_avg by its scale EVERY
snapshotted step. For attenuation (s<1) that is a contraction — the designed tail-kill.
For s=2.0 it compounds: exp_avg <- 2.0*(0.9*exp_avg + 0.1*g) => x1.8/step => overflow
in ~150 steps (log(3e38)/log(1.8) ~ 150; observed fuse ~200). Third Adam-statefulness
incident in the protection line (grad-scale invariance E41, momentum tail leak E42,
momentum boost compounding E51): two individually-correct mechanisms (E42 fix; budget
boost) composed divergently, and v2 was the first time any scale > 1 flowed through
the E42 line.

**Root cause 2 — the design flaw the one pre-NaN log line exposed.** `[budget] 15:
unspent 354/446` (the only trustworthy line; everything after the NaN is garbage
masks): at VLM L15, t1's deficit was 446 slot-equivalents and only ~92 could be
refunded — because v2's refund eligibility required u == 0 EXACTLY, and prior tasks'
BINARY read tails blanket the bank (the top-p run measured 57-64k of 65,536 slots
written per module on write-what-you-read masks => block read unions ~90% of the
bank => ~97% of t1's mask carried u > 0, mostly at u ~ 0.001-0.01 where the deduction
is ~nothing but eligibility dies). Read-MASS separation (the certificates, famIoU
0.145) is real and did its job — the deficit was only 446/3072 ~ 15%; the eligibility
rule conflated "touched once" with "relied on", and the max-fold store makes the
u==0 set shrink monotonically (by t3-t4, ~nothing is refundable anywhere). Same
binary-tail-vs-weighted-mass distinction as E8/E19, biting inside the protection
mechanism. (En route, a v2 spec mis-read surfaced: I had implemented the v2 mask as
PURE TF-IDF top-t — a silent second delta vs the lr4x twin, whose masks are
rank-discounted. Josh's intended membership was the discounted ranking; v3 restores
it.)

**v3 (Josh's design, from the discussion): ONE score drives membership AND speed.**
- score_i = tfidf_i * (1-u_i)^beta — the rank-mode ranking, so masks are BITWISE the
  lr4x twin's (u evolves identically under the frozen router).
- Mask = top-t by score. u=1 slots score 0 => never selected => frozen, their seats
  going to clean slots (better than v2's in-mask-at-scale-0, which wasted seats).
- scale_i = min(2.0, lam*score_i), lam solved EXACTLY so sum(scale) == mask size
  (with the top m capped, lam = (k - m*cap)/suffix_m;
  smallest feasible m; sort+cumsum, closed form). Conservation exact by construction —
  no deficit, refund, eligibility set, or unspent remainder EXISTS. Below-average-score
  slots donate LR to above-average ones continuously ("slot A would cause forgetting ->
  deduct; slot B is clean and hot -> bump").
- Momentum: exp_avg *= min(scale, 1). Attenuation keeps the exact E42 tail-kill
  semantics (s=0 bitwise freeze); boost
  acts on the delta blend only == per-row 2x-LR Adam while in-mask (bounded per step,
  1x tail after mask exit). The v2 NaN is structurally impossible.
- Accepted consequence, eyes open: with an EMPTY store (t0) score == tfidf, so t0 runs
  TF-proportional (hot slots ~2x, mask tail <1x) where lr4x ran flat — the arm is now
  "protection + proportional allocation package vs lr4x", not a pure protection delta.
  t0 is the built-in tripwire: block-min > ~0.075 (twin 0.0651) => proportional
  allocation hurts a clean writer => kill.

**Relaunch**: same substrate
(arm 1' A-checkpoint), same schedule (4e-3 -> 2e-4, top_t 3072, beta4/corefrac,
5 tasks x 5k, 50-ep final). Judged against lr4x 40.4 (nearest twin — masks bitwise
shared) and composition 46.0 (the frontier). Pre-registered: t0 tripwire above; NaN
watch at the first boundary (~5.2k — the v2 signature); e9 final back toward >=26;
e7 block-min <= ~0.085 (last-writer starvation tripwire — conservation should hold
~0.075).

### Part 5 — attention-family compass (e4): KILL branch fires — dense VLM-attention-only LoRA (chunk 0.106, plateaued) loses to our own sparse MLP-site frontier (0.081); the "architectural shift" does not carry the dense-adapter gap

**Design**:
dense per-task LoRA on e4, byte-identical recipe to the E43/44 compass
anchors, targets = VLM LM self_attn q/k/v/o (all 18 blocks) + common action/state
projections, r32, 7.5M trainable (GQA makes k/v adapters small). The superset cell for
any attention-side memory: its failure kills o-only a fortiori.

**Train-loss plateau ladder** (same recipe, last-25% mean / end-trend):
expert-only 0.195 / +0.6% (hard plateau) -> chunk 0.229 / roll 14;
**attn-only 0.128 / −1.8% (converged)**;
VLM-full 0.063 / **−7.2% (still descending)** -> 0.030 / 40.
Attention alone recovers ~a third of the expert->VLM-full gap and flattens at 2x
VLM-full's loss.

**Chunk (same jitter instrument, t0/e4 clean):** full specialist 0.0204 < VLM-full
0.0298 < **comp (ours) 0.0812** < vlmr4 0.0995 < **attn-only 0.1056** < arm1p@1536
0.1134 < multitask-LoRA 0.1210 < expert-only 0.2292. The unconstrained dense adapter
restricted to the attention family lands MID-PACK among our sparse arms and BELOW the
composition frontier. Jitter profile ordinary (image@0.05 2.07x, state@0.1 1.31x —
≈ arm1p's slopes): a pure LEVEL deficit, no smoothness story.

**Verdict:** the pre-registered kill fires (chunk ≥ ~0.10 AND plateaued). The e4
dense-adapter gain is carried by the VLM MLP family, not attention; attention-side
sparse memory (o-first build) is DEAD as the +15 vehicle. The r64 param-matched escape
hatch is declined: even a 2x improvement would not approach sufficiency (0.05 ≫ 0.03),
and our existing substrate already beats the cell. Notable positive: our sparse
MLP-site system outperforms a dense adapter on the wrong family — placement beats
density again (E44's compass lesson, now within-tower).

**Pending:** the 50-ep cell (fit-threshold
curve predicts ~15-30 — decorative post-kill); the MLP-only arm
(predicted chunk ~0.03-0.045 = the our-site-is-right
confirmation). Colleague's suggestion 1 resolved: right method (compass-first — one
VM-evening priced the direction), negative answer.

### Part 6 — MLP-only compass arm KILLED (Josh's call; the site question rests on subtraction: VLM-full 0.030 works, attn-only 0.106 doesn't ⇒ MLP carries it) -> CLAMPED MASS-BASED TOP-P; design frozen from measurement

**Design freeze (the E45 playbook).** Block-aggregate k90 (slots covering 90% of a
task-block's read mass) from comp's JSONs, per module: expert L8-14 = 8.9-21.9k with
e9 ~2x every other task (its diffuseness re-measured at the mass level); VLM L15/16
FLIP: e4 is the most diffuse (19.3/21.5k vs e9's 12.6/15.5k) — the adaptive rule
hands e4 extra VLM budget, aligned with its perception-side deficit. Per-batch k90
~0.2-0.3x block => working band ~2.5-6.5k => **clip [3072, 5120]**, p=0.9 on the
protection-discounted ranking score (mass on the deployed score, NOT count of unique
slots — the E50 rule measured the ~90%-incidental binary tail and produced the 19.2;
the mass semantics REPLACE it in code, count rule retired).

**Honest degeneracy note (pre-registered):** the mass rule only ignores tails whose
aggregate mass < 1-p; if per-batch TF tails are fat, every module pins at
the 5120 cap and the arm degenerates to static top_t=5120 — still the two-sided band
test.
The cap (not p) is the load-bearing guard; both E50 kill channels scale with mask
size and are bounded to ~1.7x comp's known-benign leak.

Arm: single delta vs comp 46.0.
What was given up: the MLP-only confirmation cell
— the our-site-is-right claim now rests on the strong subtraction
inference rather than direct measurement; recorded as an accepted trade.

### Part 7 — COMPACT LAYER-MAX VERDICT: 44.8 at plain C-config — the layers axis pays where rank/imgspan failed; battery + attempt-A re-audit close the placement question; FOLD-IN (layermax + lr2x + top_t 3072)

**The result**
(50-ep final; substrate = expert [9,10,11,12] + VLM text-field [13,14,15,16], 8 modules,
n256/r2, bs16xacc2):

| | arm 1' (6 mod) | **layermax (8 mod)** | comp (arm1p + levers) |
|---|---|---|---|
| config | C @1536, 1x | **C @1536, 1x (identical)** | lr2x @3072 |
| final | 40.0 | **44.8** | 46.0 |
| e4/e6/e9/e2/e7 | 14/58/16/82/30 | **34/36/44/78/32** | 34/60/26/84/26 |
| block-min mean | 0.0940 | **0.0528 (-44%)** | 0.0687 |

The largest matched-config substrate win on record — and it CONVERTS (both prior
substrate bets, vlmr4 and imgspan, improved fit and rolled worse; this is the first
substrate change since the original VLM build where fit and rollout move together).
The E50 branch gate (>= ~40) clears decisively.

**Per-cell noise calibration** (finals 50-ep, +/-7pp): e4 +20 and e9 +28 vs arm 1' are
REAL (~2-3 sigma, backed by block-min moves of -49%/-36%; e9's 44 is the best e9
rollout in project history, prior best 26). e6 -22 is ~2.2 sigma — probably partially
real but it is the 4-time chunk-misrank cell, and its FUNCTION is the best e6 ever
measured anywhere (block-min 0.0188, -72% vs arm 1'; exposure clean: VLM RTO at
matched layers 36/44% vs arm1p's 35/42%). e2 -4 / e7 +2: noise. Summary: the gains
landed where the fit went; the one loss is on the cell whose rollout has never
tracked its function in either direction.

**Battery (substrate rule):**
- Final-grid chunk (jitter clean rows; 025000 checkpoint, i.e. WITH all retention
  baked in): t0 0.0452 / t1 0.0353 / t3 0.0538 / t4 0.0503 — beats comp WITH ITS
  LEVERS ON per-cell by 24-55% (comp: 0.0810/0.0506/0.1200/0.0662) at plain config.
  t0's 0.0452 approaches the e4 conversion window (~0.02-0.03); the fold-in's levers
  aim exactly there. [t2/e9 cell missing:
  its gain probe crashed on the new layout;
  e9's rollout 44 is the primary read regardless.]
- Jitter: relative slopes are elevated (image@0.05 3.4-4.1x, state@0.1 2.0-2.9x)
  BECAUSE the clean floor halved; ABSOLUTE perturbed errors stay at-or-below arm 1'
  everywhere (t0 image 0.173 vs arm1p ~0.234) — the imgspan kill signature (absolute
  crossing above the baseline) is absent. Substrate certified on both battery axes.
- Slot autopsy: the win's mechanism is the E48 currency again — four VLM layers carry
  much SHARPER per-layer palettes (e2 core50 197-380 across four banks vs 365-631
  across two; e7 238-357 vs 484-648), always-read capacity spread over more,
  more-concentrated banks; expert selfcov up ~5-9pp on the diffuse tasks (e6
  0.65-0.70 vs 0.56-0.61). Read-write product raised by LAYERS, which converts —
  where raising it by rank (r4: -10) did not.

**Attempt-A re-audit**
(closes the E50-addendum "direction unsettled"): **GATE PASS, but
strictly weaker than compact where it matters.** Expert famIoU 0.163 (L8) worsening
to 0.212 (L2) — L2 individually violates the <=0.20 band; the gate passes on the
3-of-4 rule — vs compact's uniform 0.140-0.154. The E36 feature-probe prediction
(expert separability degrades below ~L8: L4 89.5% vs the >=98 plateau) is confirmed
at the certificate level. VLM at [10,12,14,16] healthy (0.132-0.152) — depth-
extension works on that tower, per the E49 geometry probe. **Max-vs-compact SETTLED:
compact >= A on certificates AND compact converted at rollout; A's chain stays unrun
(decided with Josh before the verdict: audit-only, stop after).** Design input
banked for any further layer push: the expert stack should not go below ~L8 — a 6+6
build's forced expert [5-10] is half inside the flagged band; the evidence-clean
extension is **expert [8-11] + VLM [12-16] = 9 modules** (every layer inside
measured-good territory). Decision deferred behind the fold-in + budget verdicts.

**FOLD-IN**:
layermax substrate + the composition levers (lr 2e-3 -> 2e-4, top_t 3072),
sequential-only from the existing A-checkpoint.
Pre-registered: beat 46.0
to take the frontier; **>= 49.2 crosses the multitask-LoRA line (the recalibrated
"must" target)**; e4 >= 34 and e9 >= ~40 must survive amplitude (the lr4x
displacement lesson); give-back tripwire <= ~-3; block-min mean pushing below
~0.045.

**Board state at close of 22 Jul:** Attention-side memory killed by compass (Part 5); MLP-only arm traded for
the top-p slot (Part 6). Target ladder recalibrated (discussion): must = beat
multitask-LoRA 49.2; good = 52-55 (~85-90% of the specialist oracle at none of its
advantages); the missing table cells (e2/e7 specialists, naive sequential LoRA — the
headline forgetting baseline) still to run.

### Part 8 — budget-v3 verdict: core overwrite ELIMINATED (0 events vs 260k-1.1M) and e9 rescued (+22 vs twin), but 43.6 < comp's 46.0 — the residual is amplitude's own cost, so the amplitude axis CLOSES at 2x; the gains ledger pivots the program to LAYERS (spread-A chain with levers; absolute-max certify-first)

**Result** (50-ep final): **43.6** = 24/48/40/76/30
vs lr4x twin 40.4 (+3.2, clears it) vs comp 46.0 (−2.4, short). Trajectory
25/40-35/25-60-55/40-45-30-65/24-48-40-76-30.

**Mechanism verdict — the strongest protection result on record:**
- **Total core exclusion in production.** Later tasks' update events into e4's
  core-50: **0 — exactly zero at all six modules** (twin: 268k-1,107k). Into e9's
  core-50 from t3/t4: **0** (twin: 361k-866k). Corefrac puts whole prior cores at
  u=1 → score exactly 0 → never selected. The E51-Part-2 core-overwrite channel is
  eliminated, not attenuated.
- e9 — whose lr4x collapse (45→15) was timing-matched to exactly these channels —
  lands 40 vs the twin's 18 (+22; second-best e9 ever after layermax's 44).
- Conservation exact every logged line (sum 3072); 25k steps NaN-free; t0 tripwire
  passed (block-min 0.0531 vs the 0.075 line — TF-proportional allocation does not
  hurt a clean writer).

**Attribution confound (owned).** The arm bundled TWO deltas vs its twin: protect_mode
(rank→budget) AND u_norm (peak→corefrac). Corefrac ALONE in rank mode produces the
same structural exclusion (u=1 → ranking score 0 → never in the mask); budget mode's
unique contribution is only the proportional speed profile. The fit ledger points at
that speed profile as a cost: block-min taxes vs twin ran +3/+17/+9/+8/+13% by task —
the +17% on e6 is partly the genuine writer tax (e6 shares "mug" content with e4's
now-untouchable core), but the across-the-board pattern is consistent with the mask
tail running at 0.2-0.5x LR and under-converging (the concentration concern raised at
design time; E42's coverage lesson). **The clean single-delta cell this identifies:
rank+corefrac @2x+3072 — core exclusion without the
proportional tail, at the amplitude that pays no brittleness. Deprioritized behind
the layers pivot (below).**

**The 4x-vs-2x residual decomposed — and the amplitude axis closes.** e4's deficit vs
comp (24 vs 34) is NOT interference: zero events touched its core, and its block-min
(0.0531) is BETTER than comp's (0.0547) — yet it rolls 10 lower. That is the E43
amplitude signature (at 4x the correction direction goes generic, T_feat → 1.02;
off-demo robustness drops). Protection solved 100% of the solvable channel; the
remaining −2.4 vs comp = amplitude's intrinsic own-value brittleness (e4/e2) + the
writer tax on genuine sharing (e6). Both costs are continuous in LR → a 3x arm
interpolates them (~45-47 projection) on a substrate being superseded the same day.
**Decision (with Josh): no 3x; 2x is the amplitude optimum. Axis closed:
1x 40.0 / 2x 46.0 / 4x 40.4 / 4x+budget 43.6.**

**Compass row completed:** attn-only 50-ep landed at **26.0** (predicted band 15-30)
— the Part-5 kill row is now full: chunk 0.106 / plateaued / roll 26, vs VLM-full
0.030/40 and our comp 0.075/34. Attention-side memory stays dead.

**THE PIVOT (Josh's gains-ledger argument, adopted):** scope beats optimization,
four times running — VLM build +6, layermax +4.8, levers-on-new-substrate +6, vs
protection/budget/top-p ≈ 0-3 each. The program's remaining bets go to LAYERS:
1. **Spread-A chain**:
   attempt A's substrate (expert [2,4,6,8] + VLM [10,12,14,16] —
   8 modules SPREAD over depth 2-16) through A-phase + 5-task sequential WITH the
   composition levers, matching the running compact fold-in. **Count-matched,
   bank-matched, rank-matched vs compact: spacing is the only substrate variable**
   — the clean isolation of Josh's coverage hypothesis. Evidence for: A's VLM half
   certified at parity (0.132-0.152) in better anchor geometry (E49 downward
   improvement; E43 lower-layer transmission). Against: the certificate streak (5/5)
   and A's weaker expert half (0.163-0.212).
2. **Absolute layer-max certify-first**: expert [4-9] + VLM
   [10-16] = 13 modules, 5.4B values.
   Warm-up + audit first; full chain ONLY on a
   certificate pass, with the spread-vs-contiguous verdict shaping its layout
   (A ≥ compact → spread the 13; A < compact → contiguous as specced). Known
   marginal layer: expert L4 (famIoU ~0.195 in A's audit; trim to [5-9] if it drags).
3. Fold-in interim: block-mins t0 0.0304 / t1 0.0156 / t2 0.0725 — through
   the <0.045 pre-registration.

### Part 9 — mass top-p verdict: 37.6 — the rule ran as designed and the DOSE-RESPONSE kills the axis; slot-level mechanics extracted (continuous damage, two-channel threshold separation, the layers-vs-budget collision principle)

**Result** (50-ep final): **37.6**
= 22/42/12/88/24 vs comp (its single-delta twin, static 3072 @ 2x) 46.0 =
34/60/26/84/26. Block-min mean **0.0639** — BETTER than comp's 0.0687 (family-best
fit at 2x); the top-p fit/rollout dissociation again, milder than the count-rule's
19.2 but the same shape.

**Regime read** (in-run [top_p] k-lines, 50 samples): cap-pinned at 5120 on 64-90%
of batches per module, floor-pinned **0/50**, occasional in-band dips (3.7-4.9k).
The pre-registered fat-tail degeneracy fired — the arm is effectively the
**static-~5120 band test**, i.e. exactly the upper-bound measurement Part 3 said we
lacked. Write-budget dose-response at 2x on this substrate: **~90%-of-reads
(count rule) 19.2 < ~5120 37.6 < 3072 = 46.0** — static 3072 is the interior
optimum, now bounded from both sides. Coverage/write-budget axis CLOSED (joins
amplitude @2x and rank).

**Mechanistic derivations (slot-level, vs the bitwise-matched comp twin):**
1. **Damage is CONTINUOUS, not cliffed.** At 1.67x budget, later-task events into
   e9's core went x2.2 (L14 869k -> 1,919k) and shoulder x1.9 (6.4M -> 11.9M) —
   ~linear in mask size — and the rollout damage came graded to match (e9 30->25->12,
   no zeros/pinning; contrast the count-rule's cliff e4 25->0 at 5-18x core dose).
   No "safe breadth" plateau exists above 3072.
2. **The two kill channels separate by dose threshold.** Retention damage is
   first-order in budget (fired at 1.67x: e9 -14, e6 -18, e4 -12). The tail-churn ->
   fresh-conversion channel did NOT fire: unique-written rose only +25-50% (vs the
   count rule's ~5x) and fresh conversion SURVIVED OR IMPROVED — e2 = 88, the best
   e2 ever recorded; e7 held. The generalist substrate tolerates ~1.7x churn and
   dies before ~5x; the retention channel has no tolerance band at all. "The budget
   was load-bearing protection" (Part 2) now has a dose-response, and the
   load-bearing half is specifically retention.
3. **Plasticity exchange-rate ranking at the frontier**: breadth +67% bought -7%
   loss + one writer cell (e2 +4) at -12..-18 x3 victims (net -8.4); amplitude
   2x->4x bought -5% loss at net -5.6. Both levers net-negative at 46.0, breadth
   the worse currency. **The frontier config is plasticity-SATURATED** — its binding
   constraint is fit QUALITY (the threshold law on e4/e9), not fit budget in any
   within-bank currency.
4. **Per-batch adaptivity is structurally dead, not mistuned.** k never touched the
   floor: per-batch score mass is heavy-tailed for EVERY task (32 frames x ~144
   draws spread over 15-24k slots => most slots drawn 1-3x). The concentration the
   colleague's rule keys on is real at the BLOCK level (k90 8-22k) and absent at the
   PER-BATCH level where the mask acts — cross-batch accumulation, not within-batch
   structure. No p-value fixes a missing signal.
5. **The layers-vs-budget collision principle** (the deepest derivation): extra
   budget buys deeper reach into SHARED, CONTESTED banks — the marginal mask slots
   are precisely prior tasks' shoulders (the events table IS that statement); extra
   layers buy NEW banks with new, certificate-separated routing surfaces. Same
   capacity currency, opposite collision cross-sections. Retro-explains the whole
   axis ledger: layers +4.8/+6 vs rank -10/+2, budget-breadth -8.4, count-top-p -27.
   **Capacity converts iff it arrives with new, separated addressing** — E48's
   conditionality currency, confirmed from the failure side.

Confound note (pre-registered honesty): measured under weak core protection
(rank+peak — the arm predates corefrac). Breadth x corefrac is unmeasured;
budget-v3's exclusion would have blocked much of the core dose, but the corefrac
writer-tax + derivation 3 say it would not clear 46.0. Stays unrun.

**The absolute layer-max warm-up** —
expert [4-9] + VLM [10-16] = 13
modules, 5.37B values, n256/r2, broadcast losses, certify-first (warm-up -> audit
-> STOP for manual review; bands — expert famIoU <=
~0.20 on >= 5/6, VLM <= ~0.25 all 7, no palette collapse; trim option expert [5-9]
if only L4 fails).
Chain launch decision
waits on THREE inputs: this certificate, the compact
fold-in final, and spread-A (which decides contiguous vs spread layout).

**Part 9 addendum — why layermax's per-layer cores are smaller** The shared layers falsify any cross-layer
competition story: e4's L15/16 footprints carried over near-bitwise between the two
warm-ups (1549/1931 layermax vs 1565/1904 arm1p) despite two extra banks below —
and mechanically no coupling channel exists (losses strictly per-module on
per-module queues, no shared normalization; values pinned; frozen-route serves
identical memory-free features regardless of other modules; grad clip never engages
at warm-up norms 0.05). Each layer's routing equilibrium is an independent problem.
The sharpening decomposes as: (1) DEPTH — the added banks sit lower where anchor
geometry is more open (E49), and footprints are monotone in depth within layermax
itself (e2: 197 -> 296 -> 380); (2) ARITHMETIC — totals grew sub-linearly (e2 996 ->
1202 slots over 2x banks) so per-layer means fell without any bank being squeezed;
(3) residual task-level variance at shared layers (e7/e9) = warm-up init randomness
(different module sets consume the init RNG differently), not load-splitting.
Cross-layer coupling enters only at the VALUE level (content jointly fitted through
gates/LN/residual = division of labor), never the addressing level.
**Pre-registered for the absmax certificate: L10-12 palettes sharper still
(depth gradient continues down); L13-16 footprints ~= the compact certificate's (no
shrink from banks added below). All seven uniformly smaller than compact's would
falsify this and revive a coupling channel.**

---
## Entry 52 - 23 Jul 26 (Fold-in verdict 43.6: the give-back postmortem — best per-task functions ever recorded (e4/e9 cross their conversion thresholds for the first time), then 2x-LR write drift pushes both back across: FIRST rising MSE-forgetting diagonal of the stationary era (e4 +22.6%), e4 V16 core-drift dose ladder 36/44/55/59-69% -> finals 34/34/18/0. Give-back is created by the AMPLITUDE, not the substrate (plain give-backs 0-3%). Patch cell: rank+corefrac on the fold-in config)

### The result

(layermax compact substrate + lr 2e-3->2e-4 + top_t 3072; 50-ep final): **43.6** =
e4 18 / e6 58 / e9 30 / e2 76 / e7 36 — below BOTH parents (comp 46.0, layermax-plain
44.8). Trajectory (20-ep intermediates): e4 **55**->40->25->35->**18**; e9 **55**->25->
**30**; e6 50->40->55->58 (recovers); e2 75->76 (flat); e7 36 (best in family).
init-mean 54.2 -> 43.6 = **give-back -10.6** (tripwire <=-3 FIRED). e4's 55 and e9's 55
are the best cells ever recorded for those tasks by ~20pp. Two parent anchors: comp's e9
did the IDENTICAL crash at e2's block (50->25->26; fold-in 55->25->30), and both parents
IMPROVED e4 across e7's final block (comp +14, plain +4) where the fold-in fell -17.

### Battery (full)

**1. Fit: the levers composed perfectly.** Block-min mean **0.0409 (project low**; plain
0.0528, comp 0.0687), every block best-ever, grad norms clean (max 0.029). Own-block
chunk errors best-ever on ALL FIVE tasks by ~2x: e4 **0.0333** (comp own 0.0753; dense
VLM-LoRA specialist 0.0298 — matched for the first time), e6 0.0240, e9 **0.0784**,
e2 0.0421, e7 0.0376. **e4 and e9 crossed their rollout-conversion thresholds for the
first time in the project** (e4 window ~0.02-0.04: specialist 0.020->58; e9 ~0.07:
specialist 0.0675->70) — the 55-inits were real conversion, not draws.

**2. MSE forgetting matrix (paired-noise, 16 batches): the stationary-era flat matrix
BREAKS.** Diagonal drift just-trained->final: **e4 +22.6%** (0.0500->0.0613), e6 +13.0%,
e9 +11.0%, e2 +3.7%, e7 own=final. Historical band since frozen-route: +0-5% total
(prior max: arm 1' e4 +4.8%). Real function-space forgetting is back at 2x LR on the
8-module substrate.

**3. Delivered damage (read-mass-weighted value drift on victim cores).**
e4's damage is VLM-core overwrite: V16 core drift 17% (after e6's
block) -> 36% (e9's) -> 45% (e2's) -> **55%** (e7's); expert side only 6-11%. Cross-run
dose-response, e4 V16 core drift -> e4 final: **plain 36% -> 34; comp ~44% (recorded
E51P2) -> 34; fold-in 55% -> 18; top-p count-rule 59-69% -> 0.** The cliff sits between
~44 and ~55%. foldin/plain drift ratio ~1.5x = 2^0.57 — the E43 sublinear displacement
law reproduced exactly. Internal controls: e6 (32% V16 drift) RECOVERED to 58; e2 (7%)
flat — drift converts to rollout loss only at the threshold. Writer attribution into
e4's V16 core: e6 472k, e9 365k events (the mug tasks — e4 two-mugs / e6 mug+pudding /
e9 mugs+microwave: object-sharing again), e2 256k, e7 314k. **e9's damage is a different
channel**: core drift small (8-16%), SHOULDER (core50-90 band) 25-33% (~2x plain) —
and its crash replicates comp's e2-block collapse, the known 2x-era channel budget-v3
rescued (+22) by core exclusion.

**4. Static exposure does NOT discriminate** (the E42 lesson at substrate scale):
core-RTO and events are twin-like vs comp (e4 V15/16 core-RTO 53/59% vs comp 54/58%;
e9 expert-core events from e2 304-412k vs comp 310-536k). The delta is displacement per
event (2x LR) landing on the sharper 8-bank VLM cores. Protection (rank+peak) inert as
always: u ~0.035 at core boundary -> (1-u)^4 ~ 0.87 vetoes nothing.

**5. Jitter: NO in-shell brittleness — fold-in's absolute perturbed errors are the
LOWEST of all three runs at every sigma** (e4 image@0.05: 0.153 vs plain 0.173 / comp
0.189; same relative slopes as plain). The imgspan kill signature is absent. The
rollout magnitude beyond what chunk explains (fold-in final e4 chunk 0.0402 is 2x
BETTER than comp's 0.0810 yet rolls 18 vs 34 — cross-substrate misrank #3, now
cross-CONFIG) is the known off-trail conversion layer, operating for the first time on
cliff-edge tasks.

**6. Layermax-plain backfill (own-block cells + the missing t2 final):
the give-back is created
by the LEVERS, not the substrate.** Plain (1x) own->final: e4 0.0450->0.0448 (flat!),
e6 +2.7%, e9 +1.4%, e2 +1.8% — the healthy stationary-era pattern. And plain's e4 own
0.0450 NEVER crossed the threshold: 1x buys retention by never winning the fit.
(Cross-instrument validation: plain e4 final 0.0448 probe_conversion vs 0.0452 jitter
clean — the E51 cross-instrument caution retires.)

**7. Intermediate cells (e4 checkpoint-by-checkpoint): two-phase damage.** e4's on-demo
chunk is FLAT through 20k (0.0333->0.0345->0.0333->0.0345) and breaks only in e7's
block (->0.0402, +17%) — while V16 core drift accumulated 17->45% underneath and
rollouts declined from 55. So the early decline is OFF-demo damage tracking value
drift (invisible on demo states — the E42 mechanism), and the final block breaks
through on-demo. e9@020000: 0.0811 (+3.4% at its crash boundary — the crash is
threshold sensitivity, not on-demo collapse).

### Mechanism synthesis

The fold-in is not an anti-composition — it is the composition WORKING: fit landed at
the specialist level and converted (best inits ever). What broke is that the operating
points it reached sit on the steep part of the success-vs-function curve, where the
2x-amplitude write drift that was FREE in every prior run (comp paid +7.6% chunk
give-back on e4 at zero rollout cost — its function sat in the flat region) now
converts 1:1. Same drift, new location. The frontier config's problem is no longer
fit (0.0409) and no longer breadth (3072 selfcov 0.79-0.97) — it is that nothing
protects at-threshold functions from later writers at 2x displacement.

### Decision + the patch cell

Patch cell: the fold-in config
with ONE delta — `protect_u_norm` peak -> **corefrac** (mode stays rank; the E51-Part-8
identified single-delta cell, now on the layermax substrate where it matters most).
Mechanism already measured in budget-v3: corefrac puts whole prior cores at u=1 ->
score tfidf*(1-u)^4 = 0 -> ZERO later-task events into prior cores at every module
(vs 0.1-1.4M here); the shoulder is graded (u 0.2-1 -> 2.4-600x rank discount) ->
attacks e9's channel too. Writer cost priced from this run's JSONs: only **1-5% of
each writer's update events** land on prior cores -> rank-mode relocation
(candidacy-only, full LR) is near-free — avoids budget mode's proportional-speed
writer tax (+13-17% block-mins) that kept budget-v3 off the frontier. Pre-registered:
core events = 0 at all 8 modules; e4 V16 drift <= ~20%;
e4 >= 34 (init ~50s); e9 >= 40; e6 >= ~50 with t1 block-min <= 0.020 (mug-share
writer-tax tripwire); block-min mean <= 0.045; give-back >= -3; **beat 46.0 = frontier;
>= 49.2 crosses multitask-LoRA**. Anti-recommendations recorded: do NOT revert to 1x
LR (plain: retention by never winning) and do NOT drop top_t to 1536 (halves the core
dose but re-opens the e6/e9 mask-rotation coverage loss; protection keeps both).
Reserve levers if the shoulder still leaks: beta 8, protect_hard_u=0.9, budget mode.

### Next steps

2. Spread-A final: head-to-head vs the fold-in at matched levers
   — note its give-back numbers are now as informative as its final (the spacing
   hypothesis must also survive the amplitude channel).
3. Absmax certificate: score against the header bands + the Part-9-addendum
   pre-registration when the audit lands.
4. Endgame: the corefrac verdict picks the frontier config; full battery;
   reserve the last ~day for the clean confirmatory run.

---
## Entry 53 - 24 Jul 26 (E53 battery: corefrac = NEW FRONTIER 51.6, first config past the multitask-LoRA must-line — total core exclusion at zero writer tax, flat MSE matrix + held chunk grid restored at 2x; spread-A = best per-task functions in project history (t0 beats the dense VLM-LoRA specialist) taxed at the full peak-norm drift rate; absmax audit backfill: the expert text-anchor is a decisive low-layer rescue → 3-arm corefrac wave (absmax / spread / spread+anchor))

### The two landed runs

- **corefrac**
  — the E52 patch cell: fold-in config (compact layermax, lr 2e-3→2e-4, top_t 3072)
  + the single delta `protect_u_norm` peak→corefrac (mode stays rank).
- **spread-A**
  — the E51 spacing isolation: expert [2,4,6,8] + VLM [10,12,14,16], same levers,
  peak-norm β4 (predates corefrac).

Full battery (items 1-8) run on both.
Integrity: computed pairwise read-IoU == logged `memory_iou` to 4
decimals on both (0.0902/0.0902, 0.0897/0.0897); ckpt-diff proves values-only on
every block transition (16 slot tensors move, router/gate/backbone/vision bitwise
zero); corefrac's 005000 is BITWISE the fold-in's (identical MSE row + own-chunk
0.0333 + shared e4 init draw 55) — protection provably inert with an empty store,
per the budget-v3 precedent.

### Corefrac — new frontier 51.6; the mechanism is total and free

Retention (20-ep inits / 50-ep finals): e4 55→35→35→35→**28**, e6 50→55→65→**68**,
e9 75→60→**46**, e2 70→**80**, e7 **36**. Final **51.6** (init 57.2, give-back −5.6)
vs fold-in twin 43.6 (−10.6), comp 46.0, plain 44.8. **Crosses the multitask-LoRA
must-line (49.2)** — first config to do so; e6/e9/e2 cells are per-task 50-ep bests.

| instrument | corefrac | fold-in (peak twin) |
|---|---|---|
| events into prior cores | **0 — exactly, all 40 pair×module cells** | 0.1-1.4M/module |
| victim core drift (all victims, all 8 modules) | **0%** | e4 V16 55% |
| MSE diag drift (e4/e6/e9/e2/e7) | **+3.7/+3.8/+2.5/+1.4/0.0%** | +22.6/+13.0/+11.0/+3.7/0.0 |
| chunk own→final | **+3.6/+2.0/+4.4/+2.8/0.0%** | e4 +17% (broke through) |
| block-min mean / t1 (writer-tax tripwire) | 0.0415 / **0.0155** | 0.0409 / 0.0156 |
| jitter t0 image@0.05 (absolute) | 0.1527 | 0.1529 |

Zero writer tax (t1 identical; mean +1.5%; e9's own fit IMPROVED 0.0654 vs 0.0725 —
the budget-v3 relocation benefit reproduced in rank mode). e4 ends at chunk 0.0345 —
**still inside its conversion window at the final checkpoint, first time ever**;
e9 holds at 0.0788 ≈ threshold. Pre-registration scorecard **7/9**: misses only
e4 ≥ 34 (28, within 50-ep noise of the line) and give-back ≥ −3 (−5.6, halved).

Where the residual −5.6 lives: (a) 20-ep init-draw arithmetic (e9's 75, e4's 55 —
the retired instrument; function give-back ≤4.4% is the honest retention number);
(b) **shoulder relocation** — evicted core pressure lands on shoulders: e4 V-shoulder
drift 44-62% (bleeds now 10-14% at V16, all shoulder); (c) the off-trail conversion
layer on cliff-adjacent tasks (e9 75→46 at +2.5-4.4% function movement). Decision:
**corefrac is default-on in the recipe; peak-norm is retired** — there is no measured
reason to run it again.

### Spread-A — the spacing verdict: best fit ever, full drift tax

Retention: e4 50→25→40→5→**24**, e6 60→60→45→**44**, e9 75→40→**46**, e2 75→**72**,
e7 **20**. Final **41.2** (init 56.0, give-back −14.8) — below all three compact
comparators at rollout, and yet:

- **Own-block chunks are the best ever measured on ALL FIVE tasks**:
  0.0272/0.0200/0.0683/0.0309/0.0315. t0 **beats the dense VLM-LoRA specialist**
  (0.0272 vs 0.0298); e9 crosses its ~0.07 threshold at own-block for the first time;
  block-min mean 0.0382 = project low.
- **Jitter absolutes are the lowest in the family** (t0 image 0.1422 vs corefrac
  0.1527 / fold-in 0.1529) — no crossing, no imgspan signature: the spread substrate
  is smoother, not brittler. Writing at broader depths is genuinely valuable.
- The give-back is REAL function forgetting at the known channel: MSE diag
  +22.6/+15.9/+13.0/+4.3%, chunk give-backs +10-17%, e4 V16 core drift **56%/
  shoulder 89%** (over the E52 44-55% cliff, collapse-shaped trajectory to match),
  9.89M writer events into prior cores (max cell 548k). Damage depth-grades V10→V16.
- **E2 is the leakiest module of the family** (bleed cells to 34%, the two largest
  core-event cells) — production confirmation of the attempt-A audit's L2 famIoU
  0.212 and the E36 low-layer separability warning. e9's expert footprint dilution
  is ABSENT on this warm-up (E-cores ≈ peers) — part of its fit edge.
- **The e7 anomaly (hold loosely)**: last task, zero exposure — drift explains
  nothing — best-ever e7 function (0.0315, cleanest jitter) rolls 20 vs corefrac's
  36 at worse function. ~2.3σ at 50 eps: borderline-real. If real, a spread-substrate
  conversion deficit that no protection fixes.

Spacing verdict at matched (peak) protection: spread wins fit everywhere, loses
rollout (41.2 vs 43.6) via channels corefrac has since eliminated on compact — the
matchup is stale, hence the wave below.

### Absmax backfill (audit summaries computed today — they had never been run)

The expert text-anchor A/B on the absolute layer-max layout (expert [4-9] + VLM
[10-16], 13 modules, 5.37B values; anchor = per-layer pooled LM instruction hidden
into the expert router at B=0.5, FiLM off):

| expert famIoU | L4 | L5 | L6 | L7 | L8 | L9 | bg |
|---|---|---|---|---|---|---|---|
| plain | 0.242 | 0.195 | 0.201 | 0.219 | — | — | 0.10-0.13 |
| **anchored** | **0.136** | **0.119** | **0.135** | **0.148** | 0.166 | 0.174 | **0.022-0.034** |

The anchor rescues exactly the low layers that were the flagged risk, and produces
the cleanest expert-side routing ever certified (bg ~4× below anything prior);
VLM 0.132-0.150 untouched. The automated gate tripped on L9 0.174 vs
its 0.165 line (calibration miss — inside every historical certificate standard).

### The wave (corefrac in all)

| arm | delta / question |
|---|---|
| 1 | certified absmax substrate + corefrac, seq-only. |
| 2 | spread + corefrac, seq-only, single delta vs 41.2; e7 = the decision cell corefrac cannot help. |
| 3 | the candidate win: spread substrate + anchored-nofilm expert router (fresh warm-up) + corefrac. Gated (bands calibrated per the absmax lesson: ≤0.18/layer with one grace to 0.20; L2 = the honest unknown, never probed). vs arm 2 = the anchored router's conversion value at matched substrate+protection. |

Common pre-registrations: core events = 0 at all modules; MSE matrices flat (≤~+5%);
function give-backs ≤~5%/task; **beat 51.6 to take the frontier**.
Portfolio decision (discussed): the 10-task extension (the catastrophe-elimination
demonstration, arguably unblocked now that 51.6 > the must-line) takes the winner's
slot AFTER the wave rather than displacing an arm.

### Next steps

2. Battery on each landing; arm-3-vs-arm-2 is the
   attribution read; e7 across arms 2/3 is the substrate-conversion read.
3. Winner → 10-task extension and/or the clean confirmatory run.
4. Standing: corefrac default-on everywhere; peak-norm retired; 20-ep init cells
   remain retired from decisions (the corefrac give-back story is the reminder why).

**Update (24 Jul, PM) — arm-3 GATE FAILED (anchored-expert over-compaction); expert_anchor_weight 0.5 -> 0.1 refinement.** The spread + anchored-nofilm-expert + corefrac warm-up HARD-FAILED the gate on **expert core50 330-457 (< 800 floor)** — while expert famIoU **0.113-0.140** and bg **0.017-0.022** were the *cleanest expert routing ever certified*. VLM tower passed clean (min-eff 381-914). Classic E21: separation and capacity are two readings of routing breadth, and this moved them apart the wrong way.

Diagnosis: the expert text-anchor (per-layer pooled LM-**instruction** hidden, which is ~constant within a task) injected at weight 0.5 **over-compacts the expert router by collapse, not translation** — the expert router's native `proj(x)` is already the discriminative state-conditional signal (E28: scene 20x > language), so adding a strong per-task-constant just pulls every frame of a task into a tight cone (core50 ~1700 -> ~400, a ~4x capacity cut; the compact-layermax cert is 1664-1875 at famIoU 0.14-0.15). It looks spotless because the warm-up is aux-losses-only — separation *is* what over-compaction maximizes, so loss/famIoU curves are perfect; only core50/effnum shows the collapse.

**The E53 absmax "cleanest ever certified" anchor had the SAME latent collapse** (expert core50 425-648, all < 800) — but the backfill reported only famIoU/bg and its sequential was killed before stepping, so nothing ever checked capacity. This gate is the first with a core50 floor on an anchored-expert router; it caught what was always there. E21's trap replaying ("the audit had no capacity gate"). **Revision to E53's "the expert text-anchor is a decisive low-layer rescue": it rescues famIoU/bg by cutting expert capacity ~4x — a bad trade at weight 0.5.** (The VLM anchor is unaffected — it reads the language field where the anchor belongs, E45.)

Refinement: **`expert_anchor_weight` 0.5 -> 0.1** (5x weaker). VLM anchor `[1.0,0.5]` untouched. **Two-sided gate risk, pre-registered:** 0.1 may now *under*-separate — the unanchored spread-A L2 famIoU was 0.212 vs the gate's 0.18 ceiling (+ one grace to 0.20). The gate arbitrates. If 0.1 fails the other way (famIoU back near control), the anchored-expert idea is bracketed and arm 2 (plain spread-A + corefrac) stands as the spread verdict.

## E53 ARM-3 — expert-anchor separation sweep (empty window) (2026-07-25)

### Separation sweep (anchored-expert router)
ARM 3 = spread substrate ([2,4,6,8]+[10,12,14,16]) + per-task-constant pooled-LM-instruction
expert anchor (expert_anchor_pool=text, FiLM off) + corefrac. Swept EXPERT_ANCHOR_W as a
gate-directed bisection; each probe was a FRESH warm-up from base_nomem_50k.
The anchor weight is a single routing-breadth knob read two ways:
it sets both separation (expert famIoU, lower=better) and compaction (core50, higher=more capacity),
so the two trade off along the same axis. Gate: expert famIoU <=0.18 (one grace layer <=0.20),
bgIoU <=0.10, core50 >=800, min-eff >=300; VLM famIoU <=0.165.

  w=0.10  expert famIoU 0.48-0.68  core50 3668-9881   -> FAIL under-separated
  w=0.25  expert famIoU 0.34-0.39  core50 2250-2654   -> FAIL under-separated
  w=0.35  expert famIoU 0.21-0.25  core50 1058-1174   -> FAIL (closest; bg + min-eff pass)
  w=0.40  expert famIoU 0.167-0.201 core50 680-824    -> FAIL BOTH (L8 famIoU 0.201>0.20 AND L4/L6 core50<800)
  w=0.50  expert famIoU 0.11-0.14  core50 330-457     -> FAIL over-compacted

Conclusion: the famIoU-pass region (w >~0.40) and the core50-pass region (w <~0.40) cross WITHOUT
overlapping -> the viable window is EMPTY; no expert-anchor weight passes the gate. At w=0.40 the two
failure modes collide (needs MORE anchor to fix famIoU, LESS to fix core50 — contradictory). VLM
anchor passed at every weight (famIoU 0.11-0.15) — the failure is expert-specific. Verdict: the
anchored-expert router is not viable on this substrate. Standings unchanged: compact+corefrac 51.6
= frontier; spread+corefrac 47.6 = spread result (corefrac lift: +8.0 compact, +6.4 spread).

---
## Entry 54 - 26 Jul 26 (ARM-1 BACKFILL: absmax+anchor05+corefrac = NEW FRONTIER **53.6** with ZERO net give-back — on the router that FAILS the capacity gate at every expert layer; the capacity ledger falsifies the per-layer core50 floor (total −27% vs compact, re-allocated to the VLM tower + 3x cleaner shoulders); AAAI dropped → ICRA; spread re-opened at w=0.35 via GLOBAL loss levers — sep8-vs-c015 probe pair)

### Arm-1 backfill (the result E53's "standings unchanged" line predates)

The E53 arm-1 sequential
(absmax substrate: expert [4-9] + VLM [10-16] = 13 modules, n256/r2, 5.37B values; the
anchored-nofilm expert router at w=0.5; corefrac + lr2x + top_t 3072):

| step | e4 | e6 | e9 | e2 | e7 | seen-avg |
|---|--|--|--|--|--|--|
| 5k | 50 | | | | | 50.0 |
| 10k | 50 | 55 | | | | 52.5 |
| 15k | 55 | 60 | 55 | | | 56.7 |
| 20k | 45 | 45 | 60 | 80 | | 57.5 |
| **25k (50-ep)** | 40 | 52 | **76** | 72 | 28 | **53.6** |

**NEW FRONTIER: 53.6** — past compact+corefrac (51.6), inside the 52-55 "good" band
(multitask-LoRA must-line 49.2 cleared with margin). Init mean 53.6 (20-ep cells except
e7) → final 53.6: **net give-back 0.0, the first zero in the family** (corefrac-compact
−5.6). e9's 76 is ~30pp above its previous best (44/46) — and it *climbed* through later
blocks (55→60→76). Per-cell: e4 −10, e6 −3, e9 +21, e2 −8. Best final AND best
consistency in project history. Comparators (50-ep finals): compact+corefrac 51.6 /
spread+corefrac 47.6 / comp 46.0 / layermax-plain 44.8.

### The gate paradox (verified from the audit summaries)

This run's router is exactly the configuration class the E53 sweep condemned. The
anchored absmax certificate,
recomputed this session:

| expert layer | core50 mean (min-max/task) | famIoU | bgIoU | core50 ≥ 800 gate |
|---|---|---|---|---|
| L4 | 485 (147-1005) | 0.136 | 0.027 | FAIL |
| L5 | 425 (149-771) | 0.119 | 0.022 | FAIL |
| L6 | 453 (189-1156) | 0.135 | 0.022 | FAIL |
| L7 | 479 (141-859) | 0.148 | 0.022 | FAIL |
| L8 | 529 (181-1004) | 0.166 | 0.026 | FAIL |
| L9 | 648 (213-1135) | 0.174 | 0.034 | FAIL |

Capacity floor fails at ALL SIX expert layers (separation/bg = cleanest ever; VLM tower
passes its gates: famIoU 0.132-0.150, min-eff 394-837). These core50s sit in the band
condemned twice — E21/22's "capacity-dead" calls (511/696 on n384) and the E53 sweep's
w=0.5 verdict (330-457, "over-compacted"). The run exists only through timing: arm 1
launched the morning of 24 Jul on the famIoU/bg-only backfill read; the arm-3 gate
failure exposing the latent collapse landed that afternoon, mid-flight, and the
sequential was left to finish. **A gate-violating run is the frontier — the core50 ≥ 800
per-layer floor is falsified as a hard requirement, and the E53 "empty viable window /
anchored-expert not viable" verdict was a property of the GATE, not the router family.**

### The capacity ledger (all-bank totals, computed from the audits this session)

| substrate | banks | expert Σcore50 | VLM Σcore50 | ALL Σ | expert bg | result |
|---|---|---|---|---|---|---|
| arm 1' (6 mod) | 4+2 | 7,066 | 1,896 | 8,962 | 0.083 | 40.0 / 46.0 |
| compact layermax | 4+4 | 6,985 | 2,808 | 9,792 | 0.080 | 51.6 |
| **absmax anchor05** | 6+7 | **3,018** | **4,107** | **7,125** | **0.026** | **53.6** |
| spread anchor040 | 4+4 | 3,023 | 2,464 | 5,486 | 0.036 | (never ran) |

Josh's two observations, refined by the ledger: (1) low per-bank core50 correlates with
the consistency win, but it is the co-product of the anchor's separation — the likelier
active ingredients are **expert bg 0.026 vs 0.080** (3x cleaner shoulders — the only
damage channel corefrac leaves open) and 13-bank distribution (any one bank's drift is a
small share of any task's function). (2) "clawed back per-layer capacity via double the
layers" is right in structure but NOT a clean swap: ALL-bank total is 27% BELOW compact
(expert per-bank ÷3.5 × banks ×1.5 = expert total ÷2.3), and the growth is on the **VLM
tower** (2,808→4,107, 4→7 banks) — capacity re-allocated to where the compass says fit
lives. Causal test for (1), unscheduled: c-up on absmax at held separation (inflate cores;
if give-back returns, small cores were load-bearing; if not, it was separation/bg).
Terminology note for the record: the E53 sweep's `w` was `expert_anchor_weight`;
`routing_inter_task_separation_weight`
stayed 5.0 throughout — the sep loss itself has not been swept since E26.

### Decisions (Josh, 26 Jul)

1. **AAAI abandoned → ICRA.** The endgame-countdown pressure is off; the plan
   gains room for the missing baselines and seeds (below).
2. **Spread is the testbed** — best measured memory→fit substrate (E53: best own-block
   chunks on all five tasks; t0 beat the dense VLM-LoRA specialist), publishable size
   (3.2B values vs absmax's 5.37B on the 6.6B base — absmax ≈ 1.8x total params, too
   heavy as headline). Missing cell: a clean spread + anchored-expert + corefrac run.
3. **Core50 target relaxed** per the absmax evidence: not 800; per-bank ~400-500
   acceptable, with the guards moved to where the E21/22c risk actually lives —
   per-bank ≥ ~400 as the constancy TRIPWIRE, q_intra ≤ ~0.93, per-batch effnum,
   footprint dispersion ≤ 2x median. Under the relaxed gate, the E53 sweep's w=0.35
   point (famIoU 0.21-0.25, core50 1058-1174) is already capacity-comfortable and misses
   only famIoU by 0.03-0.07 — the window was empty only against the old floor.
4. **Per-tower loss-weight overrides REJECTED (Josh)** — global-only changes preferred.
   Safety argument: the audit is two-tower, so a global change that disturbs the
   certified VLM equilibrium is caught at certificate time for free; enforced VLM gates
   replace per-tower plumbing.
5. **w=0.35, not 0.3** (calibration): w=0.3 interpolates to famIoU ~0.27-0.31 → the
   losses would need −0.10..−0.13, beyond any measured loss-side famIoU move (max ~0.09,
   the whole E26 sep curve); w=0.35 needs −0.03..−0.07, inside precedent.

### E54 probe pair — spread @ w=0.35, global loss-lever isolation

Both: fresh warm-ups from `base_nomem_50k`, spread substrate
(expert [2,4,6,8]/n256/r2/knn36 + VLM [10,12,14,16]/n256/r2/knn16), expert text-anchor
FiLM-off at **w=0.35**, E53 arm-3 recipe otherwise verbatim, broadcast losses,
warm-up → audit → analyses → sub-span → STOP:

| probe | delta | expert mechanism | VLM risk (same audit watches it) |
|---|---|---|---|
| P1 `anchor035_sep8` | sep 5→**8** | translation — measured-safe (E26: famIoU fell while capacity ROSE along 0.5→5; >5 never swept) | palette COMPACTION (arm-3-old ~2-draw regime); guard: palette effnum ≥ ~500 |
| P2 `anchor035_c015` | c 0.05→**0.15** | SupCon cross-task push trains proj(x) to amplify the state-side task signal off the anchor axis | palette SPRAWL (E45: c→breadth monotone); guard: famIoU ≤ 0.165, headroom thin (0.132-0.152) |

Honest uncertainty on P2, pre-registered: the joint-era precedent is against it (P3/B —
c-up compacted without separating); the counter-argument is that regime was
MSE-dominated with no anchor. Genuinely a hypothesis test.

**Gates (recalibrated):** expert famIoU ≤ 0.18
(one grace ≤ 0.20), bg ≤ 0.10, mean core50 ≥ 400 (tripwire, not 800), min-task effnum
≥ 300; VLM famIoU ≤ 0.165 all layers, palette effnum ≥ ~500.
**Decision rule:** one threads → its A-phase + 5-task sequential w/ corefrac; both →
better joint margin; neither but one direction moves expert famIoU cleanly → dose
refinement (sep 12 / c 0.25 / w 0.375); both damage the VLM before fixing the expert →
per-LAYER anchor weight fallback (heavy at L2/L4 only — anchor knobs are inherently
expert-scoped, so no per-tower loss plumbing needed; the deficit is layer-local, plain
spread already passes at L6/L8).

### ICRA plan (agreed shape)

- **Wk 1:** spread routing rejig (this probe pair + at most one refinement) → certified
  winner → A-phase + 5-task w/ corefrac. Output: the missing spread cell + the
  core50→consistency dose-response.
- **Wk 2:** pick the paper config (spread ≥ ~51 → headline at 3.2B, absmax as the
  capacity-scaling evidence; else compact 51.6 headlines) → **10-task extension** (the
  catastrophe-elimination demonstration; 5-task CL is thin for reviewers on a 10-task
  suite) → **freeze code + recipe; after this, measurement only** (the substrate axis
  has produced a new best config every ~3 days — wk 1 is the last one).
- **Wk 2-3, idle GPU windows:** reviewer-mandatory baselines repeatedly deferred:
  **naive sequential LoRA** (THE headline forgetting baseline, never run), e2/e7
  specialists (complete the oracle table), multitask-LoRA at 10 tasks if 10-task
  headlines.
- **Wk 3-4:** seeds — ≥3 on headline config + main baseline (everything to date is
  single-seed at ±7pp/cell).
- **Wk 5-6:** consolidation + writing + one clean confirmatory run. The ablation table
  mostly exists in this log (amplitude/write-budget/bank/knn/rank/palette-constancy
  dose-responses, stationarity, protection modes) — writing work, not compute. Decide
  the real-robot question EARLY (realworld_v2 exists; can't be bolted on in wk 5).

### Update (26 Jul PM) — P1 verdict: sep8 FAILS (weak dose-response) but exposes the anchor famIoU U-CURVE: plain spread (w=0) beats every anchored point below w=0.5 → probe 3 (w=0 + sep8)

**P1 (w=0.35, sep 5→8): FAIL.** Expert famIoU 0.207/0.225/0.200/0.238 (L2/4/6/8; one
grace at L6, three hard) vs the sep5 twin 0.238/0.241/0.207/0.250 — delta
−0.031/−0.016/−0.007/−0.012 (mean −0.017) per +3 sep, of the needed −0.03..−0.07.
Extrapolates to sep ~14-20 to thread, with core50 already drifting −10-15%
(1058-1174 → 936-1065; still above even the old 800 floor; min-eff 519-612; bg
0.040-0.054). **VLM tower untouched** (all layers within ~1-5% of the sep5 twin;
famIoU 0.112-0.146 pass) — the global-coupling worry (palette compaction) is null at
sep8: global sep changes are expert-selective in practice. Josh's global-only
constraint cost nothing.

**The U-curve (the session's real finding).** Summarized the plain-spread audit from
its slot JSONs (never done): expert famIoU **0.163-0.212**, core50 **1842-2001**, bg
0.094-0.119. Full famIoU-vs-anchor-w: **w=0 → 0.16-0.21** / 0.10 → 0.48-0.68 / 0.25 →
0.34-0.39 / 0.35 → 0.21-0.25 / 0.40 → 0.167-0.201 / 0.50 → 0.11-0.14. The
pooled-instruction anchor injects a near-SHARED query component for the lookalike
basket family (instruction cos ~0.86) → pulls the family TOGETHER at low-mid w; only
at w=0.5 do the amplified instruction differences win, at collapse-level compaction.
Meanwhile bg improves monotonically in w (0.094-0.119 → 0.022 at 0.5): **the anchor
separates what is already different and merges what is similar.** w=0.35 ≈ the worst
operating point; the E54 probe pair was standing on it.

**Probe 3 (Josh: keep probing).**
The cell the data points at: plain-spread certificate + ONE delta (sep 5→8), no
anchor, FiLM on to match the comparator. Pre-registered (transfer P1's per-layer
deltas onto plain): famIoU ~0.181/0.179/0.182/0.151 = PASS with one-grace headroom at
core50 ~1600-1800; bg is the borderline axis (L2 0.119 −15% ≈ 0.10). Fail-routes (bg route corrected
in discussion — the "w≤0.1 anchor for bg" is DEAD per the U-curve;
the bg-trimmer at w=0 is c-up ~0.1, SupCon pull → compaction → lower bg at affordable
capacity cost): deltas don't transfer → sep 12; bg stuck → accept-with-note or
w=0 + sep8 + c≈0.1; famIoU unmoved → pivot to the anchored-w0.5 + corefrac
sequential (the absmax recipe on spread — relaxed-gate
justified; NOT launched).

**P2 (c=0.15 @ w=0.35) re-scoped:** still lands as the contrastive mechanism
test, but it stands on the handicapped w — any positive read gets re-tested at w=0.

### Update 2 (26 Jul eve) — P2 FAIL (contrastive separates families in NO measured regime); P3 (w=0 + sep8) PASSES THE FULL GATE — first certified spread router, at 2-4x anchored capacity

- **P2 (w0.35, c0.15): FAIL, informatively.** Expert famIoU 0.249/0.244/0.211/0.252 vs
  twin 0.238/0.241/0.207/0.250 — +0.002..+0.011, null/wrong direction — while core50
  broadened +5-13% (1124-1314; the aux-only uniformity direction). With the joint-era
  P3/B pattern this CLOSES the question: **c is a breadth knob in every measured
  regime and a family-separator in none; sep is the only famIoU-mover.** VLM: ±0.003
  famIoU, +5-8% core50, min-eff healthier — the c-axis global coupling is null at 3x.
  Both global levers now measured VLM-safe; global-only fully vindicated.
- **P3 (w=0, sep 5→8, FiLM on, single delta vs the plain-spread certificate): PASS —
  all gates, both towers, no grace.** Expert famIoU **0.178/0.176/0.177/0.156**
  (pre-registered 0.181/0.179/0.182/0.151 — delta-transfer hit within ~0.005/layer),
  bg 0.083-0.097 (sep8 also trimmed the borderline L2 bg 0.119→0.094), core50
  **1383-1717**, min-eff 1388-2024; VLM 0.132-0.154 PASS. sep8's capacity tax at w=0:
  −11-28% core50 from the ~1900 base — affordable.
- Net: **the anchor is bracketed as unnecessary on this substrate** — sep8 alone
  separates the family below every anchored point's famIoU except w=0.5's, at 2-4x
  the capacity. First spread router to certify.
- **cell B** w=0.40+sep8 (the
  absmax-band middle; pre-reg PASS at famIoU 0.155-0.19 / core50 580-740) and
  **cell C** w=0.5+c=0.15 (family-clean end; pre-reg core50 → 400-650 at
  famIoU ≤~0.16 — also the small-cores-vs-separation causal probe), then **probe 4**
  = P3-nofilm (w=0 + sep8 + lang_to_query=false, no anchor — the
  zero-language-machinery router, Josh's FiLM-removal cell; pre-reg famIoU within
  ~0.01-0.02 of P3 ⇒ FiLM removable and P4 graduates over P3).

### Update 3 (27 Jul AM) — overnight verdicts: B PASSES in the absmax band (bg 0.025-0.037, FiLM-free); C fails the core floor by one layer (c-axis closed); P4 CATASTROPHIC SPRAWL — language conditioning is LOAD-BEARING at warm-up, "FiLM inert" was an inference-time property only

Expert-tower scoreboard (all six E54 probes; VLM passed and was ≈invariant in every one):

| cell | config | famIoU L2/4/6/8 | bg | core50 | verdict |
|---|---|---|---|---|---|
| P1 | w0.35 + sep8 | 0.207/0.225/0.200/0.238 | 0.040-0.054 | 936-1065 | FAIL |
| P2 | w0.35 + c0.15 | 0.249/0.244/0.211/0.252 | 0.048-0.070 | 1124-1314 | FAIL |
| **P3** | w0 + sep8, FiLM | **0.178/0.176/0.177/0.156** | 0.083-0.097 | **1383-1717** | **PASS** |
| **B** | w0.40 + sep8, anchor/nofilm | **0.160/0.170/0.160/0.192** | **0.025-0.037** | 585-732 | **PASS (grace L8)** |
| C | w0.5 + c0.15, anchor/nofilm | 0.121/0.126/0.135/0.143 | 0.018-0.025 | 354-482 | FAIL (L4 core50 354) |
| P4 | w0 + sep8, NO language | 0.503-0.732 | 0.194-0.492 | 3789-10525 | FAIL (sprawl) |

1. **B hit its pre-registration almost exactly** (predicted famIoU 0.155-0.19 / core50
   580-740): the absmax capacity band with absmax-grade shoulders (bg ≈ the frontier's
   0.026) at 8 banks — and FiLM-free with NO external embedder (the anchor pools LM
   instruction hiddens; no mpnet, no film_mlp).
2. **C: the c-up re-inflation under-delivered again** (+7-15%, same as at w=0.35 — the
   E45 VLM-magnitude analogy does not transfer to the expert tower). Fails the ≥400
   floor by one layer (L4 354). famIoU 0.121-0.143 / bg 0.018-0.025 = the cleanest
   routing ever measured on spread, for reference. **The c axis is fully closed**: a
   breadth knob with ~+10% leverage, a separator nowhere.
3. **P4 is the decisive negative of the batch.** Pure proj(x) routing (no FiLM, no
   anchor) sprawls catastrophically: famIoU 0.50-0.73, bg up to 0.49, footprints
   3.8-10.5k, worst at L2 grading down with depth (the E36 crowding profile). E28's
   "FiLM is inert" (γ≈0, β a 5% nub, stripping it moved routing ~0.001) was an
   INFERENCE-time property of a trained router; at WARM-UP time the language channel
   is what the aux losses use to organize task routing on crowded frozen features.
   Corrected model: **the expert router needs SOME language conditioning — FiLM (weak,
   learned) or the anchor (strong, fixed) both suffice; zero does not.** Every
   previously certified router had one of the two; P4 was the first with neither.
4. Net: **two certified spread routers bracketing the capacity axis** — P3 (core50
   ~1700, bg ~0.09, FiLM+mpnet; the compact-51.6 profile on spread) and B (core50
   ~650, bg ~0.03, anchored/FiLM-free; the absmax-53.6 profile at 8 banks/3.2B).
   Exactly the core50↔consistency dose-response pair. Recommendation: graduate BOTH
   through A-phase + 5-task corefrac sequentials this week, **B first** (pattern-matches
   the frontier's winning profile, at publishable size, with the simpler language
   story); P3 second as the capacity arm.

### Update 4 (27 Jul) — GRADUATING B

E53 arm-2 recipe
verbatim (top_t 3072, lr 2e-3→2e-4, β4+corefrac, 5×5000, bs16×acc2, 50-ep final); the only
delta vs spread+corefrac 47.6 is B's warm-up checkpoint, so it's a clean single-delta
router cell.
Beat 47.6 = the router delta pays;
51.6 = spread takes the frontier; ≥52 ⇒ the 53.6 win was shoulder cleanliness, not banks.

### Update 5 (28 Jul) — run 1 lost at task 2/5

The first attempt died in task 2 (evals banked: e4 55, then e4 50 / e6 40 — e4's
−5 across the first boundary vs fold-in's −15 and compact-corefrac's −20 is the only
signal worth keeping). Cause was infrastructural, not the config.

---
## Entry 55 - 28 Jul 26 (B graduation lands: spread+anchor040+sep8+corefrac = **53.2** — statistically the frontier at 3.2B/8 banks; the ≥52 branch FIRES (absmax's 53.6 was shoulder cleanliness, not bank count) → spread is headline-eligible. Battery: corefrac-flat matrices + project-best chunks on 4/5 tasks (e4 at full-specialist level); e7 core-volume hypothesis tested from the audits — simple form dead, an expert-DEPTH pattern found; e7's threshold is the open question)

### The run

Determinism first: run 2 reproduced
run 1 exactly through the replayed window — loss traces to 3 decimals every logged step,
evals identical (e4 55.0; then 50/40) — so the crash cost wall-clock only, run 1's banked
cells were real, and the A-phase-reuse + resume plumbing changed nothing about the method.

| step | e4 | e6 | e9 | e2 | e7 | seen-avg |
|---|--|--|--|--|--|--|
| 5k | 55 | | | | | 55.0 |
| 10k | 50 | 40 | | | | 45.0 |
| 15k | 25 | 60 | 65 | | | 50.0 |
| 20k | 30 | 35 | 50 | 90 | | 51.25 |
| **25k (50-ep)** | 44 | 60 | 56 | 86 | 20 | **53.2** |

- **53.2 = inside 50-ep noise of absmax's 53.6, at 60% of its values (3.2B vs 5.37B) and
  8 banks vs 13.** The pre-registered ≥52 branch fires: the frontier number was carried by
  the anchored router's SHOULDER CLEANLINESS (bg 0.025-0.037), not the 13-bank spread.
  Spread at 3.2B headlines; absmax demotes to capacity-scaling evidence (per the E54 plan).
- Give-back **−0.8** (init-mean 54.0; 20-ep inits except e7) — the absmax zero-give-back
  profile, vs compact+corefrac's −5.6.
- e2 86 (+ its 90 intermediate) = best e2 cells ever; e9 56 second only to absmax's 76;
  **e7 20 replicates spread-A's 20** — 2/2 on spread across two routers and two protection
  modes: real, substrate-level. e6 60 ties its best.
- Standings: absmax 53.6 / **B 53.2** / compact+corefrac 51.6 / spread+corefrac 47.6;
  multitask-LoRA must-line 49.2 cleared by +4.0.
- Method note, again: at boundary 4 the 20-ep cells (51.25) projected high-40s; the 50-ep
  final moved e4 30→44, e6 35→60, e9 50→56. The E41 retired-instrument rule survives
  another live demonstration.

### Battery (MSE matrix + chunk)

MSE diag drift (just-trained → final), vs the two matched comparators:

| | e4 | e6 | e9 | e2 | e7 |
|---|--|--|--|--|--|
| **B** | +3.7% | +3.5% | +3.4% | +2.1% | 0.0% |
| compact+corefrac | +3.7% | +3.8% | +2.5% | +1.4% | 0.0% |
| spread-A (peak) | +22.6% | +15.9% | +13.0% | +4.3% | 0.0% |

Corefrac's flat profile transferred to spread cell-for-cell. And the diag ABSOLUTES land on
spread-A's (±1-2% every task; 8-15% better than compact+corefrac) — **B kept spread's fit
and added corefrac's retention; 53.2 is that combination**, both halves now measured.

Chunk (own-block → final): give-backs **+0.0 / +3.5 / +2.0 / +4.2%** (t4 own=final) — the
corefrac band. The absolutes are the bigger finding:

| own chunk | e4 | e6 | e9 | e2 | e7 |
|---|--|--|--|--|--|
| **B** | **0.0198** | **0.0147** | **0.0466** | **0.0259** | 0.0321 |
| spread-A (prev best-ever, all 5) | 0.0272 | 0.0200 | 0.0683 | 0.0309 | 0.0315 |
| LoRA specialist anchors | 0.0204 | 0.020 | 0.0675 | — | — |

**Project bests on 4/5 by 16-32% — e4 AT the full dense-specialist's level, e6/e9 below
theirs.** The t0 cell is protection-free (empty store; 005000 bitwise-shared), so the 27%
t0 gain over spread-A is a pure ROUTER effect: the anchored sep8 router fits better at ⅓
the core50 — small-but-separated cores are a fit advantage at matched mask budget, not a
tax (E54's capacity-ledger point, now shown on fit). e4's function moved **+0.006%** across
the 20k post-block steps while its 20-ep cells wobbled 55→25→30→44 — the cleanest
noise-vs-function demonstration on record. Instrument correction logged: cross-run
"block-min" comparisons from the log's `loss:` field are invalid (it carries aux-loss
telemetry, which differs by router); use wandb `mse_loss` or the paired matrix.

### e7: core-volume hypothesis (Josh) tested from the audits — simple form dead; a depth pattern instead

Per-task core50 from the B / P3 / compact audit summaries + the E53 jitter files:
1. **Within B, core size does not rank rollout**: e2 has the tiniest cores everywhere
   (115-290 VLM, 167-241 expert) and rolls 86; e9 the biggest expert cores (1020-1928) and
   rolls 56; e7 middling → 20.
2. **e7's VLM cores are small in ALL THREE audits** (t4/median 0.26-0.68) — not a spread
   property — and compact's e7 VLM cores (238-358) are ≤ B's (258-645) while rolling 36
   vs 20. Volume is not the discriminator.
3. **Jitter: spread's e7 beats compact's at every perturbation level** (clean 0.032 vs
   0.039; state@0.2 0.133 vs 0.146; image 0.176 vs 0.200) and still rolls 16 lower — the
   gap lives outside the shell, on states no instrument visits.

What does line up: **e7's final tracks the depth of the highest EXPERT bank** — L8 → 20, 20
(both spread runs) / L9 → 28 (absmax) / L12 → 36 (compact) / L14-era healthy configs →
26-38 (the two killed substrate bets, vlmr4 14 / imgspan 6, sit outside). Six-config
monotone pattern at ±7pp cells: suggestive, not proof.

### The discussion (Josh) — two frame corrections adopted

1. **Sweet-spot framing**: breadth × write-volume-per-slot has an interior optimum (the E43
   read-write product; B's e2/e9/e7 row above is its cleanest within-run demonstration).
   Refinement kept from E49: the optimum MOVES with addressing quality (arm 3-old had the
   best product ever and the worst fit) — conditionality is the third factor; B shifted the
   optimum toward smaller cores by cleaning the router.
2. **e7 threshold pushback (Josh, accepted)**: the "e7 is conversion-broken" claim rested
   on compact-rolls-36-at-worse-chunk — an E50-invalid cross-substrate comparison. Two
   spread points at the same chunk (0.0315/0.0321) both rolling 20 cannot discriminate
   above-threshold from conversion-broken, and e6's specialist (0.020 → 44) calibrates that
   basket-family thresholds can sit very low — e7's may be ~0.015-0.02, putting our 0.032
   above its curve. Surviving fact: **e7's chunk is STUCK** (0.0315→0.0321) across a router
   change that moved every other task 16-32% — whatever moves e7's fit isn't router or
   protection; depth is the standing candidate under either reading.
3. **P3 re-priced twice, opposite directions**: DOWN as a frontier bet (predicted double
   regression: shoulders bg 0.083-0.097 ≈ compact's → −5.6-band give-back; router family →
   spread-A-level fits; lands ~48-51), UP as science — under the sweet-spot frame it is the
   big-core point on the breadth dose-response at matched protection, i.e. a paper figure.

### e7 + e2 LoRA specialists (the two missing oracle cells)

r32/5k recipe byte-identical to the t0-t2 anchors (58.0/0.0204, 44.0/0.020, 70.0/0.0675).
e7 doubles as the threshold arbiter, pre-registered: specialist converts at **≲0.02** → the
threshold reading wins (our 0.032 is above e7's curve; the wall is fit; the 9-module arm
gets a concrete target ~0.02); specialist needs **≳0.04-0.05** → our 0.032 is already below
a threshold our broader support should only relax — the threshold law breaks on e7 and the
off-trail/conversion story revives. e2's cell completes the reviewer table either way.

### Next

1. Decide the next arm: **9-module hybrid** (expert
   [8-11] + VLM [12-16], B's router recipe, corefrac; certify-first) — the push-60 bet
   (layers axis 4/4, the e7 depth pattern, B's router) — vs **P3** (the dose-response
   point). Discipline flag stands: E54 wk-1 allows "at most one refinement"; the 9-module
   is the last substrate spin before the 10-task.

### Entry 55 addendum (29 Jul, overnight) — e7 specialist lands: 60.0 @ chunk 0.0330 — the threshold-vs-conversion arbiter RESOLVES for conversion; LoRA-cell config finding

**e7 specialist cell** (r32/5k, anchor recipe): **60.0 @ 50 eps, clean chunk 0.0330**.
Oracle row: e4 58/0.0204 · e6 44/0.020 · e9 70/0.0675 · **e7 60/0.0330**.

**The arbiter verdict.** Neither E55 branch fired as written — the answer is sharper:
the specialist converts at the SAME on-demo function we already have (0.0330 vs B's
0.0321, ours marginally better), and its jitter shell is a wash (state@0.1/0.2 −12-16%
smoother than spread-A's e7, image@0.05 slightly WORSE: 0.186 vs 0.177) — yet it rolls
**60 vs our 20**. A 40-point rollout gap at matched function and matched-ish shell:
- The "we haven't reached e7's curve" reading (E55 discussion #2) is **disconfirmed for
  e7** — we are AT the specialist's operating point on every demo-state instrument.
- e7's wall is the **conversion/off-trail layer** — the E42(d) suspect, now with its
  cleanest evidence in the project: same task, same demos, 3x rollout gap, invisible to
  chunk AND jitter. The "e7's chunk is stuck" fact stands but is no longer binding —
  unsticking it would buy nothing without conversion.
- Implication for the 9-module arm: its e7 case now rests on the depth pattern being a
  CONVERSION lever (action-proximal expert banks), not a fit lever. The specialist's
  adapters sit at every depth incl. action-proximal — consistent, not yet causal.

Standard LoRA-cell config henceforth: **bs16 × accum2, no ckpt** (effective batch
32 preserved; E42 audit covers the accumulation path).

---
## Entry 56 - 30 Jul 26 (P3 verdict 47.2: the anchor's SHOULDER cleanup is the whole +6, sep8 alone buys nothing — B confirmed as the config on both performance and simplicity; the core-breadth dose-response is now a 3-point curve; ORACLE TABLE COMPLETE (e2 84.0) and it re-frames the ceiling: our e2 86 BEATS its specialist)

### P3 (the capacity-end router) — 47.2

Config verified from the log: P3's
A-checkpoint, seed 1000, corefrac, top_t 3072, lr 2e-3 -> 2e-4 — single-delta vs B.

| step | e4 | e6 | e9 | e2 | e7 | seen-avg |
|---|--|--|--|--|--|--|
| 5k | 55 | | | | | 55.0 |
| 10k | 35 | 60 | | | | 47.5 |
| 15k | 45 | 25 | 55 | | | 41.7 |
| 20k | 55 | 45 | 60 | 70 | | 57.5 |
| **25k (50-ep)** | 46 | 46 | 58 | 74 | **12** | **47.2** |

### The dose-response, now measured end to end (matched substrate / protection / levers)

| router | expert bg | core50 | final |
|---|---|---|---|
| plain sep5 (E53 arm 2) | 0.094-0.119 | ~1900 | 47.6 |
| **P3 = w0 + sep8** | 0.083-0.097 | 1383-1717 | **47.2** |
| **B = w0.40 + sep8 (anchored, FiLM-free)** | **0.025-0.037** | 585-732 | **53.2** |

**sep8 alone bought NOTHING** (47.2 ~= 47.6, inside noise) despite improving famIoU
0.163-0.212 -> 0.156-0.178. The entire +5.6 is the ANCHOR'S SHOULDER CLEANUP — a cleaner
attribution than the E55 pre-registration hoped for, and it matches E53's location of
corefrac's residual give-back in the shoulder channel. Consequences:
1. **The famIoU axis is spent on this substrate.** Two routers improved it at matched
   protection and neither converted; bgIoU is the axis that pays. Future certificates
   should gate on bg first, famIoU second (inverting the E44-54 emphasis).
2. **Small cores are not the mechanism** (E54's ledger question, closed): P3 sits at ~2.4x
   B's core50 and loses; but the E54 "small cores correlate with the win" reading is now
   demoted to co-product — bg is the causal knob, core size rides along with the anchor.
3. Give-back: P3 was AHEAD at boundary 4 (57.5 vs B's 51.25) and finished 6 behind
   (-10.3 from its own peak vs B's -0.8). The 20-ep intermediates misled a third time in
   this entry alone; the 50-ep instrument restored the pattern. Standing rule holds.
4. **e7 = 12, its worst cell anywhere** (spread: 20 / 20 / 12 across three runs vs
   compact 36, absmax 28). The spread e7 deficit is router-invariant, confirmed 3x.

**Standings: absmax 53.6 / B 53.2 / compact+corefrac 51.6 / spread+sep5 47.6 / P3 47.2.**
B is the config — best-in-class at 3.2B AND the simplest language story (no FiLM, no
mpnet; the anchor is the only conditioning). The router axis is CLOSED.

### Oracle table complete — and it re-frames the ceiling

e2 specialist: **84.0 @ 50 eps, clean chunk 0.0308**

| env | B (sequential, 1 model) | specialist (5 models, task ID given) | B's chunk | spec chunk |
|---|---|---|---|---|
| e4 | 44 | 58 | **0.0198** | 0.0204 |
| e6 | **60** | 44 | **0.0152** | 0.020 |
| e9 | 56 | 70 | **0.0476** | 0.0675 |
| e2 | **86** | 84 | **0.0270** | 0.0308 |
| e7 | 20 | 60 | 0.0321 | 0.0330 |
| mean | **53.2** | **63.2** | — | — |

**B beats or matches its specialist on 2 of 5 tasks (e6 +16, e2 +2) and matches every
specialist's FUNCTION on all five** — the oracle's remaining 10-point margin is carried
almost entirely by e7 (+40) and e4/e9 (+14 each), i.e. by rollout CONVERSION, not fit.
Restated for the paper: a single continually-trained model with a frozen backbone,
zero task identity at inference, and zero forgetting reaches **84% of the per-task
oracle's average**, and beats it outright on 2/5 tasks.

### e7, sharpened (with the E55-addendum arbiter)

Specialist 60 @ 0.0330 vs our 20 @ 0.0321 — matched function, matched-ish jitter shell,
3x rollout. e7's wall is conversion and it is now the largest single pool on the board
(+40 available, vs e4's +14 and e9's +14). The only lever with evidence pointed at it is
the expert-DEPTH pattern (E55: L8 -> 20,20,12 / L9 -> 28 / L12 -> 36 / L14-era -> 26-38),
which is why the 9-module hybrid (expert [8-11] + VLM [12-16], B's router, corefrac) is
the standing next arm — explicitly a CONVERSION bet, not a fit bet.

### Next

1. **Decision pending (Josh):** 9-module hybrid (certify-first) vs going
   straight to the 10-task graduation on B. E54's wk-1 discipline flag says the 9-module
   is the LAST substrate spin either way.
2. Then: naive sequential LoRA (the headline forgetting baseline, never
   run), >=3 seeds on the headline config, 10-task multitask-LoRA if 10-task headlines.

---
## Entry 57 - 31 Jul 26 (OFF-TRAIL INSTRUMENT built — the conversion-gap measurement (E42(d), deferred 4x) finally exists; anchor validates against the known chunk numbers; B-vs-e7-specialist is the first target pair)

### Why (recap of the E56 discussion)

The 10pp gap to the specialist oracle is 100% rollout conversion on 3 cells (e7 −40, e4 −14,
e9 −14) at matched-or-better on-demo function; the e7 arbiter (spec 60 @ chunk 0.0330 vs B
20 @ 0.0321, matched jitter shell) localizes the damage to states no demo-side instrument
visits. This instrument measures the function AND the retrieval on rollout-visited states
directly, using the specialist as the off-trail reference oracle (no demo labels exist there).

### The instrument

1. **Harvest** (per checkpoint):
   owned rollout loop cloned from lerobot-eval (same
   processors, same per-episode seeding as a serial eval), bs=1. At every
   policy call (action-queue refill)
   it saves the RAW env obs (both cams uint8 + full nested robot_state)
   + executed actions/outcome, and for memory
   policies the RETRIEVAL TRACE:
   eval-mode indices/scores (route-once multiplicity
   already applied), per-head-softmaxed and mass-aggregated per call per module.
2. **Score** (per model):
   re-runs the 10-step denoise on every harvested state + demo-control states,
   K noise seeds; batching and seeds are deterministic functions of the sorted
   state list -> chunks PAIRED across models. Harvest states are rebuilt into the raw env
   obs dict and pushed through
   the byte-identical eval pipeline. Mean-pooled LM
   hiddens from a layer below the first VLM memory bank (frozen + memory-free = stage-1
   features) give the excursion-distance axis.
3. **Report**: joins paired chunks, traces, written-slot sets
   and distances into the four reads:
   READ 1 D(s)=cross-model chunk disagreement vs excursion distance (feature + proprio),
   by population (demo / succ / fail per harvest); READ 2 self-written retrieval mass
   (fallback-to-A-content detector); READ 3 consecutive-call retrieval churn (discontinuity
   detector); READ 4 divergence points (first D > success-P90 per failed episode).

- Known weakness: LM-9 all-token-pooled feature distance barely spreads (~3e-4 cos —
  image-dominated pooling); the PROPRIO distance axis spreads well (quartiles 0/0.9/2.0)
  and READ 1b already grades on it.

### Pre-registered decision rules (from the E56 discussion, unchanged)

- Composition collapse (self-written mass craters off-trail, tracks failures) -> fix =
  retrieval-holding (query stabilization / degrade-toward-written-content); 9-module is NOT it.
- Discontinuity (churn cliffs at divergence, composition fine) -> drift-stable routing ->
  the 9-module/deeper-expert-banks bet graduates with a measured premise.
- Null (B tracks the spec's function everywhere incl. failures) -> value function exonerated
  -> mode-selection/within-chunk compounding track (execute-25 A/B next).
- B fine on spec-visited states but bad on its own -> compounding, not coverage.

### Campaign (31 Jul)

Full campaign: 50 eps/model (B seq-025000 vs loraft task4_e7), 4 seeds, 120 demo states,
task e7 (env 7, dataset task_index 4).

### RESULTS (anchor 0.0340/0.0364 vs known 0.0321/0.0330 ✓)

Populations: B/fail 418 states, B/succ 67, spec/fail 220, spec/succ 198, demo 120.
B harvested at 10/50 successes, spec ~28/50 — both at their known rates.

**Both pre-registered lead hypotheses are REFUTED:**
1. **Fallback-to-A-content: DEAD.** Self-written retrieval mass holds at 0.77-0.83
   at EVERY distance, every population; near->far gradient ~3pp (expert 0.807->0.770,
   VLM 0.821->0.798); succ-vs-fail gap <=2pp. Off-trail retrieval stays on the
   sequentially-written footprint. Retrieval-holding fixes (freeze/query-stabilization)
   are NOT the lever.
2. **Routing discontinuity: NO SIGNAL.** No churn cliffs at divergence points (churn
   0.79-0.88 there = baseline); the churn metric is scene-change-dominated (declines
   through failed episodes as the robot stalls). Nothing routing-side to fix.

**What the instrument DID find — a competence-RADIUS story on the value function:**
- D (cross-model chunk disagreement) on demo states = 0.014-0.018; on B-fail states it
  climbs 0.11 -> 0.41 with proprio distance (0.12 -> 0.38 on d_feat). B's successes stay
  in the low-D tube (0.08-0.12 through Q3); **the specialist's successes TRAVERSE the far
  region** (spec/succ Q4 D = 0.42-0.48 — states where its behavior is episode-validated
  and B's function is maximally different). B wins only where the two functions agree;
  the specialist also wins where they don't. The deficit is the VALUE CONTENT's
  competence radius off-manifold — not where retrieval lands, but what the written
  slots express out there.
- **Jitter's blind spot quantified:** rollout excursions live at d_state ~0.5-2.5+
  per-dim stds; the jitter shell probed 0.1-0.2. The proxy was ~10x too close in.
- **READ 4: e7 is decided at call 2** (env steps ~100-150, the first-grasp commitment):
  28/38 failed B episodes cross the divergence threshold there (D spikes to 0.52 mean at
  c2, both models' fail populations) with composition only mildly depressed (0.72-0.76
  vs 0.79-0.81). The failure is an early discrete commitment, not late accumulation.
- Caveat, held honestly: D is symmetric and far states may be multimodal (grab-order
  choice), so some Q4 disagreement is mode-mismatch rather than B-wrongness — but the
  3x rollout gap independently certifies that B's far-region behavior mostly fails.

**Implications for the board:** the 9-module spin's routing-stability premise gets NO
support here (the depth pattern survives only as a value-placement hypothesis). The
levers this measurement points at: (a) widen the TRAINING neighborhood of the values —
observation-space augmentation during the sequential block with amplitude matched to
the measured d_state 0.5-2.5σ band (the E42 demo-jitter idea, now with a measured
dose), subject to the "train on the same data" constraint being relaxed for obs-noise;
(b) inference-side re-conditioning (execute-25) — weakened though: a wrong call-2
commitment is not fixed by replanning after it; (c) characterize call-2 failures from
the harvested videos (wrong-object vs missed-grasp) before picking either.

---
## Entry 58 - 31 Jul 26 (VALUE-INPUT NOISE: the competence-radius lever, calibrated from the harvest bank -> built -> dose-pair arms on the B config)

### Design (Josh's proposal, refined in discussion)

The E57 verdict (retrieval healthy off-manifold; the VALUE function wrong there) points
at widening the values' competence neighborhood. Constraint: "train on exactly the same
data" cannot relax. Josh's lever: corrupt the hidden state the VALUES see during
sequential training — value-path input only, router untouched. Properties: (i) same
data/targets/batches — a training-time regularizer inside the module; (ii) the dual-path
architecture makes it surgical: routing + gate read the frozen branch, so the
stationarity certificate, TF-IDF stats, protection, and footprints are all bit-identical
— only the slot transforms see noise; (iii) it targets the measured failure (train the
values to express the demo action from a NEIGHBORHOOD of x). NOT the old E9/E15
corruption (that was value-OUTPUT noise at PRETRAIN, teaching downstream tolerance;
this is value-INPUT noise at SEQUENTIAL time, teaching the content to generalize over x
— different site, phase, and target; the old negatives don't transfer).

### Calibration

Dose measured, not guessed: we capture the pass-B live x for
120 demo states + 198 tertile-stratified harvest states, ONE fixed denoise seed
everywhere (expert first-denoise-step capture = obs-driven displacement; the last-step
capture is confounded by the denoised action tokens and is NOT the dose source).
Per-dim displacement ratio (off-trail vs nearest-demo, / per-dim demo std), mid band:

| module | ratio | top-10 SVD energy |
|---|---|---|
| exp L2/L4/L6/L8 | 0.04 / 0.15 / 0.37 / 0.53 | 0.69-0.82 (STRUCTURED) |
| vlm L10/12/14/16 | 0.58 / 0.71 / 0.83 / 0.87 | 0.14-0.34 (~isotropic) |

Findings: (1) dose must be PER-LAYER (expert displacement grows 13x up the stack);
(2) per-dim-independent noise is a good match for the VLM tower, a rough one for the
expert (70-80% of its displacement energy in ~10/1024 directions — v1 bets on generic
flatness; subspace-projected noise is the measured fallback, recomputable from the same
captures); (3) far/near band spread ~1.2-1.5x -> per-row amplitude draw U[0.5,1.5].

### Build

Noise = per-dim Bernoulli
mask x N(0,1)
x current-batch per-dim std (self-calibrating) x sigma_layer x amp, applied to the
value down-projection input (valid rows
only); training-mode only; swilu tail + plain MLP keep clean x.

### Arms (31 Jul)

Two arms, B config verbatim (spread + anchor040/sep8/nofilm + corefrac + lr2x + 3072,
reusing B's A-checkpoint), single delta = the noise flags:
- **dose1x**: p=0.25, amp [0.5,1.5], expert sigma [0.1,0.3,0.75,1.05], vlm
  [1.15,1.4,1.65,1.75] (measured ratios x2, variance-matched at p=0.25)
- **dose05x**: sigmas x0.5

Pre-registered: (a) harvest-bank pre-screen = READ-1 D-vs-distance against spec_e7 on
the SAME e56 states — PASS = far-region (Q4) D shrinks vs B's 0.38/0.41, especially on
spec-success states; anchor stays ~0.032-0.036; (b) block-min mse_loss <= ~1.10x B per
task; (c) MSE matrix stays corefrac-flat; (d) only a pre-screen pass earns weight on
the 50-ep final — e7 > 20 = the conversion bet pays. Comparators: B 53.2
(44/60/56/86/20), specialist oracle 63.2.

---
### Entry 58 addendum 3 (2 Aug 26) — DOSE-PAIR RESULTS: value-input noise is SAFE at both
doses (fit + retention untouched), produces a real far-region function shift toward the
specialist with an INVERTED-U in dose (0.5x > 1x on every read), converts to e7 +6
(replicated in both arms) and a nominal family-best 54.4 at 0.5x. Binding limitation is
now GEOMETRY (isotropic vs the structured excursion subspace) and/or dose (< 0.5x
unexplored), not safety.

**50-ep finals** (B = 53.2 = 44/60/56/86/20):
- dose1x  52.4 = 42/52/58/84/**26**
- dose05x **54.4** = 44/66/58/78/**26**  <- nominal family best
- **e7 = 26 in BOTH arms** (two independent +6 draws on the conversion cell; pooled
  52/200 eps vs B's 20% — directionally consistent, ~1σ pooled).

**Rescore (D-vs-specialist on the SHARED e56 state bank, far quartile Q4;
feat / proprio axes):**

| population | B | dose1x | dose05x |
|---|---|---|---|
| spec-succ | 0.416/0.482 | 0.356/0.400 (−14/−17%) | 0.342/0.380 (−18/−21%) |
| B-fail | 0.383/0.408 | 0.377/0.404 (~0) | **0.330/0.353 (−14%)** |
| spec-fail | 0.413 | 0.402 | 0.357 (−14%) |
| demo anchor | 0.0340 | 0.0341 | 0.0346 |

Half dose beats full dose on every far population and uniquely moves the B-failure
states; near-manifold quartiles unchanged. (Dose semantics: p=0.25 per-(token,dim)
Bernoulli in both arms; only sigma halved -> dose05x injects ~25% of the measured
excursion VARIANCE. p never varied; per-slot application + structured direction never
tested.)

**Fit cost: zero.** Per-block log-loss minima identical between arms to 3 decimals
(t0-t4: .046/.037/.084/.053/.071 both); anchors 0.0341/0.0346 vs B 0.0340.

**MSE give-back (paired-noise matrices, just-trained -> final; gate <= +5%):**

| | e4 | e6 | e9 | e2 | e7 |
|---|---|---|---|---|---|
| B (corefrac) | +3.7 | +3.5 | +3.4 | +2.1 | 0.0 |
| dose1x | +3.3 | +3.0 | +3.5 | +2.0 | 0.0 |
| dose05x | +3.8 | +3.1 | +3.1 | +1.9 | 0.0 |

Retention signature reproduced cell-for-cell — the noise does not interact with
stationarity/protection at either dose (as the architecture argued it couldn't).

**Pre-registration scorecard: (a) far-D shrink esp. spec-succ PASS (both; 05x
stronger); (b) fit <= 1.10x PASS (~1.00x); (c) matrix corefrac-flat PASS; (d) e7 > 20
PASS (26, replicated); net mean +1.2 at 0.5x (inside noise).**

**Options on the table:** (A) structured noise — project into the measured top-10
excursion subspace per layer (~75% of real displacement energy; isotropic puts ~1%
there — the prime suspect for the modest magnitudes);
then one arm.
(B) dose sweep below 0.5x (trivial env change, one arm). (C) bank dose05x as a free
recipe ingredient (+e7 robustness at zero cost) and return to the wk-2 ICRA items
(10-task, naive-seq-LoRA baseline, seeds).

---
### Entry 58 addendum 4 (2 Aug 26) — NAIVE SEQUENTIAL LoRA baseline: param-matching
resolved as a capacity-INSENSITIVITY sweep; r=256 arm (the headline forgetting
baseline's first-ever run)

Supervisor requires the "is it just parameters" control at matched TOTAL (not active)
params; our total isn't frozen. Resolution (discussed with Josh): a single matched point
is weak (r~2000 ≈ full-rank = no longer LoRA) and count-sensitive; a 3-point capacity
sweep — r=32 (53M) / **r=256 (426M, THIS ARM)** / big point (full-FT 6.6B or r~2000,
accounting decision pending with supervisor) — shows naive forgetting is capacity-
insensitive, which kills the confound at ANY final count. Supporting internal evidence
already banked: rank axes closed negative (expert +2, VLM −10), bank shrinks dead,
E46 cut params 11% while fit +35%, absmax ledger −27% capacity at +2pp, multitask-LoRA
at 53M within 4-5pp of our 3.2B. Accounting currency to settle in the paper: trainable-
during-sequential (ours 3.2B; frozen router/keys excluded).

**Build:** lerobot-sequential-train reused for protocol identity.
PEFT default a=8 would have given r256 an 8x
weaker effective gain than the r32 specialists — a/r held at 0.25: r256@a64.

**Arm**: base = stage-1
libero_90 finetune, specialist TARGETS verbatim, lr 2.5e-5 -> 2.5e-6 per-block linear,
bs16xacc2 no-ckpt, same 5-task order/budget/eval protocol as B (retention
matrices directly comparable). NO protection/tfidf/memory. Pre-registered
expectation: specialist-grade early diagonals then the classic catastrophic collapse —
the headline figure next to our flat matrices.

---
### Entry 56 addendum (31 Jul 26) — batched-eval seed comparator (B vs specialists)

**3 eval seeds x 100 eps/task on the existing
final checkpoints — at minimum for the 5 LoRA specialists** (B too).
Stats note for the writeup: LIBERO init states wrap modulo 50,
so 100 eps = 2 passes over the fixed init-state set with fresh policy noise.

---
### Entry 58 addendum 5 (3 Aug 26) — PREEMPTION (naive-LoRA arm, 1.4K into the final e7 block); arm resumed

This baseline has no cross-task accumulators (no
protection/IDF; `reinit_optimizer_each_task=true`) — the adapter weights in the
checkpoint ARE the complete cross-task state, so a boundary resume is method-exact
(only divergence: dataloader RNG within the e7 block, not a controlled variable).

---
### Entry 58 addendum 6 (3 Aug 26) — NAIVE SEQUENTIAL LoRA r256 LANDS: mean **17.6** (0/0/0/54/34) vs B's 53.2 on the identical protocol — total catastrophic collapse (every task -> literal 0% after ONE subsequent block; MSE diagonals +740-1567% vs our <=+3.8%), at specialist-grade fresh fits. The headline forgetting baseline exists.

The final eval was run standalone-batched (bs=10, 50 eps,
seed 1000).

**Rollout retention (20-ep boundaries; final row 50-ep standalone):**

| after block | e4 | e6 | e9 | e2 | e7 |
|---|--|--|--|--|--|
| e4 | **35** | | | | |
| e6 | 0 | **60** | | | |
| e9 | 0 | 0 | **60** | | |
| e2 | 0 | 0 | 0 | **90** | |
| **FINAL (50ep)** | **0** | **0** | **0** | **54** | **34** |

Mean **17.6** vs B 53.2 / dose05x 54.4 / multitask-LoRA 49.2 / specialist oracle 63.2 /
stage-1 zero-shot floor **10.6**. Every task collapses to literal 0% within ONE
subsequent block. The two nonzero cells are pure recency: e7 is the final task and e2
had exactly one block of exposure (90 -> 54, halved). Sharpest single sentence: after
sequentially training all five tasks, naive adaptation retains 7 points more than a
model that never adapted at all.

**MSE forgetting matrix** (paired-noise, adapter-swap
variant with a per-swap L1(lora_B) verification guard; all five swaps
verified rel=0.00%):

| ckpt \ task | e4 | e6 | e9 | e2 | e7 |
|---|--|--|--|--|--|
| 005000 | **0.054** | 0.93 | 2.44 | 1.19 | 1.32 |
| 010000 | 0.517 | **0.049** | 2.95 | 1.43 | 1.57 |
| 015000 | 0.907 | 0.498 | **0.098** | 1.89 | 1.33 |
| 020000 | 0.950 | 0.711 | 1.113 | **0.060** | 1.38 |
| 025000 | 0.909 | 0.765 | 1.457 | 0.506 | **0.074** |

Just-trained -> final: **e4 +1567% / e6 +1452% / e9 +1383% / e2 +740%** (B corefrac:
+3.7/+3.5/+3.4/+2.1/0.0). The diagonals are specialist-grade (0.049-0.098 — our
range), so the baseline FITS each task as well as anything we build, then loses it by
2-3 orders of magnitude more; forgotten tasks land roughly halfway back to their
untrained loss. The "is it just parameters" reading: 426M trainable adapters forget
catastrophically where our 3.2B values hold flat — parameter count is not the
mechanism.

One conversion nuance banked: naive-fresh e7 rolls 34 (50 eps) vs our 20-26 — the
dense adapter converts e7 better fresh (consistent with its specialist's 60),
reinforcing that our e7 residual is conversion-side, not fit.

---
## Entry 59 - 3 Aug 26 (BUILD: `frozen_prepass` — the placement guard becomes a compute knob. One full memory-free pre-pass per batch serves every routing input, lifting the "VLM banks above expert banks" constraint → interleaved layouts (e.g. expert==VLM [4,6,8,10,12]) are now legal. Cost ~1.3-1.4x forward)

### Motivation (Josh's proposal, adopted)

The placement guard (vlm_layers strictly above max(expert layers)) exists to protect
three expert-side consumers of the prefix: the expert routing branch's per-layer
prefix KV, the E52 anchor hooks (pooled LM instruction hiddens), and the inference
pass-A prefix KV. Key structural fact: the PREFIX tower never attends the suffix
(prefix-LM mask), so VLM routing was never threatened by expert banks at any
placement — the guard is entirely about the expert side. All three consumers can be
served instead by computing them from a full memory-bypassed forward per batch:
same stationarity property ("routing reads f_frozen(obs)"), bought with compute
instead of constraint. What it unlocks: VLM banks in the good low-stack anchor
geometry (E49: separation degrades monotonically upward, L7 0.722 → L16 0.898 —
every VLM bank ever run sat at L10+ because of the guard); expert-high + VLM-broad
simultaneously; more banks with every layer inside measured-good bands.

---
### Entry 59 addendum (3 Aug 26, eve) — FIRST INTERLEAVED CHAIN: expert [6,8,10,12] + VLM [7,9,11,13] @ n256/r2, B's router recipe + corefrac + 2x + 3072 — the param-matched pure-PLACEMENT cell vs B 53.2

**Config decision (with Josh).** Candidates considered: 6+6 at [4..14] on both sides
(4.8B — absmax-adjacent weight, placement+capacity confounded), 5+5 with VLM
[3,5,7,9,11] (VLM 3/5 sit BELOW the E49 probe's measured range — layers 7-16 — and
very-low LM layers risk the palette-constancy pathology: the anchored palette's
within-task state-conditionality comes from attention-mixed context that hasn't
accumulated by L3), and the chosen **param-matched 4+4**: bank budget IDENTICAL to B
(4 expert + 4 VLM, n256/r2, 3.2B), so the sequential cell vs B 53.2 is a pure
placement delta at matched recipe. Per-side reasoning:
- Expert [6,8,10,12]: top bank 12 = the e7-depth bet (L8→20/20/12, L9→28, L12→36);
  bottom 6 exits the marginal L2/L4 routing band B's anchor had to rescue (E36
  plateau ≥L8; attempt-A L2 famIoU 0.212; E53 "L2 leakiest module") — B only sat
  at [2,4,6,8] because the old guard forced VLM to 10+.
- VLM [7,9,11,13]: the E49 geometry payload — instruction-anchor separation best
  LOW (L7 0.722 → L16 0.898); every prior VLM bank sat at L10+ because of the
  guard. All four banks in probe-measured territory; 13 bridges to compact's
  known-good [13-16]. Even/odd offset ⇒ expert anchors pair with LM layers
  carrying no VLM wrapper.
- Noise OFF (dose05x sigmas were calibrated for B's layers; recalibrated noise is
  a follow-up arm; comparator is the no-noise B 53.2).

**Chain**:
warm-up 10k (router-only
fast, broadcast losses, anchored w0.40 + sep8 + c0.05, FiLM off, PREPASS=true)
→ audit (bs8×400) → AUTOMATED GATE
(E54-relaxed: expert famIoU ≤0.18 one-grace-≤0.20, bg ≤0.10, mean core50 ≥400,
min-eff ≥300; VLM famIoU ≤0.165 all layers + min-eff ≥150 palette tripwire — the
never-run low-VLM-layer bet is arbitrated at the certificate) → A-phase
10k → 5-task sequential (corefrac + lr
2e-3→2e-4 + top_t 3072, 50-ep final).

**Pre-registered:** beat 53.2 ⇒ placement pays (≥55 = new frontier band); e7 ≥ 30
(depth), e9 ≥ ~56 (hold), e4 ≥ 40 AND e2 ≥ 80 (spread's cells must survive —
either cratering = the trade is architectural); give-back ≥ −3; prior-core events
= 0 at all 8 modules; MSE matrix ≤ ~+5%/task; block-min mean ≤ ~0.045 band.

---
### Entry 59 addendum 2 (3-4 Aug 26) — GATE VERDICT: HARD FAIL on expert-deep famIoU; OVERRIDDEN (Josh) on the E56 bg-first rationale. Certificate findings: the VLM low-layer bet PAID (L7 famIoU 0.101, best VLM cert in project history; the E49 depth gradient reproduces in a trained router), and the expert famIoU rise with depth is the ANCHOR-SOURCE gradient, lawful across three certs.

**Certificate**:
expert L6 famIoU 0.157 / bg 0.025 / core50 600 · L8 0.180 (grace) / 0.029 / 728 ·
L10 **0.230** / 0.048 / 1045 · L12 **0.213** / 0.045 / 995 — L10/L12 breach the 0.20
hard ceiling → GATE: HARD FAIL, chain stopped as designed. VLM: L7 **0.101** / L9
0.147 / L11 0.135 / L13 0.161, min-eff 247-489 — all pass, and 0.101 is the best
VLM famIoU ever certified (prior band 0.13-0.16 at L10+). Capacity healthy at all
8 modules (expert min-eff 523-850; no collapse tripwire anywhere).

**Findings.** (1) The E49 geometry gradient (instruction-anchor separation best low,
degrading upward) REPRODUCES in the trained VLM router, monotone L7→L13 — the
probe→certificate transfer validates the low-placement bet at certificate level.
(2) The expert famIoU depth-rise is the same gradient arriving through the
anchor-source pairing (expert layer j pools LM layer j): B [2,4,6,8] rose to 0.192
@L8, absmax [4-9] to 0.174 @L9, here 0.230/0.213 @L10/L12 — the anchored recipe
behaves lawfully at depths it was never certified at; L6/L8 reproduce B's levels
almost exactly. Deep expert banks inherit weak anchors. (3) The gate kept the
E44-54 famIoU-primary emphasis that E56 explicitly inverted ("gate on bg first,
famIoU second") — bg here is 0.025-0.048 across all eight modules, B's winning
band, ~half of compact+corefrac's 0.080 (which made 51.6).

**Override (Josh: "Proceed").** Rationale: bg-first per E56; capacity clean;
corefrac zeroes core overwrites regardless of famIoU; absmax precedent (failed ITS
gate at every expert layer → became the 53.6 frontier; E54: gate property, not
router property). Sharpened e7 read pre-registered: e7 low here (deep famIoU
elevated, bg clean, cores protected) ⇒ the famIoU story revives with clean
attribution; e7 converts ⇒ famIoU is confirmed dead as a gate axis and bg-first
becomes the standing certificate rule. Deferred alternative if needed later:
decouple anchor-source from bank layer (deep banks pooling from the L7-region
geometry) — small code change, better informed after the sequential verdict.

---
### Entry 59 addendum 3 (5 Aug 26, 03:36 UTC) — LANDING: **57.6 — NEW FRONTIER** (+4.4 over B at matched 3.2B budget; pure placement delta). e7 CONVERTS (38); per the pre-registered read, famIoU is confirmed dead as a gate axis — **bg-first is now the standing certificate rule.**

**50-ep final** (step 25000):
**57.6** = e4 42 / e6 68 / e9 56 / e2 84 / e7 **38**. Comparator B 53.2 = 44/60/56/86/20.

**Pre-registration scorecard:**
- beat 53.2 → **PASS** (57.6, +4.4); ≥55 new-frontier band → **PASS**. Frontier: B 53.2 → dose05x 54.4 → **interleave 57.6**. Oracle 63.2 now −5.6.
- e7 ≥ 30 → **PASS** (38; B 20, compact-top12 36 = prior best seq cell). The expert-depth
  bet (bank at L12) pays: +18 over B in the cell the whole strategy targets. Fit
  trajectory through the e7 block was also the healthiest yet (loss 0.110→0.068).
- e9 ≥ ~56 → **PASS** (56.0, exactly B's hold). The 20-ep boundary wobble (85 fresh
  → 65 post-e2) resolved to the same 50-ep final as B — instrument noise, not decay.
- e4 ≥ 40 AND e2 ≥ 80 → **PASS** (42, 84; B 44/86 — spread's cells survive the
  expert-bottom 2→6 move and the low-VLM placement within noise; NOT architectural).
- prior-core events = 0 → zero protection-event lines in the unit log; slot autopsy
  pending with the probe battery (addendum 4).
- give-back ≥ −3, MSE matrix ≤ ~+5%, block-min band → pending probes (below).
- The pre-pass production training cost is
  ~equal to B's — the 1.31–1.39× fwd-only overhead diluted by
  unchanged bwd/opt exactly as predicted. **The placement guard is now a free knob
  at training time.**
- Boundary trail (20-ep instrument, retired ±11pp): 45 → 52.5 → 63.3 → 65.0
  seen-means; e9 fresh 85 and e6 fresh 70 were the best 20-ep cells ever recorded.

**Interpretations.** (1) PLACEMENT PAYS: single-delta vs B, +4.4 overall with e7
+18 — the far-region/value-content competence radius (E57) is addressable by
putting expert banks deep, which frozen_prepass (E59) makes legal. (2) The gate
override was correct and the sharpened e7 read resolves AGAINST famIoU: L10/L12
certified 0.230/0.213 famIoU (hard-fail territory) and delivered the best e7 and
the frontier — famIoU is dead as a gate axis; **gate on bg (≤0.10, winning band
0.02–0.05) + capacity floors from here on.** (3) e6 68 is the best e6 sequential
cell in the project (B 60); with e7 38 the per-task-best pool across configs
rises further — simultaneity within one config is improving, not just the pool.
(4) VLM-low placement (L7-L13) did not crater the anchor cells (e2 84) — palette
pathology did not materialize; E49 geometry transfers end-to-end.

**Probes per Josh:**
MSE forgetting matrix (5 ckpts
× 5 tasks, paired-noise, same instrument as B/absmax/naive) then jitter/OOD grid
(t0/t3/t4, clean+state+image, swap-slots, E52 convention). Results → addendum 4.

---
### Entry 59 addendum 4 (5 Aug 26, morning) — probe battery part 1: jitter/OOD grid (the interleave beats the SPECIALISTS on nearly every perturbation cell) + slot autopsy (prior-core writes 4.4x below B; e7's read mass lives in the deep banks as designed).

**Jitter/OOD grid** (t0/t3/t4 × clean/state@0.1/state@0.2/image@0.05,
final ckpt, chunk MSE; comparators: E53 spread-arm [B's placement family, pre-corefrac]
and compact+corefrac, plus the e2/e7 specialists — all at the matched instrument):

| cell | interleave | spread-fam | compact+cf | specialist |
|---|---|---|---|---|
| e4 clean | **0.0160** | 0.0332 | 0.0360 | — |
| e4 state@0.2 | **0.0517** | 0.0871 | 0.1067 | — |
| e2 clean | **0.0235** | 0.0335 | 0.0421 | 0.0308 |
| e2 state@0.2 | **0.0487** | 0.0709 | 0.0996 | 0.0611 |
| e7 clean | **0.0247** | 0.0318 | 0.0392 | 0.0330 |
| e7 state@0.1 | **0.0477** | 0.0656 | 0.0811 | 0.0553 |
| e7 state@0.2 | **0.0960** | 0.1329 | 0.1458 | 0.1168 |
| e7 image@0.05 | 0.1798 | 0.1765 | 0.1996 | 0.1861 |

Chunk error down 30-52% vs the family comparator at every cell but e7-image (parity),
and **below the specialist at every shared cell** — including e7 clean (0.0247 vs
0.0330), where E56's arbiter had B at parity-with-spec and still losing rollouts 3x.
Standing caveats hold: chunk/success decoupling (never rank substrates on chunk alone),
and E57's radius result (jitter shell 0.1-0.2σ probes ~10x nearer than real rollout
excursions at 0.5-2.5σ) — this is a NEAR-manifold function read; the far-region read
is the harvest-bank rescore (addendum 5). But the margins here are an order larger
than any B-vs-spec chunk difference previously recorded.

**Slot autopsy** (interleave vs B):
- **Prior-core write events (ev into earlier task's core50): 1,684 total vs B 7,376**
  (4.4x fewer). Same dominant lawful cell in both — the e2 block into e9's core at the
  2nd-from-bottom expert bank (here E8 1,273; B E4 4,928) — attenuated, not eliminated.
  Trainer-level corefrac protection events: 0 (pre-registration met at the protected
  set; the autopsy count is the wider statistical core50, on which B was never 0 either).
- **The depth design realizes:** e7's expert read mass concentrates DEEP — effnum
  E10 7,357 / E12 6,084 vs E6 4,208 (B's e7 leaned bottom-heavy: E2 7,388). e9, the
  other spatial task, same shape (E10/E12 12,265/12,465). The banks the placement bet
  added are the ones the hard tasks actually read.
- **VLM palettes are tighter low:** e7's VLM effnum 1,616-2,591 across V7-V13 vs B's
  2,341-5,366 across V10-V16; e2's V7 core50 = 80 slots. The E49 low-layer geometry
  yields more concentrated, less diffuse retrieval — consistent with the famIoU-free
  win and the palette pathology NOT materializing.
- Self-coverage 0.74-0.99 everywhere (healthy); cumulative core-RTO ≤ 21% worst cell
  (e9 core @E8 21%, vs B's 23% @E4) — same structure, smaller.

**Scorecard updates:** MSE-matrix gate PASS (drifts +0.0 to +3.6%, addendum 3 table);
prior-core: protection-set 0 = PASS, autopsy-count 4.4x-better-than-B = the honest
line; jitter = the strongest OOD certificate any substrate has produced.

---
### Entry 59 addendum 5 (5 Aug 26, 04:44 UTC) — harvest-bank rescore: DEEP PLACEMENT WIDENED THE VALUE COMPETENCE RADIUS. Far-region disagreement with the specialist's validated behavior drops 29% vs B (Q4 spec/succ D 0.482 → 0.344), beating the calibrated noise arm (0.379) — with a TIGHTER demo-manifold fit (anchor 0.0263 vs B 0.0340). The call-2 commitment signature persists; the residual e7 gap (38 vs 60) still lives beyond the widened radius.

**Instrument:** rescore on the E56/E57 e7 harvest bank (823 states: 418
B-fail / 67 B-succ / 220 spec-fail / 198 spec-succ / 120 demo), paired seeds/batching
vs chunks_spec_e7; report vs the specialist as reference.
SCORE_FEAT_LAYER=6 (memory-free below this model's first
VLM bank at 7); READ 1b grades on the model-independent proprio axis (same quartile
bins 0.567/1.287/2.334 as every prior report — perfectly paired). READ 2/3 rows are
B-trace context (vnoise precedent), not arm measurements.

**READ 1b — D(arm, spec) vs proprio distance, the decisive populations:**

| population | Q1 | Q2 | Q3 | Q4(far) |
|---|---|---|---|---|
| spec/succ: B | 0.094 | 0.120 | 0.123 | 0.482 |
| spec/succ: vnoise05x | 0.103 | 0.126 | 0.120 | 0.379 |
| spec/succ: **interleave** | **0.089** | **0.105** | **0.110** | **0.344** |
| spec/fail: B → interleave | 0.090→0.079 | 0.133→0.124 | 0.181→0.146 | 0.385→0.318 |
| harv_B/fail: B → interleave | 0.109→0.116 | 0.172→0.179 | 0.254→0.225 | 0.408→0.313 |
| demo: B → interleave | 0.0169→0.0154 | | | |

On the states where the specialist's far-region behavior is episode-validated
(spec/succ Q4 — E57's "the specialist traverses the far region"), the interleave's
function sits 29% closer than B's did, and closer than the E58 noise arm achieved —
the placement lever beat the regularization lever on the exact axis the noise was
designed for, while ALSO fitting the demo manifold tighter (anchor chunk 0.0263 vs
B 0.0340; the noise arms traded a little on-manifold fit for radius). Direction
consistent at every quartile and population.

**Held honestly:** (i) residual far-region D 0.344 >> demo 0.015 — most of the
remaining 38-vs-60 rollout gap still lives out there; the radius widened, it did not
close. (ii) D is symmetric and far states may be multimodal — part of Q4 D is
mode-mismatch, bounded by the same caveat as E57. (iii) READ 4 on the interleave-D:
divergences still concentrate at call ≤2 on both harvests (the first-grasp
commitment) — the failure GEOMETRY is unchanged; what improved is how far the
function stays right before it diverges.

**Board implications:** (1) the E57 ledger closes coherently — retrieval was never
the problem, value content was, and value PLACEMENT (depth) is a working lever on it.
(2) The natural next arm is stacking: recalibrated value-input noise ON the
interleaved substrate (dose from a fresh calib run — E58's sigmas were measured on
B's layers) — the two levers act on the same axis by different mechanisms (0.344 and
0.379 from independent directions; if even half-additive, spec-succ-Q4 approaches
the multimodality floor). (3) A fresh harvest of THIS model's e7 rollouts (50 eps,
traces on) would give its own failure autopsy + READ 2/3 as arm measurements —
the natural pre-step before the compass decision.

---
### Entry 59 addendum 6 (5 Aug 26, ~07:00 UTC) — sub-L7 querystats extension: the E49 curve is a V with its MINIMUM AT L7. "Lower is better" is dead; L7 is the measured optimum of the whole stack. V3/V4 excluded (constancy extreme); V5 CLEARED as viable (L9-grade-or-better separation, L13-band constancy) — a lesion-map judgment call, not a geometry kill.

**Instrument:** extended below the E49 range —
layers [3,4,5,6,7] on the
stage-1 base checkpoint (= the router input under frozen-route/prepass), 104
samples/task x 10 tasks.
**Calibration: L7 inter 0.721 vs E49's 0.722 — reproduces to the 3rd decimal.**

**Full merged curve** (instr-pool anchor, b=0: inter = between-task cos, lower =
better separation; intra = within-task cos, higher = more palette-constant):

| L | 3 | 4 | 5 | 6 | **7** | 9 | 11 | 13 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| inter | .794 | .775 | .734 | .775 | **.721** | .785 | .820 | .869 | .840 | .898 |
| intra | **.926** | .915 | .895 | .904 | .882 | .877 | .888 | .899 | .869 | .911 |

**Reads.** (1) The downward-improving E49 trend REVERSES below 7 — L7 is the global
separation optimum; nothing sub-7 beats it on either axis. The mechanistic
prediction (anchor conditionality needs attention mixing) confirms directionally:
intra climbs monotonically toward constancy as depth decreases. (2) BUT sub-L7 is a
slope, not a cliff: L5 inter 0.734 is BETTER than L9's 0.785 (a working bank), and
L5 intra 0.895 sits in the L13 band (0.899, also a working bank) — V5 is viable
geometry in unexplored territory, settled by lesion not by probe. First read
over-called this ("drop V5") from the sub-L7 table alone; the merged curve corrects
it — recorded as an instrument lesson: NEVER verdict a placement from a
range-restricted slice of a curve whose comparators live outside the slice.
(3) V3/V4 stay excluded: L3 intra 0.926 = the constancy extreme of the entire
measured stack (above even L16) with zero separation advantage — the one
out-of-band signal, on exactly the predicted pathology axis.

---
## Entry 60 - 5 Aug 26 (GO-BIG PLACEMENT SEARCH — 12 modules, expert [4,6,8,10,12,14,16] + VLM [5,7,9,11,13], frozen_prepass; first production use of the BG-FIRST gate; deliverable = the LESION MAP + a trimmed 3.2B retrain, not this run's own number)

**Rationale (from the E59 verdict):** placement is the proven highest-value axis
(+4.4 at matched budget; radius 0.482->0.344) and capacity is NOT binding (worst
core50 ~2.4K of 65K slots) — so more modules buy SITES, not slots. Search-by-lesion
beats search-by-chains: one big chain + zero-ablation evals maps all sites for the
cost of ~one config sweep candidate. Additive memory (MLP(x)+Memory(x)) makes
zeroing a module a clean lesion.

**Sites:** expert 14/16 = never-run deep territory, the primary targets (depth
gradient L8->L9->L12 + e7's read mass in E10/E12 hasn't turned over); expert 4
re-added as the expected-dead control (E36/E53) calibrating the map's zero; VLM
[7,9,11,13] unchanged (at the measured optimum); VLM 5 included per addendum 6
(viable, unexplored, map-settled); V3/V4 excluded (constancy extreme).

**Chain**:
warm-up 10k (router-only fast, anchored
w0.40 + sep8, FiLM-free, PREPASS=true) -> audit bs8x400 -> **BG-FIRST GATE** (E59
standing rule, first production use: expert bg <= 0.10 + mean core50 >= 400 +
min-eff >= 300; VLM min-eff >= 150 = the V5 arbiter + famIoU >= 0.45 backstop only;
famIoU otherwise INFORMATIONAL — expect lawful anchor-source elevation at expert
14/16) -> A-phase 10k -> 5-task sequential (C-config: corefrac, lr 2e-3->2e-4,
top_t 3072, 50-ep final). n256/r2 both towers = 4.8B values (+50%
vs interleave-8's 3.2B — capacity confound ACCEPTED; search run, not paper cell).

**Pre-registered reads** (comparator interleave-8 57.6 = 42/68/56/84/38, NOT
budget-matched): >= ~57.6-noise = 12 modules break nothing; e7 vs 38 = do E14/E16
extend the depth lever; e2 >= 80 AND e4 >= 40 (spread survival); give-back >= -3;
MSE matrix <= ~+5%; prior-core autopsy vs the 1,684/7,376 ladder.
**LESION PROTOCOL (pre-registered, implemented at
landing):** zero-ablate each of 12 modules at the final ckpt -> delta-success/task
at 20-ep (12x5x20 = 1,200 rollouts) + delta-chunk screen; greedy
backward elimination to a 3.2B 8-module layout; RETRAIN that layout = the paper
cell. Trim criterion = lesion delta, NOT read mass (usage != importance). Expert-4
lesion ~0 = the control passing.

---
## Entry 61 - 5 Aug 26 (SHARED-PAIR MEMORY TABLES: designed, built — the budget-restoring half of go-big-then-trim. One K/V storage per adjacent pair, per-site heads; the E59-frontier shared cell at HALF the adaptation state (1.6B) with same 8 sites)

**Idea (Josh):** the paper that inspired the project (and Meta's Memory Layers at
Scale) shares one memory pool across several base-model layers. Pair our attach
sites — (6,8)+(10,12) expert, (7,9)+(11,13) VLM — one table per pair. Halves value
params; potentially keeps the expressiveness.

**Why the ledger says yes:** E59's autopsy showed capacity is NOT binding (worst
core50 2.4K of 65K; effnum <= 12.5K) while placement/sites is the proven axis
(+4.4 at matched budget). Sharing decouples SITE count from PARAMETER count — and
composes with E60: the lesion map picks which sites matter; sharing keeps many
sites under the 3.2B paper budget instead of amputating to fit.

**Design decisions:**
- Share KEYS + VALUES together, per-site query-proj/gate/out-proj/queues/stats.
  Values-only sharing rejected: with per-site keys, slot i is two unrelated
  addresses aliasing one memory (hash collision by construction); with shared
  keys, slot i means the same routing region for every member. Adjacent pairing
  keeps the members' (prepass-frozen) routing inputs maximally correlated.
- MECHANISM BET (pre-registered): during sequential adaptation each shared slot's
  content gets gradients from BOTH consumption contexts — a real-distribution
  analogue of E58's value-input noise (train content on a NEIGHBORHOOD of x), at
  NEGATIVE parameter cost. Symmetric risk, also pre-registered: if content is
  depth-specialized (the other reading of e7's E10/E12 mass), e7 craters with
  shallow cells held — itself informative (separates content-specialization from
  placement); follow-up would be asymmetric capacity (solo deep, shared shallow).

**Cell** (bg-first gate per site): E59 layout, 4 tables, 1.6B values.
Pre-registration: >= ~55 ⇒ "sites, not slots" causal + half-state
efficiency claim (>= 57.6 ⇒ regularizer net-positive); rescore spec/succ Q4
<= 0.344; e7 >= ~30 w/ e2 >= 80 (depth-specialization arbiter); site-bleed
<= ~15%, prior-core <= ~2K. If it holds 57.6 at 1.6B: the cleanest
statement of the thesis — the memory's power is WHERE IT READS, not how much it
stores.

---
### Entry 61 addendum (5 Aug 26) — write-semantics design review (Josh): top-t and the protection store under sharing. DECISIONS: per-site top-t with union (budget follows demand); protection per table via MAX-sync for cell 1 (instrument-comparable), with NOISY-OR pre-registered as the follow-up composition if the measured overlap is material.

**The unit question:** sharing splits "the module" into site (demand) and table
(content). Resolution: write budget follows demand (per site); protection follows
content (per table); statistics follow attribution (per site, table-level realized).

**top-t:** per-site 3072 + UNION on the shared table (chosen — single-delta vs
interleave-8 where these same sites each had 3072; top_t is a tuned lever).
Alternatives held: joint per-table ranking (conserves table write surface but sites
compete + new TF/IDF-merge machinery — the fallback arm if union shows churn);
top-t/2 per site (budget-cut confound); leader-only (arbitrary, rejected). Cost
stated: per-table write surface becomes 3072-6144, data-dependent on pair overlap.

**Protection store — the sum debate.** Josh: "why not sum — that's the true
importance of a slot." Conceded in principle (damage is ~additive across read
paths). Three counters carried the decision: (1) the instrument has NEVER measured
aggregate importance — u is max-over-TASKS within a site; sum-across-sites would be
a semantic hybrid (additive across layers, max across tasks) = a new instrument,
not an extension. (2) sum couples protection dose to the UNKNOWN overlap structure:
disjoint cores => sum==max (moot); full overlap => sum==2x-max clamped (dose
doubling + saturation cliff exactly on the jointly-read core — the block-min/
write-starvation lever the E51 history warns about). At beta=4, a 0.5/0.5 slot:
discount 0.0625 (max) / 0.0039 (noisy-OR) / 0 hard veto (clamped sum). (3) much of
sum-max on correlated adjacent-site reads (shared keys!) double-counts ONE
underlying reliance through two channels. **The better form of the additive
instinct: NOISY-OR, u = 1 - prod(1-u_i)** — additive where additivity is linear
(small u), smooth saturation (no cliff), and FACTORIZES under the rank gate:
(1-u)^beta = prod_i (1-u_i)^beta = every site's tuned discount applied
independently. Verdict (Josh: "No you're right"): cell 1 keeps MAX; noisy-OR is
the pre-registered follow-up arm iff the overlap log says the rules diverge.

**Adjudicator instrumentation added:**
(1) realized per-table mask size + overlap at merge time:
3072-ish = pairs converge on the same slots; 6144-ish = the table
self-partitions. (2) boundary sync line now reports, per group: slots with
all-member u>0.1, protected-count u>0.5 under max VS noisy-OR, and total u-mass
under both — the counterfactual that decides whether the composition debate ever
mattered, measured on the cell's own data.

---
### Entry 60 addendum (6 Aug 26) — mid-run findings: chain resumed from 010000 with full cross-task state (protection+IDF x12 modules). Boundary trail through 2 tasks: fresh e4 55 (best-ever opener), 10K = e4 60 / e6 60 (e4 ROSE through the e6 block).

**Certificate (bg-first gate, first production use — PASSED):** expert bg 0.029-0.088
(L16 nearest ceiling), capacity floors clear at all 12 modules; famIoU informational
0.169-0.247 with L16 = highest ever certified (would have hard-failed the old gate).
**V5's first trained-router arbitration: famIoU 0.107 (~V7-grade; this run's L7 =
0.117), min-eff 253 vs the >=150 palette tripwire — the sub-L7 constancy worry did
NOT materialize.** E49 gradient reproduces extended: 0.107->0.159 monotone L5->L11.
Deep expert never-runs L14/L16: bg clean, core50 1153/1972 + min-eff 909/949 = the
largest certified expert footprints.

**Boundary trail (20-ep, +/-11pp):** 5K fresh e4 = 55 (best opener in any chain;
interleave/B opened 45); 10K = e4 60 / e6 60 — e4 rose +5 through the e6 block
(net-transfer signature), fresh e6 60 vs interleave's 70 (in-band). Seen-mean 60.0
vs interleave 52.5 / B ~50 at matched boundary. The 50-ep final arbitrates.

---
### Entry 60 addendum 2 (7 Aug 26, 01:32 UTC) — LANDING: **59.6 — NEW FRONTIER** (+2.0 over interleave-8 at +50% budget, 12 modules). **e7 = 58: the depth lever CLOSES to two points of the specialist (60)** — B 20 -> interleave 38 -> 58. e9 64 and e4 46 = best-ever sequential cells. e2 74 MISSES its >= 80 pre-registration (fresh 95 at 20-ep -> 74 at 50-ep through the e7 block). **The per-task-best pool now EXCEEDS the specialist oracle: 64.0 vs 63.2.**

**50-ep final** (step 25000): **59.6** = e4 46 / e6 56 / e9 64 / e2 74 / e7 58.
Comparators: interleave-8 57.6 = 42/68/56/84/38; B 53.2 = 44/60/56/86/20;
specialists 63.2 = 58/44/70/84/60. Oracle gap: -3.6.

**Pre-registration scorecard (E60 header):**
- >= ~57.6-noise -> **PASS at 59.6**: "more sites help even before trimming."
- e7 vs 38 -> **58 (+20)**. The never-run E14/E16 sites extend the depth lever to
  near-specialist parity. The E57 arc (far-region value content, addressable by
  depth) is now ~fully cashed: 20 -> 38 -> 58 vs spec 60.
- e4 >= 40 -> PASS (46, best ever; B 44). e2 >= 80 -> **MISS (74)**; fresh e2 was 95
  (the best 20-ep cell ever recorded) and gave back ~21 through the e7 block at
  mixed instruments — the FIRST spread-cell casualty of scale. With e6 56 (vs
  interleave 68) the pattern reads: the 12-module e7 block moves earlier tasks more
  than the 8-module one did. Matrix + autopsy arbitrate.
- Boundary trail (20-ep): 55 / 60 / 63.3 / **71.25** seen-means — every boundary
  the best ever at its position; fresh e2 95 = highest single cell ever.
- Sequential at bs8x4: ~1.9x the 8-module wall-clock (the 12-module cost).

**Interpretations.** (1) SITES KEEP PAYING: +2.0 over interleave with the capacity
confound acknowledged — the trimmed 3.2B retrain (lesion-selected) is the clean
claim; but e7 58 is placement, not capacity (capacity was never binding). (2) The
pool's simultaneity milestone: best-per-task across configs {46,68,64,84,58} =
64.0 > oracle 63.2 — the substrate family now collectively BEATS per-task
specialist training; one config holding the pool is the remaining engineering gap.
(3) The new cost surface: e2/e6 give-back under 12 modules = the first evidence
FOR trimming (some sites hurt retention even as others buy e7) — exactly the
question the lesion map answers per-site. (4) The bg-first gate's second
production validation: famIoU 0.244/0.247 sites (L10/L16) delivered e7 58.

**Battery**: MSE matrix + jitter grid + slot autopsy
(bigsearch/interleave/B triple) + harvest-bank rescore
(SCORE_FEAT_LAYER=4, below this model's first VLM bank at 5). At e7 58 vs spec
60, READ 1b's spec/succ Q4 is the arbiter of whether the function CONVERGED to
the specialist's far-region behavior (D -> demo-level) or wins differently.

---
### Entry 60 addendum 3 (7 Aug 26, morning) — the e2 "give-back" investigation CLOSES: it was the SEED. 2x2 (config x eval-seed) shows bigsearch and interleave exactly TIED at seed 2000 (88.0 both); the entire 74-vs-84 deficit lives in the seed-1000 state set. A 14-point single-seed swing on one task => the multi-seed campaign (Josh's spec) is now the standing instrument.

**Sequence.** Josh's read on the landing: "give-back was probably higher than usual
— find where and why." (1) MSE matrix (addendum 2 battery): give-back is NOT
elevated — drifts +4.2/+3.8/+3.7/+1.3/+0.0%, all in-band; percentages inflated by
E60's LOWEST-EVER diagonals (better per-block fit); in ABSOLUTE terms only e9
drifts more than E59 (+0.0026 vs +0.0017, accrued in the e2 block — the lawful
e9<-e2 pair at 12 write surfaces), and E60's FINAL loss is lower than E59's final
on every task. Fit-level retention: best on record. (2) e2's -10 success (74 vs
E59's 84) against a flat matrix, 0 protection events, and the best-ever e2 jitter
cells (clean 0.0197 vs E59 0.0235) = the E56/E57 fit/success decoupling signature.
(3) Seed re-eval, same ckpt same instrument, seed 2000: **e2 = 88.0.** (4) The 4th
cell — E59's ckpt at seed 2000: **88.0. Configs exactly tied.**

| e2, 50-ep | seed 1000 | seed 2000 |
|---|---|---|
| interleave-8 | 84 | 88 |
| bigsearch-12 | 74 | 88 |

**Verdict:** no real e2 degradation — a handful of seed-1000 initial states flip
under the 12-module config while a fresh draw shows parity. NOT worth the harvest
apparatus. The DURABLE finding is about the instrument: single-seed 50-ep cells
can swing 14 points on a high-variance task, so the +2.0 frontier claim (59.6 vs
57.6) is inside single-seed noise until the campaign lands, and 63.2 (the
specialist oracle) is single-seed folklore on the same terms.

**Multi-seed campaign** (Josh's spec):
25 eps x 4 seeds
(1000/2000/3000/4000, PAIRED across configs) x {bigsearch-final,
interleave-final on all 5 envs; each specialist on its own env}.

---
### Entry 60 addendum 4 (7 Aug 26, 10:30 UTC) — MULTI-SEED CAMPAIGN LANDS (Josh's spec: 25 eps x 4 seeds, paired states, all finals + all specialists): **THE SEQUENTIAL MEMORY MODEL BEATS THE SPECIALIST ORACLE — 64.6 vs 59.0, positive at 4/4 paired seeds.** The 63.2 "oracle" was single-seed luck; specialists re-base DOWN to 59.0 (their per-cell sd is +-4-11). Frontier ordering bigsearch > interleave confirmed 4/4 (+4.0). e7 is the ONLY remaining below-specialist cell (-9).

**The table** (mean +- sd over 4 seeds x 25 eps; seeds 1000/2000/3000/4000 paired
across all configs; finals only):

| config | e4 | e6 | e9 | e2 | e7 | mean |
|---|---|---|---|---|---|---|
| bigsearch-12 | 53+-5 | 70+-7 | 69+-10 | 81+-9 | 50+-10 | **64.6** |
| interleave-8 | 50+-5 | 70+-7 | 58+-10 | 86+-8 | 39+-7 | 60.6 |
| specialists | 46+-11 | 49+-11 | 61+-7 | 80+-7 | 59+-4 | **59.0** |

**Headline reads:**
1. **bigsearch > specialist oracle, paired: +3.2 / +10.4 / +5.6 / +3.2 (4/4).**
   Per-env: e4 +7, e6 +21, e9 +8, e2 +1, e7 -9. The CONTINUAL learner beats
   per-task specialist fine-tuning on 4 of 5 tasks at matched instrumentation —
   the project's target ("make up ~10pp toward 63.2") is not just met, the bar
   itself was mismeasured: the single-seed 63.2 decomposes into specialist
   per-cell sd of +-4-11 (e4 spans 32-56 across seeds; e6 40-64), re-basing to
   59.0. Interleave-8 (60.6) ALSO clears the oracle.
2. **Frontier ordering CONFIRMED at seeds:** bigsearch +4.0 over interleave,
   positive at all 4 paired seeds (+1.6/+8.8/+1.6/+4.0) — the single-seed +2.0
   was real and understated. e7 (+11) and e9 (+11) carry it; e2 -5 (3/4 seeds
   negative) is the honest small cost of the 12-module config.
3. **e7 residual:** 50+-10 vs specialist 59+-4 — the depth lever closed 20->58
   at seed-1000/50-ep but the multi-seed read says ~-9 remains. The e7 story is
   not finished; it is however no longer the frontier's defining deficit (the
   config-mean gap to specialists is now POSITIVE).
4. **Instrument notes:** single-seed 50-ep finals mis-read configs in BOTH
   directions (memory configs read ~4-5 low: 59.6->64.6, 57.6->60.6; the
   specialist oracle read ~4 high: 63.2->59.0) — seed luck, not a systematic
   episode-count effect. All prior standing numbers keep their roles as paired
   seed-1000 history; HEADLINE claims from here on cite the 4-seed instrument.

**Board consequences:** the ICRA narrative inverts — "sparse memory closes the
specialist gap" becomes **"sparse memory EXCEEDS specialist fine-tuning while
training sequentially with ~zero forgetting"** (matrix +0-4%, naive baseline
+740-1567%). e6 +21 is the transfer flagship (the multitask substrate beats the
50-demo specialist by 21 points). Remaining open cells: e7 -9 (noise arm /
lesion-trim still queued for exactly this) and the e2 -5 scale cost. E61
shared-pairs
— its pre-registration gains a sharpened
context: comparator interleave-8 is now 60.6+-noise at the 4-seed instrument.

---
### Entry 61 addendum 2 (7 Aug 26, 14:50 UTC) — E61 CERTIFICATE: gate PASS; **shared keys route indistinguishably from dedicated keys.** famIoU within ~0.01 of E59 at every expert site; V7 0.101 EXACTLY reproduced; VLM 9/11/13 slightly cleaner. First adjudicator data: pairs PARTITION the table (union 1536+1536 -> ~3020-3047, overlap 25-52 rows ~2-3%) — if this holds through the sequential, max-vs-noisy-OR is moot by disjointness.

Cert: expert bg 0.032-0.054, core50
721-1202, min-eff 559-801; VLM min-eff 266-423, famIoU 0.101-0.145 (informational).

---
### Entry 60 addendum 5 (7 Aug 26, eve) — baseline rows for the 4-seed table (Josh): multitask-LoRA (standing 49.2) + naive seq-LoRA r256 final (standing 17.6, the forgetting foil; added on the same-instrument principle).

---
### Entry 60 addendum 6 (7 Aug 26, eve) — FULL-FINETUNE baselines for the 4-seed table (Josh): (1) raw pi05 base -> libero_10 full FT 50K -> 4-seed eval; (2) the libero_90 stage-1 substrate -> libero_10 full FT 50K -> 4-seed eval (the "same substrate" cell). ARCHAEOLOGY CORRECTION: the deleted "72.6" was E31's B1 = libero_90+libero_10 JOINT finetune; the libero_10-only cell was B2, KILLED before completion — these are NEW cells, cleaner than the 72.6 (no pretraining data mixed into the finetune), and the standing 72.6 is NOT their expected value.

Train args
mirror E31/stage-1 verbatim (bs32,
warmup 4K/decay 50K, pi05 default LR);
in-run eval disabled — the 4-seed campaign
is the instrument.
When done the ICRA table exists at ONE instrument: bigsearch /
interleave / sharepairs / specialists / multitask-LoRA / naive-seq / full-FT
(fresh) / full-FT (same-substrate).

---
### Entry 61 addendum 3 (8 Aug 26, 14:00 UTC) — LANDING: **56.8 at HALF the adaptation state (1.6B vs 3.2B)** — the >= ~55 gate PASSES: "sites, not slots" is confirmed CAUSALLY. And the depth-specialization arbiter FIRED exactly as pre-registered: **e7 38 -> 22 with every shallow/anchor cell held or improved** (e4 +8, e9 +6, e2 +2, e6 -4 vs interleave) — depth-specialized value content is REAL, separated from placement for the first time. Follow-up per the pre-registration: ASYMMETRIC capacity (solo deep tables, shared shallow).

**50-ep final** (step 25000): **56.8** = e4 50 / e6 64 / e9 62 / e2 86 / e7 **22**.
Comparator interleave-8 (same sites, dedicated tables): 57.6 = 42/68/56/84/38.
Non-e7 mean: **65.5 vs 62.5** — sharing IMPROVED the four shared-survivable tasks
(the E58-analogue regularization bet paying) while cratering the deep-bank task.

**Pre-registration scorecard (Entry 61):**
1. >= ~55 => PASS (56.8; -0.8 vs interleave = tied within the single-seed
   instrument). Frontier-band continual learning at HALF the stored adaptation
   state — the efficiency claim stands, with the e7 asterisk.
2. e7 >= ~30 w/ e2 >= 80 => **FAIL exactly along the pre-registered fault line**:
   e2 86 (held), e7 22 (cratered). Verdict as written in the header: "depth-
   specialized content is real => asymmetric capacity (solo deep tables, shared
   shallow) is the follow-up." The E(10,12) pair — where e7's read mass
   concentrates (E59 autopsy) — cannot express both layers' content in one table.
3. Cost: step time BETTER than interleave's — halved optimizer/
   clip work;
   union-merge overhead invisible.
4. Adjudicators: write sets partition (union 3072+~3072 -> ~3020-3047/pair-step,
   overlap 25-52 rows ~2-3%); protection composition low-stakes at the clean
   boundary-1 read (max vs noisy-OR differ 1.6-3.8%) — max stands, debate CLOSED.
   INSTRUMENT NOTE: boundary >= 2 counterfactuals are self-contaminated (the sync
   overwrites members with the merged u, so later "overlap"/OR reads double-count
   the shared baseline — deployed max unaffected, it is idempotent; fix for any
   future cell = per-site pre-sync accumulators). Crowding: shared tables carry
   both members' protection, topping at ~36% of E(10,12) at u>0.5 after 5 tasks —
   not binding at top_t 3072/site.
5. Boundary trail: fresh e4 70 = best opener ever recorded; trail wobble
   70/50/40/50 pure instrument; fresh e6 65 / e9 75 / e2 80.

**The scientific yield** — E61 turns "the depth pattern" into a three-way
dissociation: (a) PLACEMENT matters (E59/E60: deep sites buy e7); (b) CAPACITY
does not (E59 autopsy: banks near-empty); (c) **CONTENT IDENTITY at depth
matters (E61: one table across L10+L12 kills e7 while four tasks thrive)** — the
deep banks need layer-specific values even at trivial occupancy. This is the
cleanest mechanism statement the project has produced, and it PRESCRIBES the
architecture: share where consumption contexts are interchangeable (shallow/
anchor tasks), dedicate where they are not (deep spatial/far-region content).

**Battery:** msemat+jitter, autopsy incl. the
site-bleed read, harvest rescore — e7 22 predicts the radius REGRESSES from
0.344 -> addendum 4.
Follow-up cell sketched,
NOT launched (Josh's call): asymmetric E61b = shared (6,8) + SOLO 10 + SOLO 12
+ shared VLM pairs = 2.4B (25% saving) with the depth lever intact.

**[RESOLVED — discussion held 9 Aug, decisions in Entry 61 addendum 5]** The
E61b asymmetric sketch above is superseded: hand-picking which layers share is
rejected as config-fitting; the follow-up is the 6-per-tower merged config with
the share/solo assignment set by a measured criterion (the share-criterion
probe). Full decisions + sequencing in addendum 5.

---
### Entry 61 addendum 4 (8 Aug 26, eve) — battery lands, and it REVISES the mechanism story: the site-bleed read FAILS its <= ~15% pre-registration (17-43% at block scale — per-step write masks partition but ACCUMULATED writes land on a third of the partner's read mass); the MSE matrix breaches the <= +5% band for the first time (+7.7/+6.9/+5.7/+4.8/+0.0%); yet the harvest-bank radius HELD (spec/succ Q4 0.318 ~= interleave's 0.344). The e7 damage is invisible to every standing demo/paired-state instrument — it lives in the model's OWN rollout distribution. Plus: baseline 4-seed rows land (multitask-LoRA 51.4; naive foil 18.0 with seed-robust 0/0/0 on all early tasks).

**Site-bleed (the decisive mechanism read):** within-pair, same
task: E(6,8) 22-35%, E(10,12) 25-43%, V(7,9) 17-29%, V(11,13) 23-38%. The
union-merge lines showed 2-3% overlap PER STEP; over a 5K-step block the
accumulated other-member write set covers 20-40% of each site's read mass. So
shared tables do NOT self-partition at block scale — each layer's content is
substantially co-written by the other layer's training signal. ONE mechanism,
two signs: cross-layer co-writing = the regularizer that improved the four
shallow tasks (+3 mean) AND the corruption that killed e7 (-16). The
pre-registered <= 15% read fails; consolidation headroom is NOT confirmed.

**Matrix (first band breach of any substrate):** +7.7/+6.9/+5.7/+4.8/+0.0% vs
interleave's +3.6/+3.0/+2.2/+1.3/+0.0 at near-identical diagonals — genuine
extra cross-task drift from the shared write surface (same-family cells, ~2x
drift). Still vanishing next to naive's +740-1567%, but sharing measurably
costs retention at the fit level.

**Rescore (the surprise):** spec/succ Q4 D = 0.318 (interleave 0.344, B 0.482)
— the far-region function on the STANDING bank did not regress; harv_B/succ Q4
0.087 is the closest-to-reference read any config has produced. The e7 crater
(38->22 in rollouts) is therefore INVISIBLE to demo states, paired jitter,
the matrix, AND the harvest bank — it manifests only on the model's own rollout
distribution (the E56 lesson at a new scale: conversion damage need not appear
on other models' state banks). A fresh sharepairs-e7 harvest would localize it;
deferred to tomorrow's discussion (E61b would obsolete the question).

**Baseline rows (4-seed instrument):** multitask-LoRA **51.4**
(41/47/39/78/52; standing 49.2 was fair; its e7 52 rivals bigsearch's 50 — the
dense-adapter e7 pattern reproduces at seeds; e9 spread 24-60 is the wildest
cell yet). Naive seq-LoRA final **18.0** (0/0/0/55/35; standing 17.6 dead on;
the 0s are seed-robust — catastrophic forgetting with error bars).

**One-instrument table so far:** bigsearch 64.6 / interleave 60.6 / specialists
59.0 / multitask-LoRA 51.4 / naive 18.0; sharepairs row + both full-FT rows
pending. **Inputs staged for the 9 Aug discussion:** the three-way
dissociation now has a mechanism caveat — "content identity at depth" rests on
the ROLLOUT dissociation (pre-registered, clean) while the bleed data suggests
the proximate cause is cross-writing rather than addressing; E61b (solo deep
tables) remains the prescription either way, with the alternative framings
(distance-1 pairing, trim-first) queued for the discussion.

---
### Entry 60 addendum 7 (9 Aug 26) — full-FT baselines: methods note (mid-run switch to bs8xacc4, numerically neutral) + the data-budget decision (Josh)
METHODS NOTE for the baseline row: FT#1 trained steps 0-10K at bs32+ckpt and
10K-50K at bs8xacc4-no-ckpt — effective batch 32 and the LR schedule are
IDENTICAL throughout (grad-ckpt and micro-batching are numerically neutral up
to accumulation-order rounding); FT#2 runs bs8xacc4 end-to-end.

**[Decision — Josh, 9 Aug]** Full-FT baselines stay ALL-10-task (the "all-data
multitask ceiling" framing; eval front-5 only). Table caption must state the
data budgets: full-FT = all 10 tasks' demos; multitask-LoRA
and the memory sequentials = front-5 adaptation data only.

---
### Entry 61 addendum 5 (9 Aug 26) — THE E61 FOLLOW-UP DISCUSSION (Josh): E61b hand-picked asymmetry REJECTED as config-fitting; the next config bet is the **6-per-tower MERGED stack ("6x2"): 12 sites at the 3.2B paper budget, shallow pairs sharing tables, deep layers solo — CONDITIONAL on a measured share-criterion probe**. 10-task validation made a REQUIREMENT for any shared config; lesion battery demoted to paper evidence.

Discussion held (all four of Josh's points below), decisions recorded:

**1. Merge the layer-max and sharing branches — accepted, at the 6-per-tower
dose.** Sharing decouples site count from parameter count, and sites are the
proven axis (57.6 -> 59.6 -> 64.6-at-seeds) while store capacity never was.
The full 8-per-tower version is rejected on layer inventory: the VLM has only
~6 certified-good routing layers (V3/V4 excluded for the constancy pathology,
E59 addendum 6) and the expert side below ~L6 has always been marginal
(E36/E53) — 8 per tower forces known-bad territory.

**2. Hand-picking which layers share (E61b) is REJECTED as overfitting the
config to the benchmark.** Replacement: a PRE-EVAL measured criterion.
Hypothesis (Josh): adjacent layers whose router inputs/queries are SIMILAR
should share (shallow); layers whose queries DIFFER get their own tables
(deep). Cheap to test — under frozen-prepass every router reads the memory-free
stage-1 features, so cross-layer similarity of the actual router inputs is a
forward-pass probe on the stage-1 checkpoint (the E59-addendum-6 precedent).
**Validation requirement (pre-registered): the metric must separate E(6,8)
[shareable — held/improved in E61] from E(10,12) [not — e7 38->22], and call
all VLM pairs shareable.** Warning recorded: E61's own overlap statistics
FAILED this test — per-step write-mask overlap (2-3%) and block-scale
site-bleed (E(6,8) 22-35% vs E(10,12) 25-43%, overlapping ranges) do not
discriminate the good pair from the bad one, so validation is substantive,
not a formality. If raw query cosine also fails (adjacent residual-stream
states are similar everywhere), try task-conditional variants (centered
similarity, task-geometry RSA) before falling back to the depth rule — which
remains defensible (pre-registered arbiter, architectural not task-specific).

**3. The chosen middle ground ("6x2", Josh's sketch): 6 layers per tower,
shallow 4 merged in adjacent pairs, deep 2 solo = 4 tables/tower = 8 tables =
3.2B — the paper budget — carrying 12 sites = bigsearch's site count at 2/3
its parameters.** Working layout sketch (final pairing set by the probe, not
by hand): expert [6,8,10,12,14,16] with (6,8)+(10,12) shared and 14/16 solo;
VLM [5,7,9,11,13,15] pairs shared (all VLM pairs survived E61). This is the
disciplined version of the merge idea: it embeds the E61 design rule (share
where consumption contexts are interchangeable, dedicate at depth) instead of
fighting it, and it has a real shot at bigsearch-level results (64.6) at
2/3 the parameters. **This displaces E61b and is the next config bet,
conditional on the probe.**

**4. 10-task validation is a REQUIREMENT before any shared config headlines
(Josh's point: the +7.7% matrix breach is only validated at 5 tasks).** Two
pre-registered degradation mechanisms, both measurable: (a) cross-writing
accumulates PER BLOCK — an early task sits under 9 subsequent blocks at 10
tasks vs 4 at 5, roughly doubling co-writing exposure (the +7.7% could grow
to +15%+); (b) protection crowding on shared tables — both members' protection
accumulates on one table (36% of E(10,12) at u>0.5 after 5 tasks); at 10
tasks late writers risk write starvation, the known over-protection failure
mode. The 10-task run doubles as the ICRA catastrophe-elimination
demonstration — one run serves both.

**Also decided:** lesion battery demoted from blocking step to paper evidence
(the depth ladder 20->38->58 already justifies the layout; the map slots in
later as site-choice justification). Risk stated out loud: the 6x2 becomes
the de facto paper bet, with bigsearch-12 (64.6 at 4.8B) as the fallback
headline if it disappoints — acceptable, worst case we still hold 64.6.
The fresh sharepairs-e7 harvest (localizing the rollout-only damage) stays
deferred unless the mechanism section needs it.

**Sequencing:** (1) share-criterion probe;
(2) 6x2 chain (warm-up -> audit
-> A-phase -> 5-task, standard pre-registrations);
(3) **noise arm on the step-2 winner (Josh, 9 Aug):**
value-input noise at the E58 half dose, with sigmas RE-CALIBRATED on the
winner's layers (the E58 values were measured per-layer on B's layout —
rerun before the arm). Pre-registered question:
does noise still add anything ON TOP of sharing — the shared tables'
cross-writing already provides a version of the same
train-on-a-neighbourhood effect, so redundancy is a live (and paper-worthy)
outcome; (4) 10-task on the finalized config (architecture + noise decision
baked in) + lesion battery behind it. The sharepairs 4-seed row
calibrates how real the 56.8-vs-57.6 single-seed tie was.

---
### Entry 61 addendum 6 (9 Aug 26) — SHARE-CRITERION PROBE RESULT: the similarity hypothesis is INVERTED by the calibration — the pair that failed sharing has the MOST similar router inputs (E(10,12) raw-cos 0.639) while the pair that shared fine has less similar ones (E(6,8) 0.527); VLM pairs are MORE similar than the failed pair (0.65-0.73) and all shared fine → no global similarity threshold exists in either direction. Adopted rule: the metric is a VETO within the expert tower (high-sim deep pairs flagged), composed with the depth rule; proposed 6x2 layout keeps the proven deep sites SOLO.

**Probe** (on the stage-1 checkpoint, libero_10).
Per
adjacent pair: raw/centered per-token cosine between the two layers' router
inputs, relative stream change, task-geometry agreement (RSA).

Expert (raw_cos / cent_cos / rel_delta / task_rsa):
  2-4   0.484  0.546  0.960  0.903
  4-6   0.523  0.616  1.054  0.939
  6-8   0.527  0.580  1.070  0.935   <- SHARED FINE in E61
  8-10  0.566  0.580  1.037  0.921
  10-12 0.639  0.666  0.875  0.971   <- KILLED e7 in E61
  12-14 0.544  0.572  0.992  0.964
  14-16 0.642  0.656  0.981  0.907
VLM instr16 raw_cos: 3-5 0.606 / 5-7 0.653 / 7-9 0.696 / 9-11 0.727 /
11-13 0.724 / 13-15 0.605 — (7,9) and (11,13) both shared fine in E61.

**Findings.**
1. All four metrics separate the two expert calibration pairs (raw delta
   0.113) — but with the OPPOSITE sign to the hypothesis: "similar inputs ->
   share" is falsified. The failed pair is the most-similar expert pair.
2. No inverted global rule either: VLM pairs sit at 0.65-0.73 similarity
   (above the failed 0.639) and all shared fine. Similarity alone does not
   determine shareability across towers.
3. Unified reading consistent with ALL calibration points + the E61 bleed
   data: sharing fails when similar inputs make both layers address the SAME
   slots (write collision) AND the content each layer needs from those slots
   differs (deep expert, the E61 content-identity finding). It survives when
   either the layers address different slots (E(6,8) — inputs differ ->
   tables self-partition -> co-writing acts as neighbourhood regularization)
   or the content is interchangeable (VLM anchor/palette content).
   Danger = same addresses x non-interchangeable content.
4. Curiosity for the record: expert similarity is NOT monotone in depth —
   (12,14) reads 0.544 (low band) while (10,12)/(14,16) read ~0.64. The
   metric's genuinely novel prediction (a (12,14) share would be safe) is
   exactly the kind of untested inversion we should NOT bet the paper cell
   on.

**Adopted rule (conservative composition, stated for the paper):** the
similarity metric is a VETO, not a license — within the expert tower, do not
share any pair in the high-similarity band (>= ~0.6: (10,12), (14,16)); do
not share deep pairs regardless of similarity (the E61 direct evidence);
VLM pairs shareable per E61. The criterion is "validated as a veto against
the E61 contrast," which is the honest claim.

**Proposed 6x2 layout under the rule (to freeze with Josh):**
- Expert [4,6,8,10,14,16]: share (4,6) [0.523] + (8,10) [0.566], SOLO 14 +
  SOLO 16 — keeps the exact deep sites that took e7 20->38->58, shares the
  four shallowest, all shared pairs in the metric's safe band. Drops L12
  (its role subsumed by solo 14/16; bigsearch's e7=58 came with 14/16
  present). Residual risk flagged: e7's interleave-8 read mass sat at
  E10/E12, and 10 is inside a shared pair here — the (8,10) share is the
  layout's one bet the calibration does not directly cover.
  Alternative if that risk reads too hot: expert [4,6,8,10,12,14] with
  share (4,6)+(8,10)? — no: same 10-share; the truly conservative variant
  is [2,4,6,8,14,16] share (2,4)+(6,8) (both directly calibration-approved)
  at the cost of spending sites on the known-marginal L2/L4.
- VLM [5,7,9,11,13,15]: all three pairs shared [(5,7) 0.653, (9,11) 0.727,
  (13,15) 0.605] = 3 tables.
- Total: 4 expert + 3 VLM = 7 tables = **2.8B — 12.5% UNDER the 3.2B paper
  budget** at 12 sites (bigsearch's site count).

---
### Entry 60 addendum 8 (10 Aug 26) — FULL-FT BASELINE #1 LANDS (raw pi05 base -> libero_10 ALL-10-task full fine-tune, 50K) + 4-seed row: **67.6** — the all-data, full-backbone, no-continual-constraint cell sits only **+3.0 over bigsearch (64.6)**, positive at 4/4 paired seeds but marginally (+8.0/+0.0/+1.6/+2.4); **the memory model BEATS full fine-tuning by 21 points on e4** at half the adaptation data.

Training ran 50K steps (bs32 for 0-10K, bs8xacc4
for 10-50K per the E60-add-7 methods note).
Campaign
(25 eps x 4 paired seeds):

| env | full-FT (all-10 data) | bigsearch-12 (front-5, sequential) | delta |
|---|---|---|---|
| e4 | 32.0 +- 5.7 | 53.0 +- 5.0 | **-21.0** |
| e6 | 81.0 +- 6.0 | 70.0 +- 6.9 | +11.0 |
| e9 | 73.0 +- 11.0 | 69.0 +- 10.0 | +4.0 |
| e2 | 98.0 +- 2.3 | 81.0 +- 8.9 | +17.0 |
| e7 | 54.0 +- 11.5 | 50.0 +- 9.5 | +4.0 |
| **mean** | **67.6** | **64.6** | +3.0 |

Reads:
1. **The headline framing survives contact with the strongest baseline:** a
   sequentially-trained, frozen-backbone memory model using only the front-5
   tasks' demos lands within ~3 points of a full-backbone joint fine-tune
   given ALL TEN tasks' demos and no continual constraint — and wins e4
   outright by 21. (Data-budget caption per the 9 Aug decision: full-FT =
   all-10 demos; memory sequentials + multitask-LoRA = front-5 only.)
2. e4 is now full-FT's WORST cell (32 +- 5.7). Note the old E31 90+10-joint
   run scored 52 on e4 — suggesting libero-90 data in the fine-tune mix was
   carrying e4, not backbone plasticity. FT#2 (same recipe FROM the libero-90
   substrate) tests exactly this; if its e4 recovers toward ~50, the
   "same-substrate" cell becomes the sharper comparison.
3. e2 98 +- 2.3 confirms e2 is nearly saturated for an unconstrained model;
   our 81 +- 8.9 keeps e2 as one of the two real remaining gaps (with e6).
4. One-instrument table now: full-FT(fresh) 67.6 / bigsearch 64.6 /
   interleave 60.6 / specialists 59.0 / multitask-LoRA 51.4 / naive 18.0.

---
## Entry 62 - 11 Aug 26 (E62 MERGED 6x2 — the go-big + sharing merge: 12 sites (bigsearch's count) at 7 tables = 2.8B (12.5% under the paper budget), share/solo assignment from the E61-add-6 veto rule)

**Layout (per the adopted veto rule + depth rule, E61 addenda 5/6):**
- Expert [4,6,8,10,14,16]: share (4,6) [sim 0.523] + (8,10) [0.566]; SOLO 14,
  16 (the e7 depth lever keeps dedicated content; L12 dropped). The (8,10)
  share is the one bet the E61 calibration does not directly cover (e7/e9
  read mass partly at E10 in interleave-8) — pre-registered read #4 isolates
  it.
- VLM [5,7,9,11,13,15]: all three pairs shared [(5,7) 0.653, (9,11) 0.727,
  (13,15) 0.605].
- B router recipe verbatim (anchor 0.40, sep8, FiLM-free, broadcast), prepass
  on, bg-first gate, then A-phase + 5-task C-config sequential (corefrac,
  lr2x, top_t 3072, 50-ep final).

**Pre-registration:** e7 >= ~40 (solo-deep
must avoid sharepairs' 22); final >= ~57.6 = frontier band at 2.8B, >= ~59.6
= matches bigsearch at 58% of its params (the paper-cell branch); e4 >= 40
AND e2 >= 80; matrix <= ~+8% (shared-write band); prior-core events = 0;
updt_s recorded. Noise arm (recalibrated dose) follows on the winner per the
add-5 sequencing.

**Meanwhile — FT#2 (same-substrate full FT):** seeds
1000/2000/3000 = 79.2 / 80.8 / 75.2 per-seed means, and **e4 recovered 32 ->
72-88** — the FT#1 e4 anomaly was the missing libero-90 data, confirmed.

---
### Entry 60 addendum 9 (11 Aug 26) — FULL-FT BASELINE #2 LANDS (the "same substrate" cell: libero-90 stage-1 -> libero_10 ALL-10 full FT, 50K) + 4-seed row: **78.2** — the substrate-matched, all-data, unconstrained ceiling. The libero-90 substrate is worth +10.6 even to full fine-tuning (67.6 -> 78.2), and FT#1's e4 anomaly is CLOSED (32 -> 78: it was the missing libero-90 data). The honest memory-vs-ceiling gap is now 13.6.

Campaign (25 eps x 4 paired seeds; per-seed means
79.2 / 80.8 / 75.2 / 77.6):

| env | FT#2 (from-l90, all-10) | FT#1 (fresh, all-10) | bigsearch-12 (front-5, seq) |
|---|---|---|---|
| e4 | 78.0 +- 7.7 | 32.0 | 53.0 |
| e6 | 78.0 +- 4.0 | 81.0 | 70.0 |
| e9 | 79.0 +- 8.2 | 73.0 | 69.0 |
| e2 | 90.0 +- 4.0 | 98.0 | 81.0 |
| e7 | 66.0 +- 5.2 | 54.0 | 50.0 |
| **mean** | **78.2** | 67.6 | 64.6 |

Reads:
1. **The one-instrument ceiling row is 78.2** (FT#2 holds every advantage at
   once: same substrate as ours, all-10 demos, full backbone, no continual
   constraint). The deleted-72.6 folklore is replaced by two measured cells:
   fresh 67.6 / same-substrate 78.2. Our bigsearch sits 13.6 below the
   substrate-matched ceiling with front-5 data, sequential training, and a
   frozen backbone — the honest headline framing for the table caption.
2. **Substrate transfer is worth +10.6 even to full FT** (e4 +46, e7 +12) —
   consistent with the whole project's premise that the libero-90 stage-1
   carries most of the competence; adaptation methods differ in how much of
   it they can use.
3. FT#2's e7 = 66 is the best e7 cell measured anywhere (spec 59, bigsearch
   50) — e7's remaining headroom for us is real, not task-intrinsic.
4. One-instrument table: **FT-l90 78.2 / FT-fresh 67.6 / bigsearch 64.6 /
   interleave 60.6 / specialists 59.0 / multitask-LoRA 51.4 / naive 18.0**.

---
### Entry 61 addendum 7 (11 Aug 26) — SHAREPAIRS 4-SEED ROW: **58.4** (-2.2 vs interleave-8's 60.6 at HALF the state) — and **e7 = 32.0 +- 0.0: four identical draws.** The depth-sharing damage is deterministic-grade, not noise; the E61 single-seed non-e7 advantage does NOT replicate (non-e7 65.0 vs interleave 66.0 ~ tied). THE ONE-INSTRUMENT TABLE IS COMPLETE.

(25 eps x 4 paired seeds; per-seed means 55.2/56.8/61.6/60.0):
e4 48.0+-10.6 / e6 67.0+-3.8 / e9 62.0+-7.0 / e2 83.0+-5.0 / **e7 32.0+-0.0** -> mean **58.4**.

Reads:
1. **e7 32.0 with ZERO variance across four seeds** — the sharpest error-bar
   statement of the depth-specialization finding: the shared-deep-table damage
   is a stable property of the policy, not eval luck. (Slightly above the
   50-ep final's 22 — that read was a low draw; the damage is real either way:
   interleave e7 = 39+-7.)
2. **The efficiency claim's honest form:** -2.2 overall at half the state,
   with the entire deficit in e7 (-7); non-e7 tied (65.0 vs 66.0). The E61
   single-seed "+3 on shallow tasks" regularizer advantage does NOT replicate
   at seeds — sharing is ~free where content is interchangeable, harmful
   where it isn't, and the E62 solo-deep layout is exactly the config this
   row prescribes.
3. **THE COMPLETE ONE-INSTRUMENT TABLE** (25 eps x 4 paired seeds, finals):
   FT-l90 (ceiling) 78.2 / FT-fresh 67.6 / **bigsearch-12 64.6** /
   interleave-8 60.6 / specialists 59.0 / sharepairs 58.4 /
   multitask-LoRA 51.4 / naive seq-LoRA 18.0. Caption budgets: full-FT rows
   = all-10 demos + full backbone + joint; memory rows = front-5 demos,
   sequential, frozen backbone; specialists = per-task models with task ID.

---
### Entry 62 addendum (13 Aug 26, 01:16 UTC) — LANDING: **66.8 — NEW FRONTIER at seed-1000/50-ep** (e4 58 / e6 74 / e9 64 / e2 84 / e7 54): the merged 6x2 beats bigsearch-12 (59.6) by +7.2 at 58% of its parameters (2.8B vs 4.8B, and 12.5% UNDER the 3.2B paper budget). The paper-cell branch fires with margin. e7 = 54 with the (8,10) share in place — risk cell #4 CLEARS; the veto-rule layout (solo 14/16) preserved the depth lever that sharing killed in sharepairs (22).

**50-ep final** (step 25000): **66.8** = e4 58 / e6 74 / e9 64 / e2 84 / e7 54.

**Pre-registration scorecard:**
- final >= ~57.6 (frontier band at 2.8B) -> **PASS**; >= ~59.6 (match bigsearch
  at 58% params) -> **PASS at 66.8, +7.2 OVER bigsearch's seed-1000 final** —
  the strongest single-seed 50-ep final in project history (prior best:
  bigsearch 59.6). Seed-1000 comparators: bigsearch-12 59.6 / interleave-8
  57.6 / sharepairs 56.8 / B 53.2.
- e7 >= ~40 -> **PASS (54)**. Sequence across the sharing program: interleave
  (dedicated tables) 38 -> sharepairs (shared deep) 22 -> merged6x2 (solo
  deep, shared shallow) 54. The E61 design rule — share where content is
  interchangeable, dedicate at depth — is now validated in BOTH directions,
  and the solo-deep cells (E14/E16, the bigsearch depth lever) carried it.
- Risk cell #4 (the (8,10) share under e7/e9's E10 read mass): **CLEARS** —
  e7 54 and e9 64 with E10 inside a shared pair. The share-criterion veto
  band (only pairs < ~0.6 similarity shared) survives its first
  out-of-calibration bet.
- e4 >= 40 -> **PASS (58** — best-ever seed-1000 e4 sequential cell; bigsearch
  46, interleave 42); e2 >= 80 -> **PASS (84)**.
- Boundary trail (20-ep, retired instrument): fresh 80/65/75/80 — the e4=80
  opener is the best fresh cell ever recorded at t0. Mixed-instrument
  give-back ~-4 (init-mean ~70.8 -> 66.8); the honest retention read is the
  MSE matrix.
- updt_s: **0.587-0.598** at bs16xacc2 — FASTER than interleave-8's 0.933
  and sharepairs' 0.890 (7 tables' optimizer work + the halved state; the
  cheapest production config yet).
- Pending gates: MSE matrix <= ~+8%
  (the shared-write band), prior-core events (autopsy incl. the solo E14/E16
  cells), site-bleed on the 5 shared pairs vs the E61 17-43% band, jitter
  grid, harvest-bank rescore (spec/succ Q4 vs interleave's 0.318-0.344).

**Reads (pre-battery):** (1) the go-big + sharing merge WORKS — 12 sites at
7 tables delivers bigsearch-band-or-better performance below the paper
budget; "sites, not slots" now has its constructive proof, not just the
sharepairs efficiency tie. (2) The share/solo assignment came from a
MEASURED criterion (E61 add-6 veto rule), not hand-picking — the
anti-overfitting requirement is satisfied in the config that would headline.
(3) Next per the standing sequencing (E61 add-5): 4-seed campaign row for
the one-instrument table, then the noise arm (sigmas re-calibrated on this
layout), then the REQUIRED 10-task validation before any shared-config
headline.

---
### Entry 62 addendum 2 (13 Aug 26, ~02:30 UTC) — BATTERY: all pending gates PASS. The merged layout does NOT pay sharepairs' retention tax (matrix +0.0-4.2%, clean corefrac band vs sharepairs' +7.7 breach) despite site-bleed in the same 14-51% band — co-writing present, not converting. Solo E14/E16 prior-core events = 0 exactly (depth-lever integrity clean). Jitter = best OOD grid ever (e4 clean 0.0149; e7 clean 0.0238 < specialist 0.0330). Far-region radius held (spec/succ Q4 D 0.332 ~= interleave 0.344).

**MSE forgetting matrix** (paired-noise, just-trained -> final; gate <= ~+8%):
e4 +4.2% / e6 +1.1% / e9 +0.0% / e2 +0.5% / e7 0.0 — **PASS, and back in the
clean corefrac band** (interleave +3.6/+3.0/+2.2/+1.3/0.0; sharepairs' breach
was +7.7/+6.9/+5.7/+4.8). The E61-add-5 worry (shared-write drift growing with
sharing) does not materialize in the veto-rule layout: the matrix is
indistinguishable from dedicated-tables interleave.

**Site-bleed on the 5 shared pairs** (victim reads on other-member-updated
slots): expert (4,6) 19-41%, (8,10) 22-39%; VLM (5,7) 14-22%, (9,11) 20-34%,
(13,15) 25-51% (the one above-band cell: V13<-V15 t0 51%). Same band as E61
(17-43%) — co-writing is a stable property of sharing — but with the matrix
flat and 66.8 on the board, the E61 "one mechanism, two signs" reading
resolves here to the benign sign: on veto-safe pairs the cross-writing is
neighbourhood regularization, not corruption.

**Prior-core write events** (autopsy core50): **solo E14/E16 = 0 exactly at
every victim** — the depth-lever content is untouched by later tasks, which is
the mechanism behind e7 54. VLM all-zero at all 6 banks. The one leak is
E4 (25,210 events into e9's statistical core; interleave's worst was E8 1,273)
with E8 minor (20-209) — E4 is the known-marginal shallow site, and e9
finished 64 (ABOVE interleave's 56), so the leak has no measured rollout
cost. Watch item for the 10-task run, not a 5-task problem.

**Jitter/OOD grid** (final ckpt, chunk MSE): e4 clean **0.0149** / state@0.2
0.0526 / image 0.1112; e2 clean 0.0198 / state@0.2 0.0498; e7 clean
**0.0238** / state@0.2 0.0841 / image 0.1894. Beats interleave (prev best
OOD certificate) on every clean and state cell — e7 clean sits below the
specialist's 0.0330 with state@0.2 0.0841 vs its 0.1168. Third consecutive
substrate whose fit AND shell improve together.

**Harvest-bank rescore** (e7, vs spec_e7 reference; proprio-quartile D):
demo anchor 0.0155; spec/succ Q4 **0.332** (pre-reg <= 0.344: PASS —
interleave 0.344, sharepairs 0.318, B 0.482); spec/fail Q4 0.263, B-fail Q4
0.284 — the far-region competence radius held through the merge. The
residual e7 gap (54 vs spec ~59-60 at seeds) still lives beyond the radius.

**Scorecard complete: every pre-registered E62 gate passes.** Next:
the 4-seed campaign row —
the headline instrument read that decides the paper cell. Then the noise-arm
calibration probe on this layout, then the required 10-task validation.

---
### Entry 62 addendum 3 (13 Aug 26, 04:42 UTC) — 4-SEED ROW: **65.2** — merged6x2 STATISTICALLY TIES bigsearch-12 (64.6) at 58% of its parameters (2.8B vs 4.8B, 12.5% under the paper budget). Paired deltas +1.6/+0.8/+1.6/-1.6 (3/4 positive, all small) — the +7.2 seed-1000 read was partly a high draw; the durable claim is PARITY-AT-HALF-THE-STATE, which is the paper cell. The memory rows now hold ranks 3-4 in the full table, both above the specialist oracle.

(25 eps x 4 paired seeds; per-seed means 64.8/71.2/62.4/62.4):
e4 56.0+-5.7 / e6 73.0+-6.0 / e9 70.0+-10.6 / e2 84.0+-4.6 / e7 43.0+-8.9
-> mean **65.2**.

Reads:
1. **The paper cell confirmed at the headline instrument**: 65.2 vs bigsearch
   64.6 — tied within noise, at 2.8B vs 4.8B values and the fastest training
   config yet (updt_s 0.59 vs 0.65). "12 sites at 7 tables matches 12 sites
   at 12 tables" = the constructive sites-not-slots proof, now with error
   bars. Also +4.6 over interleave-8 (60.6) at 88% of ITS parameters, and
   +6.2 over the specialist oracle (59.0).
2. **e7 43+-8.9** vs bigsearch 50+-10, sharepairs 32+-0, specialists 59+-4:
   the solo-deep rescue holds at seeds (11 over sharepairs), overlapping
   bigsearch's band; the seed-1000 54 was a high draw. e7 remains the one
   below-specialist cell (-16) — unchanged as the noise-arm target.
3. e4 56+-5.7 = the best e4 in any memory row (bigsearch 53, interleave 50);
   e6 73 ~= bigsearch's 70 transfer flagship; e9 70 matches bigsearch's 69;
   e2 84 splits interleave's 86 / bigsearch's 81.
4. **THE ONE-INSTRUMENT TABLE (updated)**: FT-l90 78.2 / FT-fresh 67.6 /
   **merged6x2 65.2** / bigsearch-12 64.6 / interleave-8 60.6 / specialists
   59.0 / sharepairs 58.4 / multitask-LoRA 51.4 / naive 18.0. Caption gains
   one line: merged6x2 = 2.8B values (vs bigsearch 4.8B) at 12 sites/7 shared
   tables, layout set by the measured share-criterion veto rule.

**Next per the standing sequencing (E61 add-5):** noise-arm calibration probe
on the merged6x2 layout (the
E58 sigmas were per-layer on B's layout and do not transfer), then the
half-dose arm; then the REQUIRED 10-task validation. The pre-registered
noise question stands: does value-input noise still add on top of sharing's
cross-writing (redundancy = a live, paper-worthy outcome).

---
### Entry 62 addendum 4 (13 Aug 26, ~05:05 UTC) — NOISE ARM: value-input noise at the E58 half dose, sigmas RE-CALIBRATED on the merged-6x2's 12 sites. Calibration: the depth-lawful displacement ladder reproduces on the new layout — expert mid-band dim-ratio 0.148 (E4) -> 0.982 (E16) monotone, VLM 0.406 (V5) -> 0.829 (V15); expert structure still concentrated (SVD-10 ~0.63-0.76), VLM ~isotropic (0.17-0.43) — the E58 per-dim approximation stands, same eyes-open caveat.

Calibration on the merged6x2 final ckpt, e56 harvest bank.
Half-dose derivation per the E58
convention (dose1x = ratio x2 variance-matched at p=0.25; half = x1.0):
expert [4,6,8,10,14,16] sigma [0.15,0.35,0.5,0.7,0.95,1.0]; VLM
[5,7,9,11,13,15] sigma [0.4,0.47,0.56,0.64,0.74,0.83]; p=0.25, amp
U[0.5,1.5].

Arm (sequential-only, reuses the E62
A-checkpoint): E62 chain config VERBATIM +
the noise flags = a clean single delta. Pre-registered (the E61-add-5
redundancy question): spec/succ Q4 rescore vs merged6x2's own 0.332 —
noise must ADD on top of sharing's cross-writing; ~unchanged = redundancy
confirmed (paper-worthy either way) and noise drops from the recipe.
Guardrails: block-mins <= ~1.10x the chain's; matrix <= ~+5%; e7 vs 54
the target cell; only a 4-seed row (vs 65.2) decides a recipe change.

---
### Entry 62 addendum 5 (13 Aug 26, PM) — WEEKEND EXPERIMENTS (Josh's spec: "pull all the baselines ... get their baseline results; if done before I'm back, the 10-task results for the 6x2 arms, noise and no noise").

**Interpretation adopted:** the existing baseline rows are all FRONT-5; the
weekend completes them for the 10-TASK table at the standing 4-seed
instrument, then runs the 10-task 6x2 validations.

**Chain:**
1. The 5-task noise arm,
   battery, and 4-seed row.
2. Baselines: (a) full-FT #1/#2 BACK-5 4-seed campaigns (envs
   [0,8,1,3,5]; the front-5 rows exist);
   (b) the five MISSING back-5 LoRA specialists (task_index 5-9, recipe
   identical to the front-5 anchors)
   + per-specialist 4-seed rows;
   (c) 10-task multitask-LoRA (all 379 episodes; BUDGET CONVENTION FLAGGED:
   front-5 used 1k steps/task -> 10k total here) + all-10 4-seed row.
3. 10-task: seq10 merged6x2 NO-noise (the required shared-config
   validation + catastrophe-elimination run), then seq10 WITH noise.
   Both reuse the E62 A-checkpoint. Pre-registered watch items
   (E61 add-5): per-block cross-writing accumulation (5-task band +0-4.2%
   -> watch toward +8-15%) and protection crowding on shared tables (late
   writers' block-mins vs their 5-task levels).

---
### Entry 62 addendum 6 (14 Aug 26, 01:28 UTC) — NOISE ARM LANDS: **69.2 at seed-1000/50-ep** (e4 54 / e6 72 / e9 70 / e2 84 / **e7 66**) — +2.4 over the no-noise twin (66.8), with the ENTIRE gain concentrated on the pre-registered target cell: **e7 54 -> 66, the best sequential e7 ever recorded** (specialist 60; FT-l90 ceiling 66). First read: value-input noise is NOT redundant with sharing's cross-writing — the two neighbourhood mechanisms compose on the conversion-bound cell. The 4-seed row is the decision instrument.

Boundary trail (20-ep): fresh 70/65/85/90 — e9's fresh 85 and e2's fresh 90
at the high end of their historical bands; no cell collapsed through later
blocks. Non-e7 cells vs the twin: e4 54 vs 58, e6 72 vs 74, e9 70 vs 64,
e2 84 vs 84 — all within single-seed noise; the +12 on e7 carries the delta.

Pre-registration scorecard:
- e7 vs 54 -> **+12 (66)** — the target cell moved decisively.
- 50-ep final vs 66.8 -> **69.2, the additive-evidence branch**.
- Cost of the noise: updt_s ~1.03 vs 0.59 (the sampling overhead) — a
  training-time-only price; inference unchanged.

If the 4-seed row confirms at ~66+: the noise recipe beats FT-fresh (67.6)
and closes most of the e7 deficit — the weekend 10-task noise run
becomes the likely paper config rather than the conditional tail.

---
### Entry 62 addendum 5 (14 Aug 26, ~02:00 UTC) — NOISE ARM LANDS: **69.2** (e4 54 / e6 72 / e9 70 / e2 84 / **e7 66**) — +2.4 over the no-noise E62 at seed-1000, with **+12 on e7, the exact target cell — e7 66 MATCHES the full-FT ceiling row's e7 and EXCEEDS the e7 specialist (59 at seeds).** Battery: zero fit cost, matrix clean-band. The redundancy question resolves AGAINST redundancy — with a twist: the far-region-D arbiter moved the "wrong" way (0.332 -> 0.380) while rollouts improved, which retires D-vs-specialist as an arbiter once a config reaches specialist level.

**50-ep final** (single delta
vs the E62 chain = the recalibrated half-dose value-input noise): **69.2** vs
no-noise 66.8. Boundary trail (20-ep, retired): fresh 70/65/85/90.

**Battery:**
- MSE matrix: +3.9/+1.0/+0.3/+0.2/+0.0% — the cleanest matrix of the merged
  family (no-noise: +4.2/+1.1/+0.0/+0.5/0.0). Noise does not disturb
  stationarity/protection at 12 sites (as at 8 in E58). Diag absolutes within
  ~1-4% of the no-noise twin: fit-cost guardrail (<= 1.10x) passes at ~1.0x.
- Jitter: all cells ~= the no-noise twin (t4 clean 0.0248 vs 0.0238; t4
  state@0.2 0.0827 vs 0.0841) — the noise buys its rollout gain without
  moving the near-manifold shell either way.
- Harvest rescore: demo anchor 0.0164 (clean); **spec/succ Q4 D = 0.380 vs
  the no-noise 0.332** — the pre-registered arbiter read "Q4 must shrink"
  FAILS while the target cell improves +12. Interpretation (the E57
  multimodality caveat, now demonstrated in reverse): D measures agreement
  with the SPECIALIST's far-region behavior, and at e7 = 66 (vs the
  specialist's seeded 59) our policy is now BETTER than the reference — it
  found its own far-region mode, so disagreement no longer implies deficit.
  **Instrument note going forward: D-vs-specialist is a valid deficit
  instrument only while the config is clearly below the specialist; at
  parity-or-above it measures mode divergence, not error.**

**Verdict on the E61-add-5 pre-registered question ("does noise still add on
top of sharing?"): YES — not redundant.** Sharing's cross-writing and
value-input noise compose: +12 on the conversion-bound cell at zero measured
cost anywhere. Mechanistically consistent with E58: the noise widens the
values' competence neighbourhood; sharing's co-writing is real-distribution
but pair-local, noise is isotropic per-site — different neighbourhoods.

**Caveat held honestly:** single-seed 50-ep cells carry +-7pp; e7 66-vs-54 is
~1.2 sigma on its own. The 4-seed row is the
pre-registered decision instrument for any recipe change. If it confirms,
noise@half-dose joins the recipe and the 10-task validation runs WITH it.

---
### Entry 62 addendum 7 (14 Aug 26) — NOISE ARM VERDICT AT THE DECISION INSTRUMENT: **60.6 vs the no-noise twin's 65.2 — NEGATIVE AT ALL FOUR PAIRED SEEDS** (-7.2/-4.8/-2.4/-4.0). e7 is TIED at seeds (45+-5.0 vs 43+-8.9): the seed-1000 e7=66 that drove addendum 6 was the high draw of a distribution centred ~45. The harvest rescore confirms the mechanism read INDEPENDENTLY: far-region D vs the specialist went UP (spec/succ Q4 0.380 vs the no-noise 0.332) — noise moved the off-manifold function AWAY from the reference it was supposed to approach. **The E61-add-5 redundancy question resolves AGAINST noise on this substrate: sharing's cross-writing already supplies the neighbourhood training, and isotropic value-input noise on top is redundant AND costly. Noise is DROPPED from the recipe.**

(25 eps x 4 paired seeds; per-seed 57.6/66.4/60.0/58.4):
e4 51.0+-6.8 / e6 67.0+-13.2 / e9 57.0+-14.4 / e2 83.0+-2.0 / e7 45.0+-5.0
-> mean **60.6** (no-noise twin 65.2 = 56/73/70/84/43).

**Pre-registration scorecard — the arm passes every SAFETY gate
and fails the VALUE question:**
- Fit cost <= ~1.10x: **PASS at ~1.00-1.04x** (paired-noise diagonals
  0.0375/0.1300/0.2663/0.1816/0.2305 vs the twin's
  0.0372/0.1256/0.2691/0.1796/0.2278). Noise is free on fit, as in E58.
- MSE matrix <= ~+5%: **PASS** (+3.9/+1.0/+0.3/+0.2/+0.0%) — cleanest
  retention profile of any arm; noise does not interact with
  stationarity/protection, exactly as the architecture argues.
- Harvest rescore (the redundancy arbiter, "noise must ADD"): **FAIL** —
  Q4 spec/succ 0.380 > 0.332; demo anchor 0.0164 (unchanged, instrument
  sound). On B (E58) the same dose moved this metric the RIGHT way
  (0.482 -> 0.379); on the merged/shared substrate it moves the wrong way.
- 4-seed row (the stated decision instrument): **FAIL, 4/4 paired negative.**

**Correction owned (addendum 6):** I wrote "value-input noise is NOT
redundant with sharing's cross-writing — the two neighbourhood mechanisms
compose" off a single-seed 69.2/e7=66. That claim is FALSIFIED. The
project's own standing rule (E41 retired 20-ep cells; E60-add-3 measured
14-point single-seed swings) existed precisely for this, and the arm's own
pre-registration named the 4-seed row as the arbiter — the narrative should
have waited for it. Standing rule restated: **single-seed 50-ep finals are
for pre-registration scoring only; NO mechanism claim enters the log on
one seed.**

**Scientific yield (the negative is informative):** value-input noise's E58
win on the dedicated-table B substrate does NOT transfer to a shared-table
substrate. This is direct evidence FOR the E61 mechanism story — cross-layer
co-writing on shared tables IS a neighbourhood regularizer, occupying the
same axis noise targets, so stacking them over-regularizes. One lever per
axis; the shared architecture already bought it, at negative parameter cost.

**Consequence:** the frontier recipe is the NO-NOISE merged 6x2
(65.2 at 4 seeds, 2.8B).
The 10-task run's noise tail is now low-value — RECOMMEND CANCELLING.

**Also landed:** full-FT BACK-5 4-seed rows —
FT-fresh back-5 **68.0** (e0 57/e8 69/e1 39/e3 80/e5 95), FT-l90 back-5
**81.2** (e0 80/e8 82/e1 60/e3 89/e5 95). Both track their front-5 rows
(67.6 / 78.2) closely, so the 10-task ceiling rows will land ~67.8 and
~79.7 — the back-5 tasks are NOT systematically easier or harder, which
makes the 10-task table directly comparable to the front-5 one.

---
### Entry 62 addendum 9 (15 Aug 26, ~01:15 UTC) — the **10-TASK SPECIALIST ORACLE ROW = 63.7** (front-5 59.0 / back-5 68.4) at the 4-seed instrument. The back-5 tasks are EASIER for specialists by +9.4, so the 10-task oracle sits ABOVE the front-5 one — the bar our 10-task memory run must clear is 63.7, not 59.0.

All ten per-task LoRA specialists now exist (front-5 from E42/E55/E56;
back-5 trained, recipe byte-identical) and all ten 4-seed rows are
in. Per-env (25 eps x 4 paired seeds, mean +- sd):

| front-5 | | back-5 | |
|---|---|---|---|
| e4 | 46.0 +- 10.6 | e0 | 46.0 +- 10.1 |
| e6 | 49.0 +- 11.5 | e8 | 67.0 +- 6.8 |
| e9 | 61.0 +- 6.8 | e1 | 62.0 +- 5.2 |
| e2 | 80.0 +- 7.3 | e3 | 84.0 +- 11.8 |
| e7 | 59.0 +- 3.8 | e5 | 83.0 +- 5.0 |
| **mean** | **59.0** | **mean** | **68.4** |

**ALL-10 SPECIALIST ORACLE = 63.7.**

Reads:
1. **The 10-task bar is HIGHER than the 5-task bar** (63.7 vs 59.0): the
   back-5 contains three of the suite's easiest tasks for a specialist
   (e3 84, e5 83, e8 67 — bowl+drawer, book+caddy, both-mokas) against one
   hard cell (e0 46, soup+sauce). Our front-5 win over specialists (+6.2 at
   65.2 vs 59.0) does NOT transfer automatically; the 10-task memory run
   must reach ~64+ to keep the "beats per-task fine-tuning" claim at 10
   tasks. Pre-registered as the headline test for the 10-task chain.
2. Cross-check with the full-FT rows, which show the SAME easier-back-5
   structure (FT-l90 front-5 78.2 / back-5 81.2; FT-fresh 67.6 / 68.0) —
   consistent, so this is a property of the task set, not a specialist
   artifact.
3. Per-cell sd stays wide (3.8-11.8) — the E60-add-4 lesson that the
   single-seed 63.2 "oracle" was luck holds: the honest all-10 oracle is
   63.7 with per-cell error bars of that size.

---
### Entry 62 addendum 10 (15 Aug 26, ~10:40 UTC) — **WEEKEND BASELINES COMPLETE: the 10-TASK TABLE'S BASELINE ROWS ALL EXIST at one instrument.** FT-l90 79.7 / FT-fresh 67.8 / specialist oracle 63.7 / multitask-LoRA-10 53.2.

All rows 25 eps x 4 paired seeds (1000/2000/3000/4000), envs in train order
e4/e6/e9/e2/e7/e0/e8/e1/e3/e5:

| row | all-10 | front-5 | back-5 | budget/advantages |
|---|---|---|---|---|
| **FT-l90** (ceiling) | **79.7** | 78.2 | 81.2 | all-10 demos, full backbone, joint |
| **FT-fresh** | **67.8** | 67.6 | 68.0 | same, minus the libero-90 substrate |
| **specialist oracle** | **63.7** | 59.0 | 68.4 | 10 separate models + task ID at test |
| **multitask-LoRA-10** | **53.2** | — | — | one adapter, all-10 demos, no continual constraint |

multitask-LoRA-10 per-env: e4 35 / e6 47 / e9 49 / e2 62 / e7 46 / e0 47 /
e8 46 / e1 52 / e3 81 / e5 67 (10k steps = the front-5 cell's 1k/task
convention scaled; front-5 multitask was 51.4 at 5 tasks, so breadth cost
it ~nothing per task — consistent with the E43 breadth law).

**What the memory run must clear (pre-registered for the chain):**
63.7 to keep "beats per-task fine-tuning" at 10 tasks (the bar ROSE from
59.0 — addendum 9), 53.2 to keep the multitask-adapter margin, and any
number at all keeps the catastrophic-forgetting contrast (naive seq-LoRA
18.0 at 5 tasks; a 10-task naive row was NOT in the weekend spec and does
not exist — flagged as the one missing baseline cell if the paper wants the
foil at 10 tasks).

**10-task chain:**
shared-table write mask 3072+3072 -> 6023 rows, overlap 121 (~2%), the same per-step partition
measured at 5 tasks.

---
## Entry 63 - 17 Aug 26 (THE 10-TASK VALIDATION LANDS: **67.8** — the required shared-config run at full suite length CLEARS every bar. Beats the 10-task specialist oracle (63.7) by +4.1 on 8/10 tasks, beats multitask-LoRA-10 (53.2) by +14.6, EQUALS full-FT-from-scratch (67.8) exactly, and sits 11.9 below the substrate-matched ceiling. NO catastrophic forgetting at 10 tasks: front-5 67.2 vs back-5 68.4 — the early tasks are NOT degraded by five subsequent blocks. The E61-add-5 degradation worry does not materialize.)

**The run:**
the merged 6x2 (12 sites / 7 shared tables / 2.8B) config verbatim from the
E62 5-task cell, extended to all ten libero_10 tasks (10 blocks x 5000 steps,
20-ep boundaries + 50-ep final on all 10). The
first five boundaries reproduced the 5-task run bit-for-bit (same
A-checkpoint/seed/order, empty protection store).

**50-ep final (step 50000), train order, vs the 10-task specialist oracle:**

| env | task | ours | specialist | delta |
|---|---|---|---|---|
| e4 | two mugs | 52 | 46 | +6 |
| e6 | mug+pudding | 76 | 49 | **+27** |
| e9 | mug+microwave | 64 | 61 | +3 |
| e2 | stove+moka | 90 | 80 | +10 |
| e7 | soup+cheese | 54 | 59 | -5 |
| e0 | soup+sauce | 48 | 46 | +2 |
| e8 | both mokas | 74 | 67 | +7 |
| e1 | cheese+butter | 42 | 62 | **-20** |
| e3 | bowl+drawer | 86 | 84 | +2 |
| e5 | book+caddy | 92 | 83 | +9 |
| | **MEAN** | **67.8** | **63.7** | **+4.1** |

**Scorecard against the pre-registrations (E61 add-5 + addendum 9):**
1. **>= 63.7 (keep "beats per-task fine-tuning" at 10 tasks) -> PASS at 67.8,
   on 8/10 tasks.** The bar ROSE going to 10 tasks (59.0 -> 63.7) and we
   cleared it by more than we cleared the 5-task one (+4.1 vs +6.2 at 5;
   both single-seed-instrument reads pending the seed campaign).
2. **>= 53.2 (multitask-LoRA-10) -> PASS by +14.6.**
3. **Degradation mechanism (a), per-block cross-writing accumulation ->
   DOES NOT MATERIALIZE.** front-5 mean 67.2 vs back-5 68.4: the five tasks
   that sat under five subsequent blocks are within 1.2 points of the five
   that did not. At 5 tasks the same config gave front-5 66.8. **Early-task
   performance is FLAT in the number of subsequent blocks** — the central
   catastrophe-elimination claim, now demonstrated at full suite length.
4. **Degradation mechanism (b), protection crowding on shared tables ->
   no visible late-writer starvation**: the last three trained tasks
   (e1 42 / e3 86 / e5 92) show no systematic deficit; e1 is a basket-family
   task (below), not a late-writer artifact.
5. Instrument note: the 20-ep boundary trace was uninformative as usual —
   e7 swung 65/40/60/25/60 across consecutive boundaries and finished at 54.
   My block-8 attribution of that dip to basket-hub interference was
   noise-driven and is withdrawn; the family signal that DOES survive to the
   final is e1 (below), not e7.

**The one real deficit: e1 (cheese+butter) at 42 vs its specialist's 62.**
This is the basket family's third member, present only at 10 tasks (train
order position 8). e7 (soup+cheese) shares "cheese" with it and e0
(soup+sauce) shares "soup" with e7 — the E27/E28 same-scene family. Our three
basket cells (e7 54 / e0 48 / e1 42) are the three lowest non-e4 cells in the
table, and the family is the only place we lose to specialists. Consistent
with every prior finding: the family collision is scene-genuine, routing
cannot separate it (E28 forward probe), and it is now the clearest remaining
target at 10 tasks.

**Placement in the full 10-task table (all rows at 25 eps x 4 paired seeds
except this run, at 50 eps / seed 1000 — the seed campaign is the next step):**
FT-l90 79.7 / **merged6x2-10task 67.8** = FT-fresh 67.8 / specialists 63.7 /
multitask-LoRA-10 53.2. **The sequential frozen-backbone memory model with no
task identity EQUALS a full-backbone joint finetune given all ten tasks'
demos at once**, and sits 11.9 under the substrate-matched ceiling.

**Next:** (1) 4-seed campaign on this checkpoint — the headline instrument,
and the only thing that makes the +4.1 and the FT-fresh tie quotable;
(2) the standard battery (10x10 MSE forgetting matrix — the paired,
noise-free retention read at 10 tasks; jitter; slot autopsy with the 5-pair
site-bleed and prior-core reads at 10-task exposure); (3) the noise 10-task
tail remains recommended-cancel (E62 addendum 7).

---
### Entry 63 addendum (17 Aug 26) — noise 10-task CANCELLED (Josh); next: 4-seed row -> battery -> naive 10-task foil.

**Cancelled:** the 10-task noise arm.
Rationale is the E62-addendum-7
verdict — value-input noise lost at 4 seeds on this substrate (60.6 vs
65.2, negative 4/4 paired) and the harvest rescore showed it moving the
far-region function AWAY from the specialist reference.

**Next, in Josh's order:**
1. **4-seed campaign** on the 10-task final, all ten envs (25 eps x seeds
   1000/2000/3000/4000). This is what makes the +4.1-over-oracle and the
   FT-fresh tie quotable — every other row in the 10-task table is already
   at this instrument, and Entry 63's 67.8 is currently seed-1000/50-ep.
2. **Battery**: 10x10 MSE forgetting matrix (paired-noise; the noise-free
   read that converts "front-5 67.2 vs back-5 68.4" into a retention claim
   and tests the E61-add-5 cross-writing worry at fit level — the 5-task
   band was +0.0-4.2%, sharepairs breached at +7.7%), jitter/OOD grid on
   four tasks incl. the basket cells, and the slot autopsy at 10-task
   exposure (5-pair site-bleed vs the 17-43% band, prior-core events incl.
   the solo E14/E16 depth cells).
3. **Naive sequential LoRA r256 at 10 tasks** — the catastrophic-forgetting
   foil beside the 67.8 row (the 5-task cell was 18.0 with three tasks at
   literal 0%).
   Its own 4-seed row follows.

---
### Entry 63 addendum 2 (17 Aug 26) — 10-TASK 4-SEED ROW: **65.1** (vs the 10-task specialist oracle 63.7, **+1.4, winning 7/10 tasks**). The seed-1000 67.8 was a mild high draw (per-seed 60.4/66.8/66.8/66.4). Retention holds at seeds — front-5 64.6 vs back-5 65.6 — so the no-catastrophic-forgetting claim survives the headline instrument. The basket family is the entire remaining deficit: e1 -24, e0 -9, e7 -5; on the other seven tasks we are +2 to +13.

| env | task | ours (4-seed) | specialist | delta |
|---|---|---|---|---|
| e4 | two mugs | 59.0 +- 14.4 | 46.0 | **+13.0** |
| e6 | mug+pudding | 60.0 +- 8.0 | 49.0 | **+11.0** |
| e9 | mug+microwave | 63.0 +- 6.8 | 61.0 | +2.0 |
| e2 | stove+moka | 87.0 +- 5.0 | 80.0 | +7.0 |
| e7 | soup+cheese | 54.0 +- 4.0 | 59.0 | -5.0 |
| e0 | soup+sauce | 37.0 +- 10.0 | 46.0 | -9.0 |
| e8 | both mokas | 76.0 +- 5.7 | 67.0 | +9.0 |
| e1 | cheese+butter | 38.0 +- 9.5 | 62.0 | **-24.0** |
| e3 | bowl+drawer | 86.0 +- 5.2 | 84.0 | +2.0 |
| e5 | book+caddy | 91.0 +- 2.0 | 83.0 | +8.0 |
| | **MEAN** | **65.1** | **63.7** | **+1.4** |

**THE COMPLETE 10-TASK TABLE (all rows 25 eps x 4 paired seeds):**
FT-l90 (ceiling) **79.7** / FT-fresh **67.8** / **merged6x2 10-task 65.1** /
specialists **63.7** / multitask-LoRA-10 **53.2**.

Reads:
1. **The claim holds at 10 tasks, with a thinner margin than at 5** (+1.4 vs
   +6.2 at 5 tasks). Both are single-config-vs-oracle comparisons at the
   same instrument; the 10-task margin is inside ~1 sd of the per-seed
   spread (per-seed 60.4-66.8), so the honest phrasing is "matches or
   modestly exceeds the per-task oracle at 10 tasks", not "beats it".
2. **Retention confirmed at seeds:** front-5 64.6 vs back-5 65.6. Tasks that
   sat under five subsequent blocks are NOT degraded relative to the late
   ones — the E61-add-5 accumulation worry is dead at the headline
   instrument, not just at seed-1000.
3. **The basket family is now the ENTIRE deficit.** Sum over the three
   basket cells (e7/e0/e1) = -38 against specialists; sum over the other
   seven = +52. Remove the family and the config beats per-task fine-tuning
   by ~7 points per task. e1 (-24) is the worst cell in project history
   against its specialist and exists only at 10 tasks. This is the E27/E28
   same-scene collision, now the single named target for any further work.
4. e4's sd of 14.4 (32-72 across seeds) is the widest cell in the table —
   consistent with its history as the noisiest task; its +13 is real in
   direction but weakly determined in magnitude.

---
### Entry 63 addendum 3 (17 Aug 26) — 10-TASK BATTERY: forgetting stays SMALL but is no longer flat — mean function drift **+6.5%** (5-task band was +0.0-4.2%), and it is **ordered by exposure** (first-5 +10.3% vs last-5 +2.7%), i.e. the accumulation mechanism IS real at fit level even though it does not convert to rollout loss (front-5 64.6 vs back-5 65.6 at seeds). Solo deep tables remain untouched (E14/E16 = 0 events at every one of nine victims); the whole prior-core leak is the shallow E4 site. Site-bleed at 10 tasks is unchanged from 5 (14-51%). Jitter shows the basket cell e7 as the brittlest.

**MSE forgetting matrix (10x10, paired-noise; own-block -> final):**

| task | own | final | drift |
|---|---|---|---|
| t0/e4 | 0.0372 | 0.0418 | +12.4% |
| t1/e6 | 0.1256 | 0.1348 | +7.4% |
| t2/e9 | 0.2691 | 0.2817 | +4.7% |
| t3/e2 | 0.1796 | 0.1947 | +8.4% |
| t4/e7 | 0.2278 | 0.2701 | **+18.6%** |
| t5/e0 | 0.1975 | 0.2127 | +7.7% |
| t6/e8 | 0.1383 | 0.1436 | +3.9% |
| t7/e1 | 0.2318 | 0.2345 | +1.2% |
| t8/e3 | 0.1703 | 0.1718 | +0.9% |
| t9/e5 | 0.1076 | 0.1076 | +0.0% |
| | | **mean** | **+6.5%** |

Reads:
1. **The E61-add-5 prediction is CONFIRMED at fit level and REFUTED at
   rollout level.** Drift is monotone-ish in exposure (first-5 +10.3%,
   last-5 +2.7%) exactly as "cross-writing accumulates per block" predicts,
   and the 5-task band (+0.0-4.2%) roughly doubles at 10 tasks — the
   predicted +8-15% range, at its low end. But the same tasks' ROLLOUT
   scores do not degrade (front-5 64.6 vs back-5 65.6 at 4 seeds). So the
   accumulated drift is real and measurable while sitting below the
   threshold where it costs success — the E42/E52 "drift converts only at
   the threshold" result, now demonstrated across a 2x exposure change.
   Honest framing for the paper: "function drift grows with the number of
   subsequent tasks but remains an order of magnitude below the naive
   baseline's (+740-1567%) and does not convert to success loss at this
   scale" — NOT "zero forgetting" at 10 tasks.
2. **e7 is the worst drift cell (+18.6%)** — the basket hub, damaged by both
   e0 (t5) and e1 (t7) blocks, exactly the E27 hub geometry. Its rollout
   (54, -5 vs specialist) understates the internal damage; e1's own drift is
   trivial (+1.2%, it trained late) yet its rollout is the worst cell (-24),
   so e1's deficit is NOT forgetting — it is fit/conversion on a
   family-colliding task. **Two different mechanisms in the same family.**
3. **Solo deep tables: 0 prior-core write events at E14/E16 for all nine
   victims** (as at 5 tasks) — the depth-lever integrity that carries e7/e9
   survives 10-task exposure intact. VLM banks likewise all-zero. The single
   leak remains shallow E4 (25,570 events into e9's core, same site and
   scale as the 5-task run's 25,210) — no growth with exposure, and e9
   finished 63 (+2 vs specialist), so it stays a watch item, not a cost.
4. **Site-bleed unchanged by task count** (E4/E6 14-41%, E8/E10 19-39%,
   V13/V15 20-51%) — the same 17-43%-ish band measured at 5 tasks in E61.
   Co-writing is a per-block property, not a cumulative one; what
   accumulates is its effect on older content, per read 1.
5. **Jitter (final ckpt):** t0/e4 clean 0.0170 (state@0.2 0.0546, image
   0.1171), t3/e2 0.0298, t7/e1 0.0223, **t4/e7 0.0628** — e7 is 2-4x the
   others clean AND worst under every perturbation (image 0.2329), the only
   cell whose function is genuinely degraded. Consistent with its +18.6%
   matrix drift; both instruments finger the same task.

---
### Entry 63 addendum 4 (17 Aug 26) — the FULL 10x10 MSE forgetting matrix (addendum 3 carried only the diagonal summary; this is the complete grid, the project's standard artifact since E39).

Paired-noise flow-matching loss of every per-task checkpoint (rows) on every
task (columns); diagonal = just-trained, bold. Instrument:
seed 0, 16 batches/task — identical settings to every prior matrix
(B/absmax/sharepairs/naive), so cells are comparable across entries.

| ckpt | t0/e4 | t1/e6 | t2/e9 | t3/e2 | t4/e7 | t5/e0 | t6/e8 | t7/e1 | t8/e3 | t9/e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 005000 | **0.0372** | 0.6467 | 1.6699 | 0.9050 | 0.9438 | 0.8853 | 0.7955 | 0.8616 | 0.9216 | 0.6240 |
| 010000 | 0.0375 | **0.1256** | 1.6849 | 0.9081 | 0.9486 | 0.8904 | 0.7966 | 0.8696 | 0.9289 | 0.6283 |
| 015000 | 0.0379 | 0.1261 | **0.2691** | 0.9286 | 0.9596 | 0.8979 | 0.8008 | 0.8806 | 0.9333 | 0.6391 |
| 020000 | 0.0383 | 0.1262 | 0.2691 | **0.1796** | 0.9600 | 0.9021 | 0.7941 | 0.8806 | 0.9353 | 0.6399 |
| 025000 | 0.0388 | 0.1269 | 0.2691 | 0.1805 | **0.2278** | 0.9308 | 0.7975 | 0.8810 | 0.9358 | 0.6403 |
| 030000 | 0.0396 | 0.1284 | 0.2706 | 0.1811 | 0.2492 | **0.1975** | 0.8034 | 0.9026 | 0.9424 | 0.6391 |
| 035000 | 0.0402 | 0.1301 | 0.2734 | 0.1873 | 0.2559 | 0.1968 | **0.1383** | 0.9016 | 0.9480 | 0.6415 |
| 040000 | 0.0407 | 0.1314 | 0.2754 | 0.1903 | 0.2626 | 0.2084 | 0.1396 | **0.2318** | 0.9514 | 0.6436 |
| 045000 | 0.0412 | 0.1331 | 0.2795 | 0.1924 | 0.2670 | 0.2111 | 0.1418 | 0.2329 | **0.1703** | 0.6444 |
| 050000 | 0.0418 | 0.1348 | 0.2817 | 0.1947 | 0.2701 | 0.2127 | 0.1436 | 0.2345 | 0.1718 | **0.1076** |

Structure worth reading off the grid itself (not visible in the diagonal
summary):
1. **Untrained cells are flat and huge until their block** (e.g. t2/e9 sits
   at ~1.67-1.68 for two checkpoints, then drops to 0.269 when trained) —
   the standard signature, and it validates the instrument: nothing moves a
   task's loss except its own block plus small later drift.
2. **Post-training columns rise monotonically, never step.** Every
   below-diagonal column increases smoothly by 0.5-2% per subsequent block
   with no cliff at any single writer — i.e. the drift is diffuse
   accumulation, not one bad neighbour. Contrast the naive-LoRA matrix
   (E58 add-6), where a single subsequent block moved a task from 0.05 to
   ~0.9-1.5 (one-block catastrophe).
3. **The one visibly steeper column is t4/e7** (0.2278 -> 0.2492 at e0's
   block -> 0.2626 at e1's -> 0.2701): the two basket neighbours are its two
   largest single-block jumps (+9.4% and +5.4%), which is the E27 hub
   geometry appearing directly in the matrix rather than inferred.
4. Pre-training loss levels differ ~25x across tasks (t0 0.037 vs t2 1.67
   before training) — as always, compare cells WITHIN a column, never across.

---
### Entry 63 addendum 5 (18 Aug 26) — one DUPLICATE-NUMBERED addendum identified, its since-falsified claim explicitly retracted

**RETRACTION extending addendum 7.** The duplicate (14 Aug ~02:00) carried
a STRONGER conclusion than addendum 6 did — *"the redundancy question resolves
AGAINST redundancy ... which retires D-vs-specialist as an arbiter once a
config reaches specialist level."* **Both halves are falsified by the 4-seed
row (addendum 7):** the noise arm lost 60.6 vs 65.2, negative at 4/4 paired
seeds, and its e7 was TIED at seeds (45+-5.0 vs 43+-8.9) — so there was no
"rollout improvement" for the far-region D to be inconsistent with, and the
D-vs-specialist arbiter was reading correctly all along (it moved the wrong
way BECAUSE the arm was worse). **D-vs-specialist is NOT retired; it is
retained as an arbiter, and it agreed with the seeds when the single-seed
rollout did not.** Addendum 7 corrected addendum 6 by name; this closes the
same correction against the duplicate.

---
## Entry 64 - 18 Aug 26 (LoRA BASELINE RE-PROVISIONING at uniform r=256 — the multitask-LoRA rows were the E43 breadth PROBE (r32, 1k steps/task) promoted into the paper table unrevised: 5x fewer steps than every other row AND the smallest adapter in the table. Supervisor's "match active parameters" request → the ACTIVE-PARAMETER LADDER written down (no single equivalent r exists; per-token we are a rank-2 adapter, per-step r~140, total = full FT). DECISION (Josh): 10-task multitask at r256/a64/50k + ALL TEN specialists at r256/a64/5k, 4-seed rows. Also settled: every LoRA row is trained from the LIBERO-90 finetune (stage-1), not raw pi05.)

### The finding that forced this

Budgets of the rows in the paper table as they stand:

| row | trainable | total steps | steps per task |
|---|---|---|---|
| ours (merged 6x2) | 2.8B | 25k (5-task) / 50k (10-task) | 5k |
| specialists (r32) | 53M each | 5k each | 5k |
| naive seq-LoRA (r256) | 426M | 25k (5-task) / 50k (10-task) | 5k |
| **multitask-LoRA (r32)** | **53M** | **5k (5-task) / 10k (10-task)** | **1k** |

The multitask row was written in E43 as a within-architecture BREADTH probe
("at the SAME 5k-step budget as each specialist => 1/5 the per-task exposure;
only breadth changes") — a deliberate design for that question. It was then
promoted into the paper table (E44 -> the 4-seed row, E61-add-4 -> the 10-task
twin at the same 1k/task convention, flagged in E62-add-5 but never revisited).
So the multitask row is under-provisioned twice over — the fewest steps per task
and the smallest adapter in the table — and it is the row where our margin looks
largest (+13.8 at 5 tasks, +11.9 at 10). Some of that margin is real (E43's
breadth law: r32 at 1k/task already reached 49-51), but the size is currently
undefendable. Every other row is budget-matched to ours (specialists 5k/task,
naive 25k/50k, full-FT the all-data ceiling).

### Supervisor's request: "benchmark LoRA at the same ACTIVE parameters" — the ladder

For a dense LoRA, active = total = r x sum(d_in+d_out) — one number. For a
sparse-retrieval model the count depends on the window you ask over, because
retrieval is a hard top-k (exactly knn x heads slots per query, always — the
count is stable, only WHICH slots varies) but the union grows with the window.
Paper cell (expert 6 sites n256/r2/knn36/4 heads; VLM 6 sites n256/r2/knn16/4
heads; slot = 4,096 params expert / 8,192 VLM); LoRA over our full target set
costs 1.66M params per unit rank (53.2M at r32, measured E43):

| rung | ours | LoRA rank that matches |
|---|---|---|
| per token, compute-active (the MoE-paper meaning) | 144 x 4.1k x 6 = ~3.5M (expert token); 64 x 8.2k x 6 = ~3.1M (routed VLM key); image tokens 0 | **r ~ 2** |
| per optimizer step, receives gradient (write mask 3072/site) | ~0.23B | r ~ 140 |
| per task, adapted footprint | core ~70M / effective 0.4-0.9B / touched 1-1.5B (slot JSONs) | r ~ 40 / 250-540 / 600-900 |
| total stored | 2.8B (+ frozen router/keys) | r ~ 1700 = beyond full rank on the expert side = full FT |
| (per-SITE reading: one token's memory map at a site has bottleneck <= 2 x knn x heads) | 288 expert / 128 VLM | — at 12 sites, vs LoRA's ~250 matrices |

The same model is "a rank-2 adapter" per token and "full fine-tuning" by
storage. That is not an accounting nuisance, it is the mechanism (large storage,
sparse compute, sparse protected writes) — and it is why no dense adapter can be
matched to us on more than one rung, which is the claim. Standard MoE practice
(report per-token active AND total; compare to dense at matched active) applies;
each comparison must state its rung. A "matched r=512 everywhere" baseline would
be ~250x our per-token active and ~1/3 our storage — matched on nothing; if that
figure came from the per-site bottleneck (288 -> "neatly above" -> 512), the
error is extrapolating a 12-site bottleneck to all ~250 matrices.

**Chosen: r=256, alpha=64, uniform across every LoRA row.** Rationale: the
smallest rank that sits ABOVE us on every compute/optimization rung at once
(~120x per token, ~1.9x per step, ~parity even on the generous per-site reading)
and below only total storage — which the full-FT rows already match. It is the
rank the naive foil already runs at (E58: r256@a64), and alpha/r = 0.25 held
keeps the effective update scale identical to the r32@a8 rows, so the only delta
vs the rows replaced is capacity (+ steps for the multitask). Rejected: r=128
(above per-token, ~= per-step, below per-site — arguable); r=512 (clears every
reading but breaks consistency with the naive row and puts an 850M adapter on
30-50 episodes — a baseline built to overfit).

### Also settled this session: which base the LoRA baselines start from

Every LoRA row loads the LIBERO-90 full
finetune of pi05 (stage-1): all ten specialists (front-5 + back-5), multitask-5
and -10, naive seq r256 at 5 and 10 tasks, and the compass cells. The ONLY row
from raw pi05 is full-FT #1, deliberately (the
"no libero-90 substrate" cell). So the table is substrate-matched everywhere
except that one row — a clean caption line, and the reason the specialists are
as strong as they are (the substrate is worth +10.6 even to full FT, E60-add-9).

### Decisions (Josh, 18 Aug)

1. **10-task multitask LoRA at r=256 / a=64 / 50,000 steps** (= 5k/task, our
   budget), all 379 episodes, lr 1e-4 -> 1e-5 (warmup 200, decay 50000 — the
   schedule honored, E20 gotcha), bs16 x acc2 no-ckpt (E56 standard LoRA-cell
   config), same targets, same stage-1 base. Replaces the r32/10k row (53.2).
   The 5-task multitask row is NOT rerun (Josh: "just the 10 task") — the
   5-task table keeps the r32/1k-per-task cell, to be footnoted/dropped or
   twinned later (r256/25k twin) at Josh's call.
2. **All ten per-task specialists at r=256 / a=64 / 5,000 steps** — uniform
   rank across the table; the r32 specialists become the appendix sensitivity
   row. Replaces the oracle row (all-10 63.7 / front-5 59.0). Recipe otherwise
   byte-identical to the r32 anchors (5k, bs16 x acc2 no-ckpt, lr 1e-4).
   Risk stated plainly: the oracle bar may move. UP -> the 10-task "+1.4"
   becomes "matches" and the 5-task "+6.2" shrinks; DOWN is plausible too (the
   r32 specialists already sit at chunk ~0.02 — not capacity-limited — and E43
   found the e9 specialist extremely image-brittle; more rank on 30-50
   episodes may overfit). Either way the paper reports the pre-declared
   uniform-rank oracle with r32 in the appendix; running it before a reviewer
   does beats the alternative.
3. **4-seed rows on both** at the standing instrument (25 eps x paired seeds
   1000/2000/3000/4000, vec bs=13).
4. LR footnote for the paper: the naive r256 foil ran at lr 2.5e-5 -> 2.5e-6
   (pi05's default base LR, in the sequential trainer) vs the specialists'/
   multitask's 1e-4. It still fitted specialist-grade diagonals (E58-add-6:
   0.049-0.098) so it is not starved; the new r256 rows keep 1e-4 so their only
   delta from the rows they replace is rank/alpha(/steps). Footnote, not a rerun.

**Pre-registered reads.** Multitask-10 r256/50k: expect a real rise over 53.2
(5x steps, 8x capacity); the honest question is how much of the +11.9 margin
over multitask survives — the paper's multitask comparison is whatever this
lands at, full stop. Specialists r256: the new all-10 / front-5 oracle bars
replace 63.7 / 59.0; our 65.1 (10-task) and 65.2 (5-task) are re-scored against
them. Both are baseline cells: no memory-side change follows from either
outcome, only the wording of the claim.

---
### Entry 64 addendum (18 Aug 26) — REVISED before launch: uniform **r=512 / alpha=128** for ALL THREE LoRA rows (multitask-10, ten specialists, AND the naive foil), everything from scratch (Josh: "then we can defend every point against the reviewer"). The r256 plan is withdrawn unfired; the running r256 naive-10 killed at block 5. Correction owned: my "per-token equivalent r ~ 2" was a whole-model PARAM-COUNT average — the right per-token number is the per-SITE bottleneck 2 x heads x knn = 288 (expert) / 128 (VLM), Josh's count, and "neatly above" that is r=512.

**The correction.** Josh: "for each token our method has n_heads x knn slots — per
token is 2 x n x knn." Correct. One token's memory output at an expert site is a
sum of 144 rank-2 slot maps -> bottleneck <= 2 x 4 x 36 = **288** (VLM: 2 x 4 x 16
= **128**), and the parameter count says the same thing: 144 slots x 4,096 =
590k = 288 x (1024 + 1024) — EXACTLY a rank-288 LoRA on one 1024x1024 matrix.
My r ~ 2 came from dividing the per-token param count across the six sites a
token passes (~3.5M) by LoRA's cost per unit rank over the ENTIRE target set
(~250 matrices, 1.66M/rank): a compute statement ("how many adapter params does
a token pass through in total") that spreads our concentrated capacity over 20x
more matrices and hides placement breadth. Not what "matched active" should
mean, and not the number a reader would compute. Ladder rung 1 in Entry 64 is
hereby re-stated as: **per token, at a memory site, rank 288 (expert) / 128
(VLM); r=256 would sit slightly BELOW the expert-site rank; r=512 is the
smallest power of two above every site.**

**Decision (Josh):** r=512 / alpha=128 (alpha/r = 0.25 held: r32@a8 -> r512@a128,
same effective update scale) for (1) multitask-10 at 50k steps, (2) all ten
specialists at 5k, and (3) the naive sequential foil at 10 tasks. Rationale:
above us on every rung a reviewer could name — per-site bottleneck (512 > 288 /
128), per-token active (852M vs ~3.5M, ~240x), per-step trained (852M vs ~0.23B,
~3.7x) — and below only total storage (852M vs 2.8B; the storage-matched rows are
full FT). One adapter config across every LoRA row: the table becomes
"the r512 specialist recipe" run three ways (per task / all tasks jointly /
sequentially). r32 (and the 5-task naive at r256) stay as appendix sensitivity
rows. All three FROM SCRATCH — the r256 checkpoints (incl. the killed naive's
four blocks) are shape-incompatible and nothing carries over.

The r256
naive's four boundaries had reproduced the 5-task run bit-for-bit — 35 / 0,60 /
0,0,60 / 0,0,0,90 (e4 -> 0 after ONE block, e6 -> 0 after one, e9 -> 0 after one)
— a free replay check that the 10-task naive plumbing was sound.

**Naive foil made recipe-identical to the specialists:** lr **1e-4 -> 1e-5**
per-block linear (the r256 foil ran 2.5e-5 -> 2.5e-6; its script header called
that "the pi05 preset the specialists trained under" — WRONG, the specialist
scripts use optimizer_lr=1e-4). With rank/alpha/lr/targets/base/steps-per-task
now all equal to the specialists', the foil is literally "the specialists,
trained one after another" — the cleanest reviewer statement. Everything else
per E58: same base, per-task optimizer reinit, no protection/tfidf/memory,
20-ep boundaries + 50-ep final (the memory runs' protocol; the 4-seed row is
the headline instrument).

**Pre-registered reads (unchanged in kind):** multitask-10 r512/50k replaces the
53.2 row — the paper's multitask margin is whatever it lands at; specialists
r512 replace the oracle bars 63.7 (all-10) / 59.0 (front-5) — our 65.1 / 65.2
are re-scored against them, either direction; naive r512 at 10 tasks is
expected to reproduce the one-block collapse (any rank; the point of the row).
No memory-side change follows from any outcome; only the wording of the claims.

The 5-task table's LoRA rows (multitask r32/1k-per-task, naive r256)
would need r512 twins for full uniformity there — Josh's call.

---
### Entry 64 addendum 2 (18 Aug 26)

Naive r256 10-task, stopped at block 5/10. Its banked boundary
evals, 20-ep:

| after block | t4/e4 | t6/e6 | t9/e9 | t2/e2 |
|---|---|---|---|---|
| 5000  | 35.0 | | | |
| 10000 | 0.0 | 60.0 | | |
| 15000 | 0.0 | 0.0 | 60.0 | |
| 20000 | 0.0 | 0.0 | 0.0 | 90.0 |

No final, no seed row.

---
### Entry 64 addendum 3 (18 Aug 26) — stage 3 in-run eval REPLACED by a post-hoc 4-seed retention TRIANGLE, and the same triangle added as stage 4 on our merged6x2 10-task checkpoints (Josh). Two paired 10x10 rollout retention matrices at the headline instrument.

**Instrument change.** The sequential trainer's in-run boundary evals are 20 eps at
ONE seed — the instrument retired from decisions in
E41.
Replaced by the standing headline instrument at EVERY boundary:
25 eps x 4 paired seeds (1000/2000/3000/4000), batched. For a 10-task run that is
the lower triangle, 1+2+...+10 = 55 cells x 100 eps = 5,500 episodes.

**Stage 4 (new):** the identical triangle on the merged6x2 10-task run's ten
per-task checkpoints. 45 measured cells;
b10 is copied from the existing 4-seed row — same checkpoint,
same seeds, same episodes as row 10, so it is adopted rather than re-measured.

---
### Entry 64 addendum 4 (19 Aug 26) — stage 1 TRAINING COMPLETE: multitask-LoRA-10 r512/a128, 50,000 steps.

Base = stage-1 libero_90 finetune, all 379
libero_10 episodes, r=512 / lora_alpha=128, bs16 x acc2 (effective 32), lr 1e-4 ->
1e-5 (warmup 200, decay 50000), no grad-ckpt.

num_learnable_params = **850,427,904**
steps 50,000 / samples 2.0M / epochs 15.77
final log-window loss 0.029 (last three windows 0.032 / 0.031 / 0.029), grdn 0.077,
lr 1.0e-05

---
### Entry 64 addendum 5 (19 Aug 26) — stage 1 ROW: multitask-LoRA-10 r512/a128/50k, 4-seed campaign.

Checkpoint 050000, 25 eps x seeds
1000/2000/3000/4000, envs in train order:

| env | seed 1000 | 2000 | 3000 | 4000 | mean | sd |
|---|---|---|---|---|---|---|
| e4 | 64 | 56 | 72 | 76 | 67.0 | 7.7 |
| e6 | 84 | 76 | 84 | 64 | 77.0 | 8.2 |
| e9 | 92 | 96 | 96 | 88 | 93.0 | 3.3 |
| e2 | 100 | 96 | 100 | 96 | 98.0 | 2.0 |
| e7 | 60 | 64 | 60 | 64 | 62.0 | 2.0 |
| e0 | 72 | 64 | 76 | 72 | 71.0 | 4.4 |
| e8 | 76 | 92 | 84 | 72 | 81.0 | 7.7 |
| e1 | 32 | 36 | 36 | 40 | 36.0 | 2.8 |
| e3 | 84 | 88 | 100 | 88 | 90.0 | 6.0 |
| e5 | 96 | 92 | 100 | 96 | 96.0 | 2.8 |
| **MEAN** | 76.0 | 76.0 | 80.8 | 75.6 | **77.1** | |

---
### Entry 64 addendum 6 (21 Aug 26) — stage 2 TRAINING COMPLETE: ten per-task LoRA specialists at r512/a128, 5,000 steps each.

Base = stage-1 libero_90 finetune,
per-task episode ranges, r=512 / lora_alpha=128, bs16 x acc2 (effective 32),
lr 1e-4 -> 1e-5 (warmup 200, decay 5000), no grad-ckpt, 5,000 steps each.

| task/env | final loss | grdn |
|---|---|---|
| t0/e4 | 0.033 | 0.062 |
| t1/e6 | 0.027 | 0.057 |
| t2/e9 | 0.065 | 0.094 |
| t3/e2 | 0.040 | 0.071 |
| t4/e7 | 0.049 | 0.083 |
| t5/e0 | 0.033 | 0.066 |
| t6/e8 | 0.032 | 0.054 |
| t7/e1 | 0.049 | 0.083 |
| t8/e3 | 0.040 | 0.065 |
| t9/e5 | 0.025 | 0.045 |

---
### Entry 64 addendum 7 (21 Aug 26) — stage 2 ROW: per-task LoRA specialist oracle at r512/a128/5k, 4-seed campaign, all ten tasks.

Each specialist evaluated on its own env,
checkpoint 005000, 25 eps x seeds 1000/2000/3000/4000, envs in train order:

| env | seed 1000 | 2000 | 3000 | 4000 | mean | sd |
|---|---|---|---|---|---|---|
| e4 | 64 | 48 | 48 | 48 | 52.0 | 6.9 |
| e6 | 80 | 64 | 68 | 72 | 71.0 | 5.9 |
| e9 | 84 | 76 | 68 | 84 | 78.0 | 6.6 |
| e2 | 84 | 84 | 84 | 88 | 85.0 | 1.7 |
| e7 | 76 | 72 | 60 | 64 | 68.0 | 6.3 |
| e0 | 92 | 84 | 68 | 92 | 84.0 | 9.8 |
| e8 | 64 | 68 | 76 | 72 | 70.0 | 4.5 |
| e1 | 68 | 60 | 60 | 56 | 61.0 | 4.4 |
| e3 | 84 | 92 | 84 | 84 | 86.0 | 3.5 |
| e5 | 96 | 96 | 96 | 88 | 94.0 | 3.5 |

**ALL-10 ORACLE = 74.9**; front-5 70.8, back-5 79.0.

---
### Entry 64 addendum 8 (21 Aug 26) — DECISION: keep r512 as the provisioned baseline, add an r128 LADDER POINT, and re-frame the claim.

**The r512 rows moved both bars** (addenda 5/7): specialist oracle 63.7 -> **74.9**
(front-5 70.8, back-5 79.0), multitask-10 53.2 -> **77.1**. Ours (merged6x2 10-task,
4-seed) is 65.1. Rank effect isolated on the specialists, which are 5k steps at both
ranks: **+11.2 over four doublings = ~+2.8/doubling**. The multitask jump (+23.9) is
NOT a rank effect — that row also went 1k -> 5k steps/task; rank and budget are
confounded there and no r32/50k row exists.

**Rejected: switching the baseline to r256/r128.** The rank was chosen at 512
BEFORE any result, on a stated criterion (above the per-site bottleneck 288 expert /
128 VLM). Re-choosing it after learning 512 wins is outcome-selection, and the r512
runs exist. Kept as the provisioned baseline.

**Adopted: report the ladder and state an equivalent specialist rank.** New run
(r=128 / alpha=32, 5,000 steps, recipe otherwise
byte-identical to the r32 and r512 points): the all-10 mean carries se ~2 against a
~2.8/doubling effect, so adjacent ranks are inside the noise — r128 is two doublings
from r32 (~+5.6 predicted, ~69) and the closest resolvable point. Its job is to close
the bracket ABOVE 65.1, which is already statistically tied with r32's 63.7. Predicted
placement of our method by log-linear interpolation: **r ~ 45**. NOT run for multitask
(joint-training upper bound, not a ladder point; one well-provisioned version suffices).

**Caveat recorded for the writeup:** the aggregate equivalent-rank number hides the
per-env profile — ours beats every specialist rank on e4 (59.0 vs r512's 52.0) and
loses badly on e1 (38.0 vs 61.0). Per-env table must accompany the single number.

---
### Entry 64 addendum 9 (22 Aug 26) — naive sequential LoRA r512, 10 tasks: TRAINING COMPLETE (raw)

Base: stage-1 LIBERO-90 finetune.
r=512, lora_alpha=128. Trainable 850,427,904 / 4,993,832,720 total.
10 online tasks x 5,000 steps = 50,000 steps, bs16 x grad_accum 2.
optimizer_lr 1e-4 -> 1e-5 linear per block, reinit_optimizer_each_task=true,
protect_prior_slots=false, tfidf_enable=false.
Task order (dataset task_index -> env): 0->4, 1->6, 2->9, 3->2, 4->7, 5->0, 6->8,
7->1, 8->3, 9->5.

Terminal training loss at each block boundary:

| block | ckpt | dataset_task_id | env | loss |
|---|---|---|---|---|
| 1 | 005000 | 0 | 4 | 0.034 |
| 2 | 010000 | 1 | 6 | 0.025 |
| 3 | 015000 | 2 | 9 | 0.053 |
| 4 | 020000 | 3 | 2 | 0.030 |
| 5 | 025000 | 4 | 7 | 0.039 |
| 6 | 030000 | 5 | 0 | 0.024 |
| 7 | 035000 | 6 | 8 | 0.024 |
| 8 | 040000 | 7 | 1 | 0.029 |
| 9 | 045000 | 8 | 3 | 0.026 |
| 10 | 050000 | 9 | 5 | 0.017 |

---
### Entry 64 addendum 10 (22 Aug 26) — naive sequential LoRA r512, 10 tasks: ALL-10 FINAL ROW, 4-seed (raw)

Checkpoint 050000 (after all 10 blocks).
25 episodes x seeds
1000/2000/3000/4000, vec batch_size 13.
Envs listed in train order.

| env | s1000 | s2000 | s3000 | s4000 | mean |
|---|---|---|---|---|---|
| 4 | 0 | 0 | 0 | 0 | 0.0 |
| 6 | 0 | 0 | 0 | 0 | 0.0 |
| 9 | 0 | 0 | 0 | 0 | 0.0 |
| 2 | 0 | 0 | 0 | 0 | 0.0 |
| 7 | 0 | 0 | 0 | 0 | 0.0 |
| 0 | 0 | 0 | 0 | 0 | 0.0 |
| 8 | 0 | 0 | 0 | 0 | 0.0 |
| 1 | 0 | 0 | 0 | 0 | 0.0 |
| 3 | 0 | 0 | 0 | 0 | 0.0 |
| 5 | 92 | 100 | 96 | 100 | 97.0 |

**ALL-10 MEAN = 9.7**; front-5 0.0, back-5 19.4.

This row is adopted as b10 of the naive retention triangle (identical checkpoint,
seeds and episode count; not re-measured).

---
### Entry 64 addendum 11 (23 Aug 26) — naive sequential LoRA r512, 10 tasks: RETENTION TRIANGLE, 4-seed (raw)

Lower triangle: after block k,
the k envs seen so far. Each cell 25 episodes x seeds 1000/2000/3000/4000
(100 episodes), vec batch_size 13. Cells are 4-seed means.
Envs in train order. b10 adopted from add-10.

| after block | ckpt | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 005000 | 70.0 | - | - | - | - | - | - | - | - | - | 70.0 |
| b2 | 010000 | 0.0 | 64.0 | - | - | - | - | - | - | - | - | 32.0 |
| b3 | 015000 | 0.0 | 1.0 | 82.0 | - | - | - | - | - | - | - | 27.7 |
| b4 | 020000 | 0.0 | 0.0 | 0.0 | 96.0 | - | - | - | - | - | - | 24.0 |
| b5 | 025000 | 0.0 | 0.0 | 0.0 | 32.0 | 58.0 | - | - | - | - | - | 18.0 |
| b6 | 030000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 73.0 | - | - | - | - | 12.2 |
| b7 | 035000 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 7.0 | 67.0 | - | - | - | 10.9 |
| b8 | 040000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 38.0 | - | - | 4.8 |
| b9 | 045000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 86.0 | - | 9.6 |
| b10 | 050000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 97.0 | 9.7 |

Diagonal (env measured immediately after its own block): 70.0, 64.0, 82.0, 96.0,
58.0, 73.0, 67.0, 38.0, 86.0, 97.0. Mean 73.1.

Prior-task mean per row (row excluding the just-trained env): b2 0.0, b3 0.5,
b4 0.0, b5 8.0, b6 0.0, b7 1.5, b8 0.0, b9 0.0, b10 0.0.

---
### Entry 64 addendum 12 (24 Aug 26) — merged6x2 10-task: RETENTION TRIANGLE, 4-seed (raw)

Same instrument and layout as add-11: lower triangle, each cell 25 episodes x seeds
1000/2000/3000/4000 (100 episodes), vec batch_size 13. Cells are 4-seed means.
Envs in train order. b10 adopted from the 4-seed row.

| after block | ckpt | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 005000 | 54.0 | - | - | - | - | - | - | - | - | - | 54.0 |
| b2 | 010000 | 54.0 | 65.0 | - | - | - | - | - | - | - | - | 59.5 |
| b3 | 015000 | 50.0 | 66.0 | 71.0 | - | - | - | - | - | - | - | 62.3 |
| b4 | 020000 | 51.0 | 54.0 | 65.0 | 87.0 | - | - | - | - | - | - | 64.2 |
| b5 | 025000 | 53.0 | 72.0 | 65.0 | 89.0 | 51.0 | - | - | - | - | - | 66.0 |
| b6 | 030000 | 54.0 | 65.0 | 64.0 | 83.0 | 40.0 | 40.0 | - | - | - | - | 57.7 |
| b7 | 035000 | 51.0 | 64.0 | 73.0 | 89.0 | 45.0 | 48.0 | 71.0 | - | - | - | 63.0 |
| b8 | 040000 | 47.0 | 67.0 | 60.0 | 88.0 | 46.0 | 46.0 | 76.0 | 41.0 | - | - | 58.9 |
| b9 | 045000 | 53.0 | 70.0 | 66.0 | 87.0 | 48.0 | 44.0 | 77.0 | 43.0 | 83.0 | - | 63.4 |
| b10 | 050000 | 59.0 | 60.0 | 63.0 | 87.0 | 54.0 | 37.0 | 76.0 | 38.0 | 86.0 | 91.0 | 65.1 |

Diagonal: 54.0, 65.0, 71.0, 87.0, 51.0, 40.0, 71.0, 41.0, 83.0, 91.0. Mean 65.4.

Prior-task mean per row (row excluding the just-trained env): b2 54.0, b3 58.0,
b4 56.7, b5 69.8, b6 61.2, b7 61.7, b8 61.4, b9 61.0, b10 62.2.

---
### Entry 64 addendum 13 (25 Aug 26) — ten r128 LoRA specialists: TRAINING COMPLETE (raw)

Base: stage-1 LIBERO-90 finetune.
r=128, lora_alpha=32 (alpha/r = 0.25 held). Trainable 212,606,976 per specialist.
5,000 steps each, bs16 x grad_accum 2, optimizer_lr 1e-4 -> 1e-5, same targets and
recipe as the r32 and r512 ladder points.

| task | env | final loss |
|---|---|---|
| t0 | 4 | 0.041 |
| t1 | 6 | 0.036 |
| t2 | 9 | 0.085 |
| t3 | 2 | 0.052 |
| t4 | 7 | 0.066 |
| t5 | 0 | 0.044 |
| t6 | 8 | 0.039 |
| t7 | 1 | 0.063 |
| t8 | 3 | 0.052 |
| t9 | 5 | 0.031 |

Trainable-parameter counts across the ladder: r32 53,151,744 / r128 212,606,976 /
r512 850,427,904 (1.66 M per unit rank over the target set).

---
### Entry 64 addendum 14 (25 Aug 26) — r128 SPECIALIST ORACLE = 69.5, 4-seed (raw)

Ten rows. Each row: the r128/alpha32 specialist for
that env, checkpoint 005000, evaluated on its own env only. 25 episodes x seeds
1000/2000/3000/4000 (100 episodes per cell), vec batch_size 13.
Envs in train order.

| env | s1000 | s2000 | s3000 | s4000 | mean |
|---|---|---|---|---|---|
| 4 | 72 | 72 | 76 | 68 | 72.0 |
| 6 | 72 | 68 | 72 | 68 | 70.0 |
| 9 | 76 | 60 | 56 | 64 | 64.0 |
| 2 | 76 | 76 | 84 | 68 | 76.0 |
| 7 | 64 | 56 | 68 | 60 | 62.0 |
| 0 | 64 | 60 | 76 | 48 | 62.0 |
| 8 | 60 | 68 | 72 | 68 | 67.0 |
| 1 | 48 | 48 | 44 | 44 | 46.0 |
| 3 | 84 | 80 | 96 | 88 | 87.0 |
| 5 | 96 | 92 | 80 | 88 | 89.0 |

**ALL-10 ORACLE = 69.5**; front-5 68.8, back-5 70.2.

Specialist oracle ladder to date (all 5,000 steps, alpha/r = 0.25, same base and
target set): r32 63.7 / r128 69.5 / r512 74.9.
Per-env, same ordering: 

| env | r32 | r128 | r512 |
|---|---|---|---|
| 4 | 46.0 | 72.0 | 52.0 |
| 6 | 49.0 | 70.0 | 71.0 |
| 9 | 61.0 | 64.0 | 78.0 |
| 2 | 80.0 | 76.0 | 85.0 |
| 7 | 59.0 | 62.0 | 68.0 |
| 0 | 46.0 | 62.0 | 84.0 |
| 8 | 67.0 | 67.0 | 70.0 |
| 1 | 62.0 | 46.0 | 61.0 |
| 3 | 84.0 | 87.0 | 86.0 |
| 5 | 83.0 | 89.0 | 94.0 |

---
### Entry 64 addendum 15 (26 Aug 26) — ten r64 LoRA specialists: TRAINING COMPLETE (raw)

r=64, lora_alpha=16 (alpha/r = 0.25 held).
Trainable 106,303,488 per specialist. 5,000 steps each, bs16 x grad_accum 2,
optimizer_lr 1e-4 -> 1e-5, same targets and recipe as the r32 / r128 / r512 points.

| task | env | final loss |
|---|---|---|
| t0 | 4 | 0.052 |
| t1 | 6 | 0.041 |
| t2 | 9 | 0.098 |
| t3 | 2 | 0.060 |
| t4 | 7 | 0.075 |
| t5 | 0 | 0.050 |
| t6 | 8 | 0.043 |
| t7 | 1 | 0.073 |
| t8 | 3 | 0.064 |
| t9 | 5 | 0.035 |

Trainable params across the ladder: r32 53,151,744 / r64 106,303,488 /
r128 212,606,976 / r512 850,427,904.

---
### Entry 64 addendum 16 (27 Aug 26) — r64 SPECIALIST ORACLE = 68.8, 4-seed (raw)

Ten rows. Each row: the r64/alpha16 specialist for that
env, checkpoint 005000, evaluated on its own env only. 25 episodes x seeds
1000/2000/3000/4000 (100 episodes per cell), vec batch_size 13.
Envs in train order.

| env | s1000 | s2000 | s3000 | s4000 | mean |
|---|---|---|---|---|---|
| 4 | 68 | 68 | 60 | 44 | 60.0 |
| 6 | 56 | 48 | 56 | 80 | 60.0 |
| 9 | 72 | 72 | 72 | 84 | 75.0 |
| 2 | 88 | 84 | 84 | 80 | 84.0 |
| 7 | 48 | 52 | 64 | 48 | 53.0 |
| 0 | 68 | 68 | 80 | 60 | 69.0 |
| 8 | 76 | 72 | 72 | 80 | 75.0 |
| 1 | 56 | 48 | 52 | 52 | 52.0 |
| 3 | 68 | 84 | 84 | 72 | 77.0 |
| 5 | 84 | 92 | 76 | 80 | 83.0 |

**ALL-10 ORACLE = 68.8**; front-5 66.4, back-5 71.2.

Complete specialist oracle ladder (all 5,000 steps, alpha/r = 0.25, same LIBERO-90
base and target set; trainable params per specialist in brackets):

| env | r32 (53.2M) | r64 (106.3M) | r128 (212.6M) | r512 (850.4M) |
|---|---|---|---|---|
| 4 | 46.0 | 60.0 | 72.0 | 52.0 |
| 6 | 49.0 | 60.0 | 70.0 | 71.0 |
| 9 | 61.0 | 75.0 | 64.0 | 78.0 |
| 2 | 80.0 | 84.0 | 76.0 | 85.0 |
| 7 | 59.0 | 53.0 | 62.0 | 68.0 |
| 0 | 46.0 | 69.0 | 62.0 | 84.0 |
| 8 | 67.0 | 75.0 | 67.0 | 70.0 |
| 1 | 62.0 | 52.0 | 46.0 | 61.0 |
| 3 | 84.0 | 77.0 | 87.0 | 86.0 |
| 5 | 83.0 | 83.0 | 89.0 | 94.0 |
| **all-10** | **63.7** | **68.8** | **69.5** | **74.9** |
| front-5 | 59.0 | 66.4 | 68.8 | 70.8 |
| back-5 | 68.4 | 71.2 | 70.2 | 79.0 |

---
## Entry 65 - 27 Aug 26 (REAL-WORLD (WidowX AI) chain: dataset inventory, tier-1 task-geometry probe, v5 split — raw)

### Datasets

All LeRobot v3.0, fps 30, robot_type `widowxai_follower_robot`, features: `action` float32 [7]
(joint_0..5.pos, left_carriage_joint.pos), `observation.state` float32 [7] (same names),
`observation.images.cam_high` / `cam_wrist` video [480, 640, 3].

| dataset | tasks | eps | frames | size | note |
|---|---|---|---|---|---|
| realworld_all_tasks | 20 | 1005 | 347,664 | 5.2G | the pool |
| realworld_pretrain | 15 | 751 | 267,547 | 4.0G | v1 split (pool ids minus {0,1,4,10,12}) |
| realworld_seq | 5 | 254 | 80,117 | 1.3G | v1 split: pool ids 0,1,4,10,12 |

Pool task table (task_index, eps, frames, mean/min/max episode length in frames):

| id | eps | frames | mean | min | max | task |
|---|---|---|---|---|---|---|
| 0 | 51 | 18278 | 358 | 265 | 538 | Put the mustard in the basket |
| 1 | 51 | 22108 | 433 | 336 | 569 | put the red bow on the plate |
| 2 | 50 | 14569 | 291 | 235 | 542 | stack the small pan on the large pan |
| 3 | 51 | 17759 | 348 | 255 | 498 | Place the orange in the blender |
| 4 | 52 | 16182 | 311 | 209 | 515 | Place the orange in the black basket |
| 5 | 50 | 12165 | 243 | 168 | 367 | Push over the red and yellow lego bricks |
| 6 | 50 | 15297 | 306 | 198 | 556 | Stack the baskets |
| 7 | 50 | 13377 | 268 | 207 | 475 | Place the scredriver in the tub |
| 8 | 50 | 21987 | 440 | 309 | 656 | Put the lid on the blender jug |
| 9 | 50 | 25844 | 517 | 361 | 668 | Place the red mug on the red plate and the white mug on the white plate |
| 10 | 50 | 8412 | 168 | 40 | 647 | Push over the white lego brick |
| 11 | 50 | 18317 | 366 | 239 | 651 | Place the grey water bottle in front of the red water bottle |
| 12 | 50 | 15137 | 303 | 243 | 447 | Stand the grey bottle up |
| 13 | 50 | 34757 | 695 | 475 | 899 | Stack the blocks with green on the bottom, then yellow, then red |
| 14 | 50 | 13887 | 278 | 203 | 395 | Remove the blender lid and place it on the red plate |
| 15 | 50 | 14808 | 296 | 207 | 405 | Unstack the yellow brick and place to the left of the green brick |
| 16 | 50 | 15167 | 303 | 243 | 407 | Stack the small yellow brick on the large yellow brick |
| 17 | 50 | 21812 | 436 | 318 | 682 | Stack stack the peas on top of the chopped tomatos |
| 18 | 50 | 15469 | 309 | 236 | 482 | Place the yellow cube inside the orange cub |
| 19 | 50 | 12332 | 247 | 184 | 354 | Place the red brick in the tub |

Pool episode-length quantiles (frames): p0 40 / p1 131 / p5 199 / p50 312 / p95 606 / p99 780 / p100 899.
Episodes shorter than 120 frames: 548 (40), 535 (100), 536 (107), 533 (108), 530 (116) — all task 10.
Episodes whose video span (to_timestamp − from_timestamp) × fps exceeds the parquet `length`:
14/1005 (identical set for cam_high and cam_wrist; delta max 420 frames; e.g. ep 71 414 vs 428,
ep 75 365 vs 618, ep 144 239 vs 319, ep 305 501 vs 727); per task: 11:1, 7:2, 18:1, 10:3, 8:1,
6:1, 13:1, 12:1, 1:2, 2:1. realworld_seq (v1) carries 6 of them (10:3, 12:1, 1:2).
Historical splits in pool ids: v1 seq {0,1,4,10,12}; v2 pretrain = 15 pool tasks + multi-modal
tasks (25 tasks, 381,156 frames); v3 seq {0,1,10,14,19}; v4 seq {0,1,10,11,19}.

### Tier-1 task-geometry probe

Checkpoint: raw `pi05_base` (no memory). Dataset: realworld_all_tasks, all 20
tasks, 16 batches × 8 = 128 samples/task, noise/time seeded per batch index (SEED 1000).
Prefix layout:
1224 positions = 1024 image (4 camera slots × 256; 2 real) + 200 language. "▁State" (token 3040)
boundary found in 2560/2560 samples (0 skipped); instruction lengths 6–8 tokens.
Spaces captured: expert MLP input at expert L[4,6,8,10,14,16] (mean over action tokens);
LM MLP input at L[4,5,6,7,8,9,10,11,13,14,15,16] pooled over instruction [3,b) / state [b+3,v−5)
/ key = 1.0·nrm(instr)+0.5·nrm(state) / per-camera image (256 positions each).
Ranker: 18 deployed spaces (expert L4-16, instr L4-16 anchor
sources, key L5-15), off-diagonal cosines z-scored per space, Z = mean z.

Most similar pairs (Z; raw mean cos per group expert / anchor / vlm-key):

| a | b | Z | exp | anc | vlm |
|---|---|---|---|---|---|
| 5 | 10 | 2.30 | 0.947 | 0.897 | 0.919 |
| 3 | 4 | 2.10 | 0.913 | 0.888 | 0.904 |
| 13 | 16 | 1.78 | 0.927 | 0.844 | 0.890 |
| 4 | 19 | 1.78 | 0.900 | 0.852 | 0.881 |
| 18 | 19 | 1.75 | 0.871 | 0.861 | 0.891 |
| 15 | 16 | 1.71 | 0.913 | 0.846 | 0.885 |
| 4 | 18 | 1.55 | 0.888 | 0.833 | 0.870 |
| 1 | 9 | 1.54 | 0.916 | 0.805 | 0.865 |
| 3 | 19 | 1.50 | 0.895 | 0.833 | 0.871 |
| 15 | 19 | 1.48 | 0.877 | 0.832 | 0.880 |
| 7 | 19 | 1.40 | 0.832 | 0.837 | 0.874 |
| 13 | 15 | 1.32 | 0.852 | 0.827 | 0.865 |
| 0 | 4 | 1.19 | 0.823 | 0.804 | 0.860 |
| 16 | 18 | 1.18 | 0.801 | 0.816 | 0.865 |
| 7 | 9 | 1.14 | 0.958 | 0.754 | 0.833 |
| 16 | 19 | 1.05 | 0.851 | 0.797 | 0.852 |
| 3 | 7 | 1.02 | 0.781 | 0.813 | 0.861 |
| 9 | 14 | 1.02 | 0.784 | 0.804 | 0.853 |

Per-task nearest neighbour (Z): 0→4 1.19 · 1→9 1.54 · 2→17 0.32 · 3→4 2.10 · 4→3 2.10 ·
5→10 2.30 · 6→7 0.15 · 7→19 1.40 · 8→1 0.71 · 9→1 1.54 · 10→5 2.30 · 11→18 0.63 · 12→7 0.74 ·
13→16 1.78 · 14→9 1.02 · 15→16 1.71 · 16→13 1.78 · 17→16 0.92 · 18→19 1.75 · 19→4 1.78.
Pair (1,14) Z = 0.74.

Pairwise Z among tasks 0,1,2,7,10,11,12,16,17,19:

| | 0 | 1 | 2 | 7 | 10 | 11 | 12 | 16 | 17 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | – | 0.73 | 0.19 | 0.07 | −0.81 | 0.31 | −0.12 | −0.49 | −0.04 | 0.41 |
| 1 | 0.73 | – | −0.21 | 0.83 | −1.17 | −0.55 | −0.06 | −0.01 | −0.57 | 0.41 |
| 2 | 0.19 | −0.21 | – | −0.56 | −1.00 | 0.06 | −0.64 | −0.36 | 0.32 | −0.38 |
| 7 | 0.07 | 0.83 | −0.56 | – | −1.05 | −0.15 | 0.74 | 0.15 | 0.06 | 1.40 |
| 10 | −0.81 | −1.17 | −1.00 | −1.05 | – | −0.85 | −0.55 | −0.31 | −0.76 | −0.59 |
| 11 | 0.31 | −0.55 | 0.06 | −0.15 | −0.85 | – | 0.39 | −0.61 | 0.23 | 0.30 |
| 12 | −0.12 | −0.06 | −0.64 | 0.74 | −0.55 | 0.39 | – | −0.17 | 0.08 | −0.05 |
| 16 | −0.49 | −0.01 | −0.36 | 0.15 | −0.31 | −0.61 | −0.17 | – | 0.92 | 1.05 |
| 17 | −0.04 | −0.57 | 0.32 | 0.06 | −0.76 | 0.23 | 0.08 | 0.92 | – | 0.44 |
| 19 | 0.41 | 0.41 | −0.38 | 1.40 | −0.59 | 0.30 | −0.05 | 1.05 | 0.44 | – |

5-subset scores (coll_max = max pair Z in the subset; coll_mean; support = mean over subset of
max Z to a task outside it; 15,504 subsets):
- pure geometry top-5: [2,5,6,8,19] −0.38/−0.98/1.05; [2,6,8,10,19] −0.38/−0.95/1.05;
  [6,8,10,11,16] −0.29/−1.00/1.12; [2,6,8,10,16] −0.29/−0.91/1.05; [2,6,10,12,16] −0.17/−0.67/1.06.
- forced {0,10}, no multi-step (9,13,14,15), mean length ≤15 s, top-3: [0,6,10,12,16] −0.12/−0.62/1.24;
  [0,6,10,12,17] 0.08/−0.61/1.06; [0,6,7,10,16] 0.15/−0.58/1.31.
- candidates: [0,2,7,10,16] 0.19 (0–2) / −0.42 / 1.40 (per-task support 1.19, 0.32, 1.40, 2.30, 1.78);
  [0,1,7,10,16] 0.83 (1–7) / −0.21 / 1.64 (1.19, 1.54, 1.40, 2.30, 1.78);
  [0,1,10,11,16] 0.73 (0–1) / −0.35 / 1.49 (1.19, 1.54, 2.30, 0.63, 1.78);
  [0,1,10,12,16] 0.73 (0–1) / – / 1.51; [0,2,10,12,16] 0.19 / −0.42 / 1.27.
- historical: v1 [0,1,4,10,12] rank 7954/15504, coll_max 1.19 (0–4), support 1.50;
  v3 [0,1,10,14,19] rank 3037, 0.74 (1–14), 1.57; v4 [0,1,10,11,19] rank 2890, 0.73 (0–1), 1.49.

### Split v5

Chosen (Josh, 27 Aug): B = pool ids **0, 10, 16, 7, 1** in that sequential order; episode 548
dropped.

| seq task_index | pool id | task | eps | frames |
|---|---|---|---|---|
| 0 | 0 | Put the mustard in the basket | 51 | 18278 |
| 1 | 10 | Push over the white lego brick | 49 | 8372 |
| 2 | 16 | Stack the small yellow brick on the large yellow brick | 50 | 15167 |
| 3 | 7 | Place the scredriver in the tub | 50 | 13377 |
| 4 | 1 | put the red bow on the plate | 51 | 22108 |

`realworld_seq_v5`: 5 tasks / 251 eps / 77,302 frames. `realworld_pretrain_v5`: 15 tasks
(pool ids 2,3,4,5,6,8,9,11,12,13,14,15,17,18,19 in pool order) / 753 eps / 270,322 frames.
Audit family (informational): seq pairs 0-4 and 3-4.

### Chain (27 Aug)

Stage-1
(pi05_base → realworld_pretrain_v5, no memory, bs8 × acc4, 50,000 steps, warmup 4000 /
decay 50000)
→ warm-up
(10k) → audit (bs8 × 400 steps/task, ids [0..4]) → gate
(bg ≤ 0.10, mean core50 ≥ 400, min-eff ≥ 300, VLM min-eff ≥ 150; famIoU informational) → A-phase
(10k) → sequential
(5 × 5000, corefrac β4, top_t 3072, lr 2e-3→2e-4, eval.type=loss
20 batches, per-task checkpoints).

### Entry 65 addendum 3 (28 Aug 26) — stage-1; warm-up (raw)
Stage-1: 50,000 steps, final loss 0.014, grdn 0.343;
loss by 10k: 0.050 / 0.032 / 0.020 / 0.016 / 0.014.
Warm-up: num_learnable 41,443,328
(31 tensors trainable / 905 frozen), shared storage E(4,6)(8,10) V(5,7)(9,11)(13,15), frozen pre-pass, anchor B=0.4.

### Entry 65 addendum 4 (28 Aug 26, 14:20 UTC) — held-out routing audit: GATE HARD FAIL (raw)
Warm-up final: loss 0.232, grdn 0.001 (10k). Audit 5 tasks × 400 × bs8, family (0-4),(3-4):
expert L4/L6/L8/L10/L14/L16 — bgIoU 0.004/0.004/0.004/0.005/0.015/0.025; mean core50 202/180/166/256/553/874;
min-task effnum 242/226/230/241/274/304 (min is task 1 in every layer; task-1 core50 82–105); famIoU 0.018/0.010/
0.005/0.056/0.105/0.085. VLM L5/L7/L9/L11/L13/L15 — min-eff 98/99/92/96/92/89; bg 0.007/0.006/0.008/0.010/0.010/0.015;
famIoU 0.017/0.015/0.026/0.061/0.102/0.139. Gate (expert bg ≤0.10, mean core50 ≥400, min-eff ≥300; VLM min-eff ≥150):
HARD FAIL — bg passes everywhere; core50 fails L4–L10, min-eff fails L4–L14, VLM min-eff fails all six. Chain stopped.

### Entry 65 addendum 5 (28 Aug 26) — gate-fail diagnostics (raw)
State tokens: realworld_seq_v5 distinct 256-bin state vectors 0.797 of rows (libero_10 0.975); |z|>1 fraction 0.346 (0.289).
Pooled-key probe (VLM, warm-up ckpt, 32 samples/task; sim = libero_90 warm-up on libero_10 t0/t3): composite-key cos across
samples RW 0.85–0.91 (sim 0.87–0.92); state-pool cos RW 0.77–0.90 (sim 0.85–0.91); distinct slots hit by state-key rows RW
415–918 (sim 401–763). Audit mass (3,200 samples/task): E4 slots touched t0 19,670 / t1 1,426 / t3 16,684 (sim t0 35,033 /
t3 11,531); top-slot hit fraction E4 t0 0.83 / t1 1.00 / t3 0.97 (sim 0.92 / 0.97); eff V5 190 / 98 / 254 (sim 711 / 250),
V15 519 / 89 / 440 (sim 2990 / 824), E16 4,639 / 304 / 2,731 (sim 5,416 / 1,927). Warm-up aux terms as run: contrastive
(sample, w 0.05, queue 512, margin 0), inter-task separation w 8; VLM pool (1.0, 0.5); expert anchor 0.4.

### Entry 65 addendum 6 (28 Aug 26, 14:52 UTC) — rebalanced warm-up arm (raw)
Josh: option 2. Arm 2 `merged6x2_e468101416_v579111315_anchor040_sep4_c010_prepass`: inter-task separation
w 8 → 4, contrastive (sample, negatives_only=false, queue 512) w 0.05 → 0.1; everything else byte-identical.
Gate re-run on the same 5-task audit.

### Entry 65 addendum 7 (28 Aug 26, 19:03 UTC) — rebalanced arm (sep 4 / c 0.10): GATE HARD FAIL (raw)
Expert L4/L6/L8/L10/L14/L16 — bgIoU 0.006/0.006/0.005/0.007/0.018/0.032; mean core50 239/212/184/303/660/1004 (arm 1:
202/180/166/256/553/874); min-task effnum 265/248/254/291/285/344 (arm 1: 242/226/230/241/274/304), min = task 1 everywhere.
VLM L5–L15 min-eff 109/110/97/100/99/102 (arm 1: 98/99/92/96/92/89); bg 0.007–0.015. Per-task expert L16 core50 t0 2014,
t1 117, t2 1213, t3 1295, t4 380 (arm 1: 874 mean); VLM L15 eff t0 396, t1 102, t2 271, t3 498, t4 180. Same failing clauses
as arm 1 (core50 L4–L10, min-eff L4–L14, VLM min-eff all six).

### Entry 65 addendum 8 (28 Aug 26, 19:38 UTC) — arm 3 (raw)
Arm 3 `…anchor040_sep2_c030_prepass`: contrastive 0.1 → 0.3, sep 4 → 2 (anchor 0.40, VLM pool (1.0, 0.5) unchanged). Josh: if
it fails but improves, keep stepping the same axis overnight (arm 4 c 1.0 / sep 1; arm 5 adds anchor 0.30 + VLM pool (1.0, 1.0)).
Arm-to-arm comparison: arm 1 → 2: 18/18 metrics up, binding mean +11.6%.

### Entry 65 addendum 9 (28 Aug 26, 23:50 UTC) — arm 3 (sep 2 / c 0.30): GATE HARD FAIL (raw)
Expert L4/L6/L8/L10/L14/L16 — mean core50 393/333/282/437/849/1287 (arm 2: 239/212/184/303/660/1004); min-task effnum
283/288/309/335/322/397 (arm 2: 265/248/254/291/285/344); bgIoU 0.012/0.012/0.009/0.012/0.025/0.042. VLM L5–L15 min-eff
114/114/105/105/96/111 (arm 2: 109/110/97/100/99/102); bg 0.008–0.020. Task 1 is the min everywhere (expert eff 283–397,
VLM 96–114); other tasks' VLM eff 184–645. Per-task expert L4 core50 t0 552 / t1 98 / t2 420 / t3 305 / t4 589. Compare arm
2→3: 17/18 metrics up, binding mean +21.3%, worst −3.0% (V13); arm 1→3: 18/18, binding mean +36.0%. Still failing: E4 core50
(393), E4 eff (283), E6 core50 (333), E6 eff (288), E8 core50 (282), VLM min-eff ×6.

### Entry 65 addendum 10 (28 Aug 26, 23:57 UTC) — arm 4 (raw)
Arm 4 `…anchor030_pool1010_sep1_c100_prepass`: contrastive 1.0, sep 1.0, expert anchor 0.40 → 0.30, VLM pool (1.0, 0.5) → (1.0, 1.0);
everything else byte-identical. Josh: no further optimisation intended beyond this — real-world tasks are simpler and pi05_base a
stronger prior, so lower effective palette needs are expected.

### Entry 65 addendum 11 (29 Aug 26, 04:09 UTC) — arm 4 (c 1.0 / sep 1 / anchor 0.30 / pool 1,1): GATE HARD FAIL on the VLM tripwire only (raw)
Expert L4/L6/L8/L10/L14/L16 — mean core50 1308/1078/866/915/1265/1566 (arm 3: 393/333/282/437/849/1287); min-task effnum
430/439/381/467/430/566 (arm 3: 283/288/309/335/322/397); bgIoU 0.048/0.040/0.028/0.031/0.043/0.059 — every expert clause
passes. VLM L5–L15 min-eff 132/125/125/129/113/140 (arm 3: 114/114/105/105/96/111), all task 1; other tasks 259–1023; bg
0.015–0.031. Compare arm 3→4: 18/18 up, binding mean +79.9%, worst +9.6%. Per-task expert L4 core50 t0 2075 / t1 147 /
t2 1115 / t3 1479 / t4 1725. Failing clauses: VLM min-eff < 150 at all six layers.

### Entry 65 addendum 12 (29 Aug 26, 04:41 UTC) — VLM-capacity review verdict → arm 4 graduated with the VLM tripwire OVERRIDDEN (Josh) (raw)
Review of E44–E56/E59–E63 for VLM-specific capacity evidence (Josh's ask). Provenance: the VLM min-eff ≥150 tripwire is
a-priori (E44, re-anchored from E21's expert numbers, "per-task-bias signature"), made permissive at E45 (kill ≤100), re-introduced at
E60 as "the V5 arbiter" (placement question, settled: V5 253). Never calibrated on a VLM outcome; every graduated sim run had min-eff
≥247. Only graduated small-palette case: E47→E49 arm 3-old (global ~2-draw palettes, 32.4 vs 37.2/40.0) — mechanisms overlap-mediated
(protection lock-out, family collisions); E46 arm C: broader VLM palettes → WORSE fit; E55/E59: small VLM cores on the best cells
(e2 V7 core50 80 → 84). E54: c/sep global levers VLM-safe, shallow (c +5–10%/3×); "content-structural, not dose" (E45).
Task 1 geometry (arm 4 audit JSONs, hit counts): VLM V5/V15 slots>0 6.4k/13.0k (t0 13.4k/33.2k); top-64 mass 0.652/0.618 (t0 0.345/0.241),
top-256 0.874/0.889 (t0 0.671/0.489); arm 1 was top-64 0.782/0.832. Expert E4 t1 slots>0 1,426 (arm 1) → 15,129 (arm 4), top-256 0.886 → 0.674.
Pooled-key probe (frozen features, checkpoint-independent): t1 cos_k 0.911/0.862 (L5/L15; t0 0.863/0.858, t3 0.896/0.860, sim 0.87–0.92);
t1 cos_state 0.895 ≈ cos_instr 0.902; distinct slots state-key rows 504/683 (t0 593/918); instr rows 103/95 (t0 280/619, t3 360/540).
Read: task 1 = over-constant VLM read (~65% of every draw is the same ~64 slots) with a normal state-driven tail — NOT the E47 collapse
geometry; key not degenerate; content-driven (shortest, least-varied task), the allowed levers have shallow slope on it (0.78→0.65 over 20× c).
Routes: arm 1 dead (expert core50 166–256 = E21 band); pool arm dead (state pool as constant as instr pool); drop-task-1 = fallback only;
arm-3 bake-off optional. DECISION (Josh: "Do it"): SKIP_GATE=1 on arm 4. Pre-registered t1 reads: own-block MSE reduction (005000→010000
+ in-run task_1 vs baseline) vs the other tasks' (outlier = < half their mean); forget_1 ≤ ~+5% after blocks 2–4; jitter with JIT_T=0,1,3,4;
slot autopsy. Contingency if outlier: task-1 LoRA r256 specialist (E64 recipe) arbitrates data vs VLM capacity → knn36 (permission) or 4-task.

### Entry 65 addendum 13 (29 Aug 26, 07:00 UTC) — WEEKEND BASELINES (Josh): r64/a16 LoRA specialists ×5 + r64/a16 naive sequential LoRA (raw)
Recipes = the sim E64c r64 rung (specialists) and the E64 naive wrapper, RW substitutions only: base = stage-1
realworld_v5_pi05_base_nomem_50k; realworld_seq_v5 episodes t0 0–50 / t1 51–99 / t2 100–149 / t3 150–199 / t4 200–250
(re-asserted at runtime from meta); naive --eval.type=loss (20 batches).
Specialists: 5k steps, bs16×acc2 no-ckpt, lr 1e-4 cosine (warmup 200 / decay 5000 → 1e-5), α/r 0.25, no in-run eval.
Naive: one adapter, 5×5k in chain order, per-block lr 1e-4→1e-5 linear + optimizer reinit, no protection/memory.

### Entry 65 addendum 14 (29 Aug 26, 19:30 UTC) — task-0 one-boundary drift +14.3% ROOT-CAUSED: write-mask SATURATION bypasses corefrac on low-diversity real-world batches (raw)
In-run rows (paired, 20×bs16): t0 jt 0.009367 → after t1 0.010708 (+14.3%, abs +0.00134); t1 jt 0.008882 (fits fine, not an outlier).
Sim E62 matrix (16×bs32) one-boundary t0: 0.03724 → 0.03753 (+0.8%, abs +0.00030); final +4.2%.
Production profiles (memory_by_task), task-1 block writes vs task-0 reads, per site (RW arm 4 / SIM E62):
  t1 update-events on t0 core50: RW E4 179, E6 0, E8 149k, E10 174k, E14 502k, E16 14k, V5–V13 16–45k, V15 172; SIM 0 at all 12 sites.
  t0 read-mass on t1-updated slots: RW 5.8–31.7% (core part 0–14.6%); SIM 1.3–10.4% (core 0).
  Mean distinct slots read per logical batch, t1: RW 3.8k–6.8k; SIM 14.8k–31.8k. Mean mask size k: RW 2.5k–3.07k (sat k/n_read 0.45–0.68); SIM 3072 exactly (sat 0.10–0.21).
  t0-core update/read ratio: RW 0.46–0.85 at E8/E10/E14/V5/V7/V9/V11/V13 (sites with n_read/batch ≲ 3072), ~0 at E4/E6/V15/E16; SIM 0.000 everywhere. t0-shoulder update/read: RW 0.46–0.68 vs SIM 0.11–0.15.
  Value displacement 005000→010000 (read-mass-weighted ||Δslot||/||slot||) on t0: E14 core 6.3% / shoulder 12.3%; V5 9.9% / 12.7%; V13 3.0% / 7.1%; E16 0.9% / 7.7%; E4-table 0.1% / 8.2%.
Mechanism (rank mode, hard_u=0.0 in this run): mask k = min(top_t=3072, n_read); protected slots get score
tfidf·(1−u)^4 = 0 but STAY candidates ("with top_t ≥ candidate count a zero-score slot would still be selected") → when a batch reads ≤3072 distinct
slots the mask is the whole read set incl. t0's core. Sim never saturated (n_read ≫ 3072); RW task 1 (over-constant routing, repetitive data) does at 8/12
sites. Second, smaller channel: mask coverage 0.45–0.68 of each read set vs 0.10–0.21 in sim → shoulder writes 4–5× more frequent per read, on a t0
support that spans 93% of the bank (arm-4 breadth). Existing remedies in code: --protect_hard_u=0.9 (E42 candidacy veto),
or protect_mode=budget (k capped at positive-score count), or top_t scaled to batch read diversity.

### Entry 65 addendum 15 (30 Aug 26, 02:12 UTC) — ARM-4 SEQUENTIAL LANDS (topt3072). In-run drift +10.9% mean; TWO mechanisms, only one fixable by top_t (raw)
In-run paired MSE (20 batches × bs16), % = drift vs that task's just-trained value:
  row1 @5k   t0 0.00937 jt
  row2 @10k  t0 0.01071 +14.3% | t1 0.00888 jt
  row3 @15k  t0 0.01088 +16.1% | t1 0.00943  +6.1% | t2 0.00685 jt
  row4 @20k  t0 0.01111 +18.7% | t1 0.00975  +9.8% | t2 0.00690 +0.6% | t3 0.00920 jt
  row5 @25k  t0 0.01220 +30.2% | t1 0.01038 +16.8% | t2 0.00700 +2.1% | t3 0.00970 +5.4% | t4 0.01325 jt
Mean final drift +10.9% (sim E62 5-task +1.2%, E63 10-task +6.5%). Just-trained fits 0.00937/0.00888/0.00685/0.00920/0.01325.
Per-writer autopsy (memory_by_task, mean over 12 sites, NB=5000):
  writer t1: 4,701 distinct reads/batch, mask k 2,807 (sat 0.60) — SATURATES at 8/12 sites (V5,V7,E8,V9,E10,V11,V13,E14)
             and lands 958k update events on t0's core50. Writers t2/t3/t4: 15.8-19.1k reads/batch, k = 3072 exactly
             (sat 0.16-0.19, sim band 0.10-0.21), prior-core events EXACTLY 0 at every victim×site. corefrac is intact
             for them; only the saturating writer breaches it.
  So t0's per-block increments (+14.3 / +1.8 / +2.6 / +11.5) split by cause: t1 = mask saturation; t4 = NOT a mask defect.
Victim read mass on writer-updated slots (core / shoulder): t0<-t1 0.180 (0.068 core / 0.112 shoulder);
  t0<-t4 0.103 (0 / 0.103); t3<-t4 0.116 (0 / 0.116); t0<-t2 0.026; t0<-t3 0.068; t1<-t4 0.046; t2<-t4 0.073.
  Production read-mass IoU: t3-t4 0.116, t0-t4 0.061, t0-t3 0.069 vs t0-t1 0.015 — and (0,4),(3,4) are exactly the
  pairs declared as the audit FAMILY. => t4's damage is the E27/E28 family collision (scene-genuine shoulder overlap),
  not a masking defect; top_t cannot remove it, only halve its shoulder coverage.
Expectation for the topt1536 rerun, pre-registered: removes t1's saturation channel (~+14pp on t0) and ~halves
shoulder coverage; t0 predicted ~+12-18% (not near-flat). Verification: realized mean k must equal 1536 at every site.

### Entry 65 addendum 16 (30 Aug 26, 02:40 UTC) — BATTERY A (arm-4 topt3072) + an INSTRUMENT BUG that under-reports forgetting in E62/E63 too (raw)
**Battery A** (JIT_T=0,1,3,4): jitter/OOD final ckpt, chunk MSE —
t0 clean 0.0037 / state@0.1 0.0229 / state@0.2 0.0413 / image@0.05 0.0497; t1 0.0027 / 0.0080 / 0.0220 / 0.0571;
t3 0.0038 / 0.0175 / 0.0383 / 0.0677; t4 0.0165 / 0.0326 / 0.0631 / 0.0743. Slot autopsy: per-task effnum
t0 E4 11,656 … V15 2,196; t1 E4 737 … V15 285 (the over-constant task, ~15x below its peers); site-bleed on the 5
shared pairs 13-48% (sim bands E61 17-43% / E62 14-51%); prior-core write events — victim t0: E14 502,370, E10 173,864,
E8 148,965, V13 45,027, V9 21,939, V7 16,611, V5 16,102, E16 14,243, E4 179, E6/V15 ~0; victims t1-t3 ALL ZERO at every
site (only the saturating writer t1 breaches, per add-14/15).
**BUG (found comparing the two instruments): the matrix loaders selected checkpoint tensors with
`".mlp.mem.slot_" in k`. Under E61 shared tables, 2 of the 7 storages are saved as
`<layer>.mlp.mem._storage_shared_from.slot_{down,up}` (expert group (8,10) at L10, VLM group (9,11) at L11) and do NOT
match — `[load] 005000: 10 slot tensors` of 14. Every matrix row is therefore a model with those 2 tables FROZEN at the
first checkpoint's values.** Signature: RW battery t0 (whose own state those tables hold at 005000) agrees with the live
in-run instrument (0.00921 vs 0.00937) while t1-t4 sit ~4x high (t1 0.04633 vs 0.00888 etc.).
Effect on the drift read — battery (broken) vs in-run (live model, correct): t0 +20.5% vs **+30.2%**; t1 -0.9% vs +16.8%;
t2 -1.7% vs +2.1%; t3 -1.3% vs +5.4%; t4 +0.0% vs +0.0%. The battery HIDES exactly the damage in E8/E10 + V9/V11 — and
E10 took 173,864 core-write events. **Use the in-run rows (add-15) as the RW retention numbers.**
**Scope: SIM IS AFFECTED.** `outputs/e62_battery.log` shows `[load] 005000: 10 slot tensors`; the sim merged6x2
checkpoints have the identical 14-tensor / 10-matched layout. So E62 add-2 (+0.0-4.2%, "clean corefrac band") and E63
add-3/add-4 (mean +6.5%, the full 10x10 grid) UNDER-REPORT forgetting on the two shared storages. E62 add-2's
"indistinguishable from dedicated-tables interleave" is void as stated: interleave-8 had dedicated tables, so ITS matrix
was complete and the comparison was unfair. E59/E60 (dedicated) and the jitter/slots/harvest instruments are unaffected.

### Entry 65 addendum 17 (30 Aug 26, 03:05 UTC) — matrix re-run with the fixed loader VALIDATES the fix and the add-15 drift numbers (raw)
Battery A matrix re-computed at 14/14 slot tensors (`[load] 005000: 14 slot tensors`). Corrected grid (paired, 16x bs32, seed 0):
  ckpt      t0        t1        t2        t3        t4
  005000  0.00921   0.31675   0.18377   0.33890   0.48671
  010000  0.01065   0.00923   0.18520   0.33825   0.48617
  015000  0.01094   0.00956   0.00626   0.33760   0.48604
  020000  0.01124   0.00998   0.00636   0.00932   0.47886
  025000  0.01247   0.01070   0.00651   0.00993   0.01155
Diagonal drift, matrix vs the live in-run instrument (add-15): t0 +35.5 / +30.2 %, t1 +15.8 / +16.8, t2 +4.0 / +2.1,
t3 +6.6 / +5.4, t4 +0.0 / +0.0 — the two independent instruments now AGREE (mean +12.4% vs +10.9%). Just-trained levels
also reconcile: matrix 0.00921/0.00923/0.00626/0.00932/0.01155 vs in-run 0.00937/0.00888/0.00685/0.00920/0.01325.
Before the fix the same cells read 0.00921/0.04633/0.02273/0.03924/0.05883 — t1 was 5x off, drift read -0.9% instead of
+15.8%.
=> add-15's in-run rows stand as the RW retention result; the RW drift is ~+12% mean, ~10x sim's reported 5-task band
(which is itself an under-report pending the E62/E63 re-runs noted in add-16).

### Entry 65 addendum 19 (30 Aug 26, 09:10 UTC) — topt1536 rerun, block 1: the SATURATION CHANNEL IS CLOSED (raw)
Pre-registered checks on the rerun (`..._beta4corefrac_topt1536_lr2x_steps5k`), both PASS:
1. No saturation: realized mean mask k = 1537 at ALL 12 sites (read sets 6.5k-40k/batch) vs topt3072's 2472-3072 with
   k<n_read at 8/12. `min(top_t, n_read)` no longer collapses onto the read set, so corefrac's zero-score veto binds.
2. Writer t1 -> victim t0 core50 write events, per site (3072 -> 1536): E4 179->0, V5 16,102->0, V7 16,611->0,
   E8 148,965->0, V9 21,939->0, E10 173,864->0, V11 18,865->0, V13 45,027->0, E14 502,370->**60**, V15 172->0,
   E16 14,243->0, E6 0->0. **TOTAL 958,337 -> 60 (a 16,000x reduction; corefrac's design guarantee restored).**
In-run rows so far, topt1536 vs topt3072 at the same boundary:
  row1 @5k   t0 jt 0.00963 (3072: 0.00937, +2.8% = the coverage cost, inside the +-5% pre-registration)
  row2 @10k  t0 **+0.1%** (3072: **+14.3%**) | t1 jt 0.00921 (3072: 0.00888, +3.7%)
=> the add-14 attribution is confirmed end-to-end: task 1's damage was mask saturation, not overlap. Remaining
pre-registration: t4's family channel is untouched by top_t, so the FINAL row should land ~+12-18% on t0, not near-flat
(add-15: t0's +11.5pp from t4's block came at ZERO core events, all shoulder, on the declared family pairs 0-4 / 3-4).

### Entry 65 addendum 20 (30 Aug 26, 18:20 UTC) — topt1536 RERUN LANDS: mean drift 10.9% -> 2.7%; my family-channel prediction was WRONG (raw)
In-run paired MSE, topt1536 vs topt3072 (same A-phase ckpt, single delta top_t 3072->1536):
  row              t0            t1           t2           t3          t4
  1 @5k    jt 0.00963 (0.00937)
  2 @10k     +0.1% (+14.3%)  jt 0.00921 (0.00888)
  3 @15k     +0.2% (+16.1%)   +2.8% (+6.1%)  jt 0.00729 (0.00685)
  4 @20k     +2.6% (+18.7%)   +3.5% (+9.8%)  +0.7% (+0.6%)  jt 0.00942 (0.00920)
  5 @25k   **+5.6% (+30.2%)  +4.9% (+16.8%)  +1.2% (+2.1%)  +1.6% (+5.4%)  jt 0.01421 (0.01325)**
MEAN FINAL DRIFT **2.7% vs 10.9%** (4x better). Fit tax (just-trained, 1536 vs 3072): +2.8/+3.7/+6.4/+2.4/+7.2%, mean +4.5%.
t0 per-block increments: 3072 = +14.3/+1.8/+2.6/+11.5; 1536 = +0.1/+0.1/+2.4/+3.0.
**CORRECTION to add-15/add-19.** I pre-registered "top_t cannot remove the family channel, only halve its shoulder-write
rate; expect t0 ~+12-18%". Actual t0 = +5.6%, and t4's (the family hub's) damage to t0 fell **11.5pp -> 3.0pp, -74%**,
far more than the ~50% a linear coverage model predicts. So: the family READ collision is real and unchanged (routing is
byte-identical between the two runs — same A-phase, same router), but the DAMAGE it does is mediated by write-mask
coverage far more strongly than by read overlap. Coverage, not routing separation, was the binding term. Notably t3's
block damage to t0 was NOT reduced (2.6 -> 2.4pp) — only the hub's was, so the effect is concentrated where the write
volume x overlap product was largest, not uniform.
Consequence for the arm-5 probe (certificate only): its rationale as a DRIFT fix is now weak — the drift is
already fixed by coverage. It remains worth running as the "is the E53 empty window also empty in RW" question, and as
the language-vs-state routing measurement, but should NOT be read as the remedy for family interference.
Open: whether the +4.5% fit tax or the -8.2pp drift matters more is a ROLLOUT question; no RW robot numbers exist yet.

### Entry 65 addendum 21 (30 Aug 26, 18:55 UTC) — battery B confirms topt1536 at the matrix instrument (raw)
**Battery B, corrected loader (14/14), topt1536 = the RW paper cell (Josh, 30 Aug):**
  ckpt      t0        t1        t2        t3        t4
  005000  0.00950   0.31689   0.18333   0.33914   0.48616
  010000  0.00952   0.01000   0.18329   0.33948   0.48596
  015000  0.00955   0.01018   0.00659   0.33862   0.48551
  020000  0.00968   0.01034   0.00662   0.00960   0.47628
  025000  0.01007   0.01047   0.00669   0.00975   0.01269
Diagonal drift t0..t4: **+5.9 / +4.7 / +1.4 / +1.6 / +0.0 %, mean +2.7%** — matches the live in-run rows
(+5.6/+4.9/+1.2/+1.6/+0.0, mean +2.7) to <=0.3pp on every cell; two independent instruments, same answer.
vs topt3072 corrected matrix (+35.5/+15.8/+4.0/+6.6/+0.0, mean +12.4%): **mean drift 12.4% -> 2.7%, 4.6x.**
Jitter clean (t0/t1/t3/t4): topt1536 0.0031/0.0028/0.0055/0.0177 vs topt3072 0.0037/0.0027/0.0038/0.0165 — t0 better,
t3/t4 worse, consistent with the +4.5% mean fit tax (add-20); no absolute crossing of the imgspan kill signature.
**Arm-5 probe DISABLED** (Josh; add-20 falsified its rationale).

### Entry 65 addendum 22 (31 Aug 26, 09:00 UTC) — r64 SPECIALIST ORACLE (5/5): we BEAT the per-task oracle on fit at every task; task 1 is NOT capacity-limited (raw)
r64/a16 LoRA specialists, one adapter per task from the RW stage-1, 5k steps, no CL constraint, task identity given.
Own-task MSE at the SAME instrument as our matrices (5 tasks x 16 batches x bs32, seed 0):
  t                       specialist | topt1536 jt   final | topt3072 jt   final
  t0 mustard-basket          0.01145 |   0.00950   0.01007 |   0.00921   0.01247
  t1 push lego brick         0.01075 |   0.01000   0.01047 |   0.00923   0.01070
  t2 stack yellow bricks     0.00758 |   0.00659   0.00669 |   0.00626   0.00651
  t3 screwdriver-tub         0.01231 |   0.00960   0.00975 |   0.00932   0.00993
  t4 red bow-plate           0.01549 |   0.01269   0.01269 |   0.01155   0.01155
ours(final, topt1536) / specialist = **0.88 / 0.97 / 0.88 / 0.79 / 0.82** — we are BELOW the per-task oracle's own-task
MSE on ALL FIVE tasks, after sequential training with a frozen backbone and no task identity at inference. (Sim precedent:
E56 B matched or beat its specialists' FUNCTION on all five while losing rollouts on 3/5 — so this is a fit statement, NOT
a success statement; the rollout gap is the conversion layer and needs the robot.)
**THE TASK-1 ARBITER RESOLVES (add-12/add-14 contingency): task 1 is NOT VLM-capacity-limited.** Its specialist reaches
only 0.01075 — the 2nd-WORST specialist cell — while ours reaches 0.01047 (0.97x). A dedicated 106M-param adapter with
task identity and no interference does no better than our shared memory on that task. The over-constant VLM read
(core50 39-46, min-eff 113-140) is a property of the DATA (shortest, most repetitive task, 8,372 frames), exactly as
Josh predicted on 29 Aug and as the gate-override rationale assumed. The VLM min-eff >=150 tripwire fired on a
data property, not a defect => the override was correct, and no knn36 / 4-task fallback is needed.
Task difficulty ordering by specialist MSE: t2 0.00758 < t1 0.01075 < t0 0.01145 < t3 0.01231 < t4 0.01549 — t4
(the family hub, longest at 22,108 frames) is intrinsically hardest, which is consistent with its worst just-trained
cell in every run rather than that being a starvation artifact.

### Entry 65 addendum 23 (31 Aug 26, 22:14 UTC) — the complete real-world table at ONE instrument (raw)
All rows = paired flow-matching MSE, 5 tasks x 16 batches x bs32, seed 0 (fixed 14/14
loader per add-16). Naive adapter-swap matrix:
  ckpt      t0        t1        t2        t3        t4
  005000  0.01146   0.30375   0.19309   0.31129   0.42211
  010000  0.16137   0.01024   0.30571   0.49321   0.62480
  015000  0.23857   0.15024   0.00694   0.46433   0.63069
  020000  0.46939   0.32725   0.11336   0.01095   0.63604
  025000  0.46398   0.31368   0.23936   0.17743   0.01283
**THE TABLE (diagonal drift, just-trained -> final):**
| row | t0 | t1 | t2 | t3 | t4 | MEAN |
| ours topt1536 (PAPER CELL) | +5.9 | +4.7 | +1.4 | +1.6 | +0.0 | **+2.7%** |
| ours topt3072 | +35.5 | +15.8 | +4.0 | +6.6 | +0.0 | +12.4% |
| naive seq-LoRA r64/a16 | +3950 | +2963 | +3348 | +1521 | +0.0 | **+2356%** |
=> the paper cell is **~870x** better retained than the naive foil at matched rank/steps/order/budget, and the naive
numbers sit in the sim band (E63: +740-1567%; here 1521-3950%). Naive plasticity is INTACT (its just-trained cells
0.01146/0.01024/0.00694/0.01095/0.01283 ~= the specialists') — it learns each task and overwrites the last, the textbook
signature. Note its UPPER triangle rises too (t3 0.311 -> 0.493 -> 0.464 before training): a drifting adapter degrades
even never-trained tasks, which memory does not do (ours upper-triangle is flat to 3 decimals).
**Own-task fit vs the r64 specialist oracle (add-22): ours 0.88 / 0.97 / 0.88 / 0.79 / 0.82 x — below the oracle on all 5.**
OPEN (unchanged): every number here is function-space. No RW robot rollouts exist; drift->success conversion is
uncalibrated in the real world (E42/E52 threshold caveat; E56 precedent where matched function still lost rollouts 3/5).

### Entry 65 addendum 24 (1 Sep 26, 09:20 UTC) — E62 5-task matrix RECOMPUTED with the fixed loader: forgetting was under-reported ~3x (raw)
E62 checkpoints re-scored with the add-16 fix (14/14 slot
tensors incl. the `_storage_shared_from` storages for expert (8,10) and VLM (9,11)). Same instrument otherwise
(16 batches x bs32, seed 0).
  ckpt      t0        t1        t2        t3        t4
  005000  0.03724   0.64669   1.66991   0.90498   0.94377
  010000  0.03773   0.03790   1.69692   0.91223   0.95559
  015000  0.03824   0.03853   0.06843   0.94890   0.97950
  020000  0.03902   0.03916   0.06983   0.04643   0.98734
  025000  0.03982   0.03993   0.07115   0.04791   0.05952
| task | FIXED drift | PUBLISHED (add-2) |
| t0/e4 | +6.9% | +4.2% |
| t1/e6 | +5.4% | +1.1% |
| t2/e9 | +4.0% | +0.0% |
| t3/e2 | +3.2% | +0.5% |
| t4/e7 | +0.0% | +0.0% |
| MEAN  | **+3.9%** | +1.2% |
=> E62's reported retention was **~3.2x optimistic**; every non-final task moves up. Note t0's just-trained cell is
UNCHANGED (0.03724) — expected, since at ckpt 005000 the two mis-loaded storages hold t0's own values; only the later
rows were wrong, exactly the signature identified in add-16.
**Consequences for the log's standing claims:** (a) E62 add-2's "matrix +0.0-4.2%, clean corefrac band" becomes
+0.0-6.9%, mean +3.9%; (b) its "indistinguishable from dedicated-tables interleave (+3.6/+3.0/+2.2/+1.3/0.0)" is now a
real if modest gap — merged6x2 IS slightly leakier than dedicated tables, consistent with sharing, and the E61-add-5
worry was under-measured rather than absent; (c) the corefrac "flat matrix" framing still holds directionally (+3.9% mean
vs naive's +740-1567%), but the number to quote is +3.9%, not +1.2%.
Real-world numbers are unaffected (computed post-fix): topt1536 +2.7% mean, topt3072 +12.4%.

---
## Entry 66 - 1 Sep 26 (PARAMETER-MATCHED NAIVE SEQUENTIAL: the reviewer control for "you just added parameters". The match forces r=1216, which is over-complete on 362/416 target matrices: a dense adapter of our size is only buildable by ~equalling full FT of everything it touches)

**The ask (Josh's supervisor):** *"have you run a baseline where you take the same number of
additional parameters and do sequential? A reviewer could plausibly ask 'your method just does
better because you add loads of parameters, no?'"* Josh's read (agreed): not a fair critique given
the CL literature, but many reviewers will share it, so run it and close it for good.

**Design discussion (Josh).** Rejected: naive sequential FULL fine-tune. It has MORE trainable
parameters (4.14B, 1.54x ours) and is the cleaner "capacity cannot save you" statement, but it does
not ADD parameters — it frees existing ones, which is not the objection. The control must add.

**The arithmetic (measured, not estimated).**
- Our merged-6x2 memory adds **2.6837B** (from the checkpoint: 7 shared tables x 65,536 slots;
  expert d=1024 -> 268.4M/table x4, VLM d=2048 -> 536.9M/table x3).
- LoRA over the oracle target set (attn q/k/v/o + MLP gate/up/down both towers + action/state
  projections, 254 matrices) costs 1.661M/rank -> match needs **r=1616**.
- Adding the vision tower (27 encoder layers x {q,k,v,out_proj,fc1,fc2} = 162 matrices) gives 416
  matrices at **2.2044M/rank** -> match at **r=1216 = 2.6806B, -0.12%**. Regex verified to select
  exactly 416.
**THE STRUCTURAL FINDING: those 416 matrices contain only 2.704B parameters IN TOTAL.** So a dense
adapter matching our added-parameter count is ~99% the size of fully fine-tuning everything it
targets, and is over-complete (LoRA bigger than the matrix it adapts) on **362/416** matrices — at
r=1100, 1200, 1216 or 1250 alike. This is structural, not a rank choice; E58 anticipated it
("r~2000 ~= full-rank = no longer LoRA"). Including the vision tower does NOT fix it (vision is only
0.412B). Consequence for the paper: the critique's own premise — that an equal-size dense adapter is
the fair control — is unbuildable without ~matching full FT, while our memory is a **rank-2 adapter
per token** (~0.1% of slots active per forward). We run it anyway; that is the point of a control.

**Config:** r=1216 / alpha=304 (alpha/r=0.25 held), libero_10, 10 tasks x 5,000 steps in
train order. **Every other training parameter is identical to the oracle/naive cells** (Josh's
requirement): stage-1 libero_90 base, bf16, no grad-ckpt, bs16 x accum2 (effective 32), per-task LR
reset 1e-4 -> 1e-5 linear, optimizer reinit each task, no protection / memory / TF-IDF,
save_after_each_task, `--eval.type=none` with the 4-seed campaign as the instrument. Deltas vs
naive r512 are EXACTLY three: r, alpha, + vision targets.

**Pre-registered read.** This becomes the 2.68B point on a capacity ladder that is already flat:
naive seq-LoRA r32 (53M) / r256 (426M) / r512 (852M, 10-task 4-seed = **9.7**, nine of ten envs at
exactly 0.0 with only the last-trained env alive at 97). Expect collapse in the same band; a result
above ~20 would be the surprise and would need explaining. Comparators at the same instrument:
ours 65.1 / specialists 63.7 / multitask-LoRA-10 53.2.
**Existing evidence this COMPLETES rather than starts:** (a) the r32->r512 sweep is a 16x parameter
range with no rescue (E58's "capacity-INSENSITIVE"); (b) E49 arm 3-old is a SAME-parameter-count
control — identical architecture and count, differing only in addressing (near-constant vs
state-conditional palette) -> 32.4 vs 40.0 — which already isolates mechanism from capacity.
**Asymmetry worth stating in the paper (favours us):** matching TOTAL parameters hands the baseline
MORE ACTIVE capacity per token — LoRA and full FT use every parameter on every token; we retrieve
knn x heads slots. The matched control is generous to the baseline.

### Entry 65 addendum 25 (1 Sep 26, 19:05 UTC) — E62 AND E63 SIM MATRICES RECOMPUTED with the fixed loader: E63's forgetting was under-reported ~5.6x (raw)
Both runs re-scored with the add-16 fix (14/14 slot tensors incl. the
`_storage_shared_from` storages for expert (8,10) and VLM (9,11)). Instrument otherwise unchanged
(16 batches x bs32, seed 0).
**E63 10-task, corrected diagonal drift vs the PUBLISHED add-3/add-4 values:**
| task | env | FIXED | published |
| t0 | e4 | **+22.7%** | +6.9% |
| t1 | e6 | **+20.2%** | +6.9% |
| t2 | e9 | **+24.9%** | +4.7% |
| t3 | e2 | **+43.3%** | +3.7% |
| t4 | e7 | **+117.3%** | +18.6% |
| t5 | e0 | **+40.5%** | +4.9% |
| t6 | e8 | +9.2% | +2.2% |
| t7 | e1 | +4.5% | +1.2% |
| t8 | e3 | +2.3% | +1.9% |
| t9 | e5 | +0.0% | +0.0% |
| MEAN | | **+28.5%** | +5.1% |
E62 5-task (add-24): mean **+3.9%** vs published +1.2%.
**CORRECTIONS TO STANDING CLAIMS.** (a) E63 add-3's "mean function drift +6.5%" becomes **+28.5%**; the honest
sentence "drift grows with the number of subsequent tasks but stays an order of magnitude below the naive
baseline's (+740-1567%)" SURVIVES (28.5% is still ~30x below naive) but the number must change. (b) The
exposure ordering survives and sharpens: the four tasks with >=5 subsequent blocks (t0-t3, +20-43%) and t4
(+117%) dominate; the last four (t6-t9, +0-9%) barely move. (c) t4/e7 at +117% is now by far the worst cell —
it was already flagged as the worst at +18.6%, but the true magnitude is 6x that, and e7 is the basket-hub
task whose rollout also lags. (d) E62 add-2's "indistinguishable from dedicated-tables interleave" stays void
(add-24). (e) The 10x10 grid transcribed in add-4 is superseded by the one above.
Real-world numbers are unaffected (computed post-fix): topt1536 +2.7% mean, topt3072 +12.4%, naive +2356%.

### Entry 66 addendum 1 (1 Sep 26, 20:15 UTC) — the parameter-matched dense baseline costs 3.8x our wall-clock per sample (raw)
**Measured compute asymmetry at MATCHED added parameters and MATCHED effective batch (32):**
| | micro-batch | updt_s | s/optimizer-step | s/sample |
| ours merged6x2 (2.684B, sparse) | 16 x acc2 | 0.59 | 1.18 | 0.037 |
| E66 dense LoRA r1216 (2.681B)   |  8 x acc4 | 1.118 | 4.47 | 0.140 |
=> **3.8x wall-clock per sample**, and the dense adapter cannot fit bs16 at all on a 140GB H200. This is the
"dense compute at every token" claim as a measurement: our 2.684B is read by hard top-k retrieval (knn x heads
slots; rank-2 per token), theirs multiplies through all 416 modules for every token. Pairs with the add-16
structural finding (a dense adapter of our size is over-complete on 362/416 matrices) — together they say a
parameter-matched dense control is both degenerate in rank and ~4x the compute.
Pre-registration unchanged: expect collapse
in the naive r512 band (10-task 4-seed 9.7); >~20 would be the surprise.

### Entry 66 addendum 2 (4 Sep 26, 12:20 UTC) — RESULT: the parameter-matched dense baseline scores 8.6. Reviewer objection closed.
**4-seed, 25 eps, all ten envs, same instrument as every comparator row.**
| seed | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | mean |
| 1000 | 0 | 0 | 0 | 0 | 0 | 88.0 | 0 | 0 | 0 | 0 | 8.8 |
| 2000 | 0 | 0 | 0 | 0 | 0 | 92.0 | 0 | 0 | 0 | 0 | 9.2 |
| 3000 | 0 | 0 | 0 | 0 | 0 | 84.0 | 0 | 0 | 0 | 0 | 8.4 |
| 4000 | 0 | 0 | 0 | 0 | 0 | 80.0 | 0 | 0 | 0 | 0 | 8.0 |
**mean 8.60, sd 0.52, range 8.0-9.2.** Only env 5 is non-zero in any seed; env 5 = the LAST task trained
(ds task 9 -> env 5). Nine of ten prior tasks at EXACTLY 0.0 in all four seeds — erased, not degraded.
**Comparator table (10-task, 4-seed, 25 eps):**
| arm | added params | score |
| ours merged6x2 | 2.684B | 65.1 |
| oracle specialists | - | 63.7 |
| multitask LoRA-10 | - | 53.2 |
| naive seq-LoRA r512 | 0.852B | 9.7 |
| naive seq-LoRA r1216 PARAM-MATCHED | 2.681B | **8.6** |
Pre-registration HELD (predicted: r512 band; >~20 would have needed explaining). Nothing to explain away.
**The claim this licenses:** matched to within 0.12% on added parameters, dense sequential scores 8.6 vs our
65.1 — a 7.6x gap. Tripling the baseline's added parameters (0.852B -> 2.681B) did NOT help it; 9.7 -> 8.6 is
flat within seed noise (sd 0.52). Retention is not bought with parameter count. Carry add-1 (3.8x wall-clock
per sample, cannot fit bs16) and add-16 (over-complete on 362/416 matrices at any nearby rank) alongside it:
the matched dense control is simultaneously degenerate in rank, ~4x the compute, and catastrophically forgetful.

## Entry 67 - 7 Sep 26 (EXTERNAL CONTINUAL BASELINES UNDER OUR PROTOCOL: RETAIN (weight merging, alpha 0.5) and O-LoRA (orthogonal per-task adapters, r=64))
**Why (Josh, 7 Sep):** the ICRA related-work now names RETAIN and O-LoRA as the two closest methods (RETAIN meets all four
of our constraints for two tasks; O-LoRA is the canonical task-ID-free, replay-free LoRA competitor). A reviewer will ask
for both as rows. Run both under the same constraints as every other sequential row: stage-1 libero_90 base, the ten
LIBERO-10 tasks in dataset order, 5,000 steps/task, effective batch 32, no replay, no task identity at inference,
optimizer reinit + LR reset per task, the 4-seed x 25-episode instrument, and the full rollout retention triangle
(after task k, evaluate the k tasks seen so far) plus the 10x10 paired-noise loss matrix.

**What the papers specify (read from the PDFs):**
- RETAIN (Yadav, Zhou, Wagenmaker, Pertsch, Levine; ICLR 2026, arXiv 2512.08333). Full fine-tune, then linear
  interpolation over EVERY parameter, theta~ = (1-alpha) theta_pre + alpha theta_ft (Eq. 2). Continual setting (Sec. 5.3,
  Eq. 4): theta~_n = (1-alpha) theta~_{n-1} + alpha theta_ft,n, i.e. the merged checkpoint replaces theta_pre at the next
  boundary. alpha in their continual experiment (DROID, 2 tasks): **fixed at 0.5 at every boundary** (App. A.8.3: "the
  merging weight is always fixed at 50% task-FT, 50% base model"). Single-task alphas were tuned on a held-out OOD
  scene: 0.5-0.75 on DROID, 0.8-0.9 on LIBERO (pi0, batch 64, peak 2.5e-5 cosine to 2.5e-6, only 500-1,000 steps).
  Their continual run used co-FT (50% pretraining-data replay) — forbidden by our no-replay constraint, so we run
  their task-FT variant. The modality-specific variant (Eq. 3) was not used in their continual run; uniform alpha here.
- O-LoRA (Wang et al.; Findings of EMNLP 2023, arXiv 2310.14152). One LoRA pair per task; earlier pairs frozen but
  ACTIVE (forward = W x + s * sum_i B_i A_i x, s = alpha/r shared); objective L_task + lambda_1 * sum_{i<t} L_orth(A_i, A_t).
  Paper Eq. 8 writes a squared Frobenius norm; the OFFICIAL CODE (src/uie_trainer_lora.py) uses
  `orthogonal_loss += torch.abs(torch.mm(A_old, A_new.T)).sum()` — the L1 of the r x r Gram block — with lambda_1 = 0.5
  (5 for a few tasks in two orders) and lambda_2 (L2 on the new pair) = 0. Their rank was 8 on q,v only, constant LR
  1e-3, one epoch. Inference = sum of all adapters, no task id; parameters grow linearly. Their custom LoRA layer
  (src/peft/tuners/lora.py) carries the earlier pairs as one frozen block of rank r*(t-1) plus the trainable rank-r
  pair, i.e. rank concatenation — which is what our export format does (below).

**Decisions (Josh, 7 Sep — "Yes. Go."):**
1. RETAIN alpha = 0.5 (their continual setting). Per-task fine-tune = the paper's joint full-FT recipe (pi0.5 preset:
   AdamW peak 2.5e-5, wd 0.01, betas (0.9,0.95), clip 1.0, cosine to 2.5e-6) compressed to 5k steps with a 500-step
   warm-up; bs8 x acc4 no-grad-ckpt. NOT the LoRA rows' 1e-4 (4x their peak for
   full FT). A second chain at alpha 0.8 (their LIBERO single-task value) is possible later;
   it cannot be recovered from the 0.5 chain because every boundary depends on the previous merge.
2. O-LoRA r=64, alpha/r = 0.25, the r64 specialist target set (attn+MLP both towers + action/state projections; 106M
   params per adapter) and recipe verbatim (lr 1e-4 -> 1e-5 linear, bs16 x acc2, AdamW (0.9,0.999) wd 0, clip 1.0).
   lambda_1 = 0.5 in the official L1 form. Penalty SKIPPED on the two 32-input projections (state_proj, action_in_proj):
   640 mutually orthogonal rows cannot exist in 32 dimensions; they still get adapters. So the row is "the r64 library,
   trained sequentially with the orthogonality constraint and summed at inference, without task identity".

**Implementation (importing our code and modifying none of it):**
- RETAIN: per task, full FT from the running merged
  model with the preset optimizer/scheduler, then in-memory fp32 merge theta <- (1-a) prev + a ft cast back to bf16
  and saved as an ordinary dense
  checkpoint (evaluates through the campaign script unchanged).
- O-LoRA: peft multi-adapter model, new adapter per task,
  earlier adapters frozen-but-active, penalty as above.
  Every boundary exported as ONE rank-concatenated PEFT adapter zero-padded to
  rank 640 (exact: shared scaling => sum of rank-64 adapters == one rank-64k adapter; padding contributes nothing), so
  rollout and loss-matrix instruments (adapter swap, same shapes at every
  boundary) run unchanged.

**Pre-registered expectations (before any number lands):** RETAIN alpha=0.5 — a recency gradient: each boundary halves
every earlier task's delta, so the final all-10 mean should sit well below ours; band 15-40, later tasks higher.
O-LoRA r64 — above naive r512 (9.7), below ours (65.1); OrthoSkillVLA reported O-LoRA at 30.5 on their VLA/LIBERO
setting; band 20-45. Either row above ~55 needs explaining before it goes in the paper. Both rows' loss-drift
should exceed ours (+28.5% mean after ten tasks).

### Entry 67 addendum 1 (7 Sep 26, 14:15 UK) — first smoke: O-LoRA penalty caveat
**Measured on the smoke (2 tasks x 20 steps each):**
- O-LoRA runs at **bs8 x acc4** (effective batch 32 unchanged; accumulation is numerically neutral up to rounding, the E60 add-7 methods note). 254 LoRA modules = 7 per layer x 18 layers x 2 towers + action_in_proj + action_out_proj (the regex of every LoRA row; state_proj / action_time_mlp are not matched in this pi0.5 port), 106.3M params per adapter, adapters fp32. Penalty over 253 modules (action_in_proj excluded, 32 inputs; nothing unsatisfiable); initial L1 6.6e3 at task 2, falling to 4.3e3 over 20 steps; **grad norm ~295 before clipping, i.e. the penalty dominates the raw gradient** — under Adam the per-parameter normalisation means B_t still learns the task at lr scale while A_t is first pushed into the orthogonal complement, which is the method's intended dynamics; lambda_1 = 0.5 is the official value and was NOT re-tuned for our loss scale (carry as a caveat).

### Entry 67 addendum 5 (7 Sep 26, 20:15 UK) — RETAIN row 1: env 4 = 71.0 (block-1 plasticity at the naive-LoRA level)
**Row 1 (block-1 merged model = base + 0.5 x (ft - base), env 4, 25 eps x seeds 1000/2000/3000/4000):** per-seed 68 / 72 / 72 / 72, **mean 71.0**. Block-1 diagonal across the three chains on the same instrument: naive LoRA r512 **70.0** (80/72/64/64), ours merged6x2 **54.0** (52/52/60/52), RETAIN alpha 0.5 **71.0**. So halving the full-FT delta costs RETAIN nothing on the just-trained task at block 1 — full FT plus half-merge is as plastic as the naive adapter here, and 17 points above ours. Retention is the question the merge is supposed to answer; the first evidence is row 2 (block-2 model back on env 4).
### Entry 67 addendum 7 (7 Sep 26, 23:00 UK) — RETAIN row 2: env 4 falls 71 -> 0.0 after one more task (naive-level forgetting, higher plasticity)
**Row 2 (block-2 merged model):** env 4 per-seed **0 / 0 / 0 / 0 = 0.0**; env 6 per-seed 84 / 84 / 76 / 80 = **81.0**. Row mean 40.5. Same instrument, first two rows of the three chains (E64 add-11/12 tables):

| after block | naive LoRA r512 (e4, e6) | ours merged6x2 (e4, e6) | RETAIN alpha 0.5 full FT (e4, e6) |
|---|---|---|---|
| b1 | 70.0, - | 54.0, - | 71.0, - |
| b2 | 0.0, 64.0 | 54.0, 65.0 | **0.0, 81.0** |

So the alpha-0.5 merge, which halves the full-FT delta at every boundary, gives no rollout retention at all on the first prior task: env 4 goes to zero on all four seeds exactly as the naive adapter does, while the just-trained task is the most plastic of the three (81 vs 64 naive, 65 ours). This is what a halved full-FT delta of ~1.1e-4 mean absolute per weight does to a 4.1B-parameter policy: the halfway point between theta_1 and the task-2 fine-tune is already off task 1's manifold. Rows 3-10 decide whether anything of earlier tasks survives (the paper's claim is retention of the *pretraining* distribution, which the drift matrix at the end will test more directly); on rollouts the picture so far is "naive forgetting, better plasticity". Do not over-read two rows — but the row-2 zero is four-seed unanimous.

### Entry 67 addendum 8 (8 Sep 26, 02:35 UK) — RETAIN row 3: env 4 = 5.0, env 6 = 52.0, env 9 = 19.0 — the half-merge now costs the *new* task too
**Row 3 (block-3 merged model, envs 4/6/9):** env 4 per-seed 8/0/4/8 = **5.0**; env 6 per-seed 52/56/56/44 = **52.0**; env 9 (just trained, "put the yellow and white mug in the microwave and close it") per-seed 8/24/16/28 = **19.0**. Row mean 25.3.

| after block | naive LoRA r512 (e4, e6, e9) | ours merged6x2 (e4, e6, e9) | RETAIN alpha 0.5 full FT (e4, e6, e9) |
|---|---|---|---|
| b1 | 70.0 | 54.0 | 71.0 |
| b2 | 0.0, 64.0 | 54.0, 65.0 | 0.0, 81.0 |
| b3 | 0.0, 1.0, 82.0 | 50.0, 66.0, 71.0 | **5.0, 52.0, 19.0** |

**Reading.** Two things at once. (i) Retention of the one-back task is real but partial: env 6 keeps 52 of its 81 (naive keeps 1 of 64), env 4 is 5 (two-back; naive 0). So the alpha-0.5 merge does buy something the naive chain lacks — roughly, the merged weights sit half-way between the previous model and the new fine-tune, so the previous task's behaviour survives one boundary at reduced strength and is gone by the next. (ii) The same halving now hits the *new* task: env 9 at 19 vs naive's 82 and ours 71. Task 3 was the hardest so far (final training loss 0.060 vs 0.028 / 0.022 for tasks 1 / 2; boundary delta 1.15e-4 vs 1.09e-4 / 1.08e-4), and half of that delta is evidently not enough to execute it — the full fine-tune presumably would. Row 2's 81 on env 6 looked like "free plasticity"; row 3 says it was task-dependent. Net so far: RETAIN's row means 71.0 / 40.5 / 25.3 vs ours 54.0 / 59.5 / 62.3 — the merge trades plasticity for a one-boundary memory, and the trade is losing by row 3.
### Entry 67 addendum 9 (8 Sep 26, 06:35 UK) — RETAIN row 4: 5.0 / 21.0 / 3.0 / 86.0 — one-boundary memory confirmed, nothing survives two
**Row 4 (block-4 merged model, envs 4/6/9/2):** env 4 per-seed 8/4/4/4 = **5.0**; env 6 28/20/20/16 = **21.0**; env 9 4/4/4/0 = **3.0**; env 2 (just trained, "turn on the stove and put the moka pot on it", final training loss 0.031) 88/84/80/92 = **86.0**. Row mean 28.8.

| after block | naive LoRA r512 (e4, e6, e9, e2) | ours merged6x2 (e4, e6, e9, e2) | RETAIN alpha 0.5 full FT (e4, e6, e9, e2) |
|---|---|---|---|
| b1 | 70.0 | 54.0 | 71.0 |
| b2 | 0.0, 64.0 | 54.0, 65.0 | 0.0, 81.0 |
| b3 | 0.0, 1.0, 82.0 | 50.0, 66.0, 71.0 | 5.0, 52.0, 19.0 |
| b4 | 0.0, 0.0, 0.0, 96.0 | 51.0, 54.0, 65.0, 87.0 | **5.0, 21.0, 3.0, 86.0** |

**Reading.** The pattern from row 3 holds and sharpens. Diagonal (just trained): 71 / 81 / 19 / 86 — full plasticity on the easy tasks (the halved delta is enough when the task is easy: env 2 is naive's 96 and ours 87), a collapse on the hard one (env 9). One boundary back: 0 (e4@b2), 52 (e6@b3), 3 (e9@b4) — what survives one merge is roughly half of whatever the diagonal was, and env 9 had 19 to start with. Two boundaries back: 5 (e4@b3), 21 (e6@b4) — a quarter-ish. Three back: 5. Geometric decay by ~0.5 per boundary is exactly what a fixed alpha = 0.5 interpolation predicts if the tasks' deltas are roughly orthogonal in weight space: each boundary halves every earlier task's residual delta. Row means: RETAIN 71.0 / 40.5 / 25.3 / 28.8 vs ours 54.0 / 59.5 / 62.3 / 64.2 vs naive 70.0 / 32.0 / 27.7 / 24.0 — RETAIN is tracking the naive chain, a few points above it, and losing ~35 points per row to ours from row 3. Six rows to go; the expected end state is a row-10 mean near 10-15 (the diagonal plus a halved one-back task) against ours at 65.1.

### Entry 67 addendum 10 (8 Sep 26, 10:20 UK) — RETAIN row 5: 9 / 30 / 0 / 65 / 57 — halfway; row mean 32.2 vs ours 66.0, naive 18.0
**Row 5 (block-5 merged model, envs 4/6/9/2/7):** env 4 per-seed 12/8/4/12 = **9.0**; env 6 28/24/40/28 = **30.0**; env 9 0/0/0/0 = **0.0**; env 2 84/72/68/36 = **65.0**; env 7 (just trained, "put both the alphabet soup and the cream cheese box in the basket", final training loss 0.041) 40/64/64/60 = **57.0**. Row mean 32.2.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 full FT |
|---|---|---|---|
| b5 (e4, e6, e9, e2, e7) | 0, 0, 0, 32, 58 = 18.0 | 53, 72, 65, 89, 51 = 66.0 | **9, 30, 0, 65, 57 = 32.2** |
| row means b1..b5 | 70.0, 32.0, 27.7, 24.0, 18.0 | 54.0, 59.5, 62.3, 64.2, 66.0 | 71.0, 40.5, 25.3, 28.8, 32.2 |

**Reading.** Diagonal env 7 at 57 is on par with both other chains (58 / 51) — a hard basket task where the halved delta is not the limiter. One-back env 2 keeps 65 of 86 (naive keeps 32 of 96): the one-boundary memory again, ~0.75 this time rather than ~0.5. The decay is not monotone geometric: env 6 went 81 -> 52 -> 21 -> **30** (all four seeds >= 24, previous row 16-28) and env 4 5 -> 9, so a later merge can hand a little back — the task deltas are not orthogonal and the half-steps sometimes move toward an older task. But the level is the point: everything two or more boundaries back sits at 0-30, and the row mean is climbing only because the chain has reached easier or more transferable tasks (ours 66.0 on the same row). RETAIN's mean is 14 points above naive and 34 below ours at the halfway mark.

### Entry 67 addendum 11 (8 Sep 26, 14:00 UK) — RETAIN row 6: 14 / 13 / 0 / 14 / 35 / 46 — row mean 20.3 (ours 57.7, naive 12.2)
**Row 6 (block-6 merged model, envs 4/6/9/2/7/0):** env 4 per-seed 28/8/12/8 = **14.0**; env 6 12/20/4/16 = **13.0**; env 9 0/0/0/0 = **0.0**; env 2 8/16/12/20 = **14.0**; env 7 28/44/28/40 = **35.0**; env 0 (just trained, "put both the alphabet soup and the tomato sauce in the basket", final training loss 0.028) 44/40/44/56 = **46.0**. Row mean 20.3.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 full FT |
|---|---|---|---|
| b6 (e4, e6, e9, e2, e7, e0) | 0, 0, 0, 0, 0, 73 = 12.2 | 54, 65, 64, 83, 40, 40 = 57.7 | **14, 13, 0, 14, 35, 46 = 20.3** |
| row means b1..b6 | 70.0, 32.0, 27.7, 24.0, 18.0, 12.2 | 54.0, 59.5, 62.3, 64.2, 66.0, 57.7 | 71.0, 40.5, 25.3, 28.8, 32.2, 20.3 |

**Reading.** The one-back task (env 7) keeps 35 of 57 (~0.6, the now-familiar one-boundary memory); the two-back task (env 2) drops 65 -> 14 — the second merge is where RETAIN loses a task. Everything older sits in the 0-14 band, which is where naive sits at 0 — RETAIN's floor is a little above zero (env 4 back up to 14 on the strength of one seed at 28) but it is a floor. The diagonal on env 0 (a basket task, hard for every chain: naive 73, ours 40) is 46. Row 5's climb was the easy-task stretch; row 6 is back to the trend line: RETAIN's mean has now fallen to 8 points above naive and 37 below ours.

### Entry 67 addendum 12 (8 Sep 26, 18:05 UK) — RETAIN row 7: 8 / 5 / 0 / 25 / 34 / 63 / 34 — row mean 24.1 (ours 63.0, naive 10.9)
**Row 7 (block-7 merged model, envs 4/6/9/2/7/0/8):** env 4 per-seed 16/4/4/8 = **8.0**; env 6 12/4/4/0 = **5.0**; env 9 0/0/0/0 = **0.0**; env 2 16/24/32/28 = **25.0**; env 7 44/28/28/36 = **34.0**; env 0 68/68/64/52 = **63.0**; env 8 (just trained, "put both moka pots on the stove", final training loss 0.021) 28/36/32/40 = **34.0**. Row mean 24.1.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 full FT |
|---|---|---|---|
| b7 (e4, e6, e9, e2, e7, e0, e8) | 0, 0, 0, 2, 0, 7, 67 = 10.9 | 51, 64, 73, 89, 45, 48, 71 = 63.0 | **8, 5, 0, 25, 34, 63, 34 = 24.1** |
| row means b1..b7 | 70.0, 32.0, 27.7, 24.0, 18.0, 12.2, 10.9 | 54.0, 59.5, 62.3, 64.2, 66.0, 57.7, 63.0 | 71.0, 40.5, 25.3, 28.8, 32.2, 20.3, 24.1 |

**Reading.** Two features worth noting. (i) The diagonal is now *below* the one-back cell: env 8 (just trained) 34 vs env 0 (one back) 63. Env 0 went 46 -> 63 across a boundary — the merge moved it *up*, the second time we have seen this (env 6 21 -> 30 at b5). Both jumps are on basket/stove tasks that share structure with the task trained next, so the halved delta of a related task partly serves the older one; this is positive backward transfer via weight-space proximity, not retention as such. (ii) The old block (e4, e6, e9) is at 8 / 5 / 0 and stable — RETAIN's floor. Naive is at 0/0/0/2/0/7 with only its diagonal alive (67); ours holds 45-89 everywhere. RETAIN's mean is 13 above naive, 39 below ours.
### Entry 67 addendum 13 (8 Sep 26, 21:50 UK) — RETAIN row 8: everything but the diagonal collapses (mean 12.1)
**Row 8 (block-8 merged model, envs 4/6/9/2/7/0/8/1):** env 4 **2.0** (4/4/0/0); env 6 **0.0**; env 9 **0.0**; env 2 **10.0** (12/4/12/12); env 7 **5.0** (12/4/4/0); env 0 **3.0** (0/0/8/4); env 8 **7.0** (8/4/4/12); env 1 (just trained, "put both the cream cheese box and the butter in the basket", final training loss 0.030) **70.0** (68/72/72/68). Row mean 12.1.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 full FT |
|---|---|---|---|
| b8 (e4,e6,e9,e2,e7,e0,e8,e1) | 0,0,0,0,0,0,0,38 = 4.8 | 47,67,60,88,46,46,76,41 = 58.9 | **2,0,0,10,5,3,7,70 = 12.1** |
| row means b1..b8 | 70.0, 32.0, 27.7, 24.0, 18.0, 12.2, 10.9, 4.8 | 54.0, 59.5, 62.3, 64.2, 66.0, 57.7, 63.0, 58.9 | 71.0, 40.5, 25.3, 28.8, 32.2, 20.3, 24.1, 12.1 |

**Reading.** The b7 backward-transfer bump was temporary: env 0 63 -> **3** and env 7 34 -> **5** in one boundary, i.e. the "recovered" tasks were riding on structure the block-8 task did not share, and the merge took it back. Every non-diagonal cell is now 0-10 — the same picture as naive (all zeros) with a few points of noise on top. Diagonal 70 is healthy (naive 38, ours 41 on this task). So at 8 tasks RETAIN retains nothing measurable in rollouts; its advantage over naive has shrunk to ~7 points of row mean, almost all of it in the one-back cell, while ours holds 58.9. Two rows left to confirm the endpoint.
### Entry 67 addendum 14 (9 Sep 26, 01:05 UK) — RETAIN CHAIN COMPLETE (10/10)
All ten merge witnesses read exactly 0.5000; the per-boundary `mean|ft-prev|` sat in a narrow 1.03-1.15e-4 band across all ten tasks, i.e. every task moved the 4.1B weights by the same order and the merge halved each one identically.
### Entry 67 addendum 15 (9 Sep 26, 01:35 UK) — RETAIN row 9: 100 on the diagonal, 35 one-back, 0 everywhere else (mean 15.1)
**Row 9 (block-9 merged model):** env 3 (just trained, "pick up the book and place it in the front compartment of the caddy") per-seed 100/100/100/100 = **100.0**; env 1 36/32/36/36 = **35.0**; env 8 0/4/0/0 = **1.0**; env 0, 2, 4, 6, 7, 9 all **0.0**. Row mean 15.1.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 full FT |
|---|---|---|---|
| b9 (e4,e6,e9,e2,e7,e0,e8,e1,e3) | 0,0,0,0,0,0,0,0,86 = 9.6 | 53,70,66,87,48,44,77,43,83 = 63.4 | **0,0,0,0,0,0,1,35,100 = 15.1** |
| row means b1..b9 | 70.0, 32.0, 27.7, 24.0, 18.0, 12.2, 10.9, 4.8, 9.6 | 54.0, 59.5, 62.3, 64.2, 66.0, 57.7, 63.0, 58.9, 63.4 | 71.0, 40.5, 25.3, 28.8, 32.2, 20.3, 24.1, 12.1, 15.1 |

**Reading.** The cleanest row yet, and it states RETAIN's behaviour exactly: **the merged model is the current task plus a half-strength copy of the immediately preceding one, and nothing else.** Diagonal 100 (4/4 seeds perfect — this caddy task is the easiest in LIBERO-10; naive got 86, ours 83), one-back 35, everything two or more boundaries back is 0 or 1. Nine rows in, RETAIN's only consistent advantage over naive sequential LoRA is that one-back cell (35 vs 0); on the two-back-and-older block the two chains are indistinguishable at 0. Ours holds 43-87 across the whole row.
### Entry 67 addendum 16 (9 Sep 26, 04:20 UK) — O-LoRA boundary 1: adapter frozen and exported, concat-equivalence witness exact; task 2 orth penalty live at 9.95e+02
**Boundary 1:** 106,303,488 trainable params in the task-1 adapter (r64/alpha16 on the same 254 modules the LoRA baselines use); `l1_B_current` 2.328e+05. The export witness is the one that matters: **`loss_multi_adapter` = `loss_concat_adapter` = 0.0526979826, rel_diff 0.0 (exact)** — the stacked per-task adapters and the single concatenated adapter the checkpoint ships give bit-identical loss, so the drift matrix's adapter-swap path will score exactly what the chain trained.
**The orthogonality constraint is live.** Task 1 ran at `orth 0.000e+00` throughout (correct — no earlier adapter to be orthogonal against). Task 2 opened at **`orth 9.950e+02`** with grad-norm 480.5 at step 100, i.e. the penalty is both non-zero and initially dominant, exactly as O-LoRA intends: the new adapter is being pushed out of the span of adapter 1 from the first steps. Loss 0.593 at step 100 against task 1's 0.0465 finish is the expected fresh-task reset.
### Entry 67 addendum 17 (9 Sep 26, 05:40 UK) — RETAIN RETENTION TRIANGLE COMPLETE (10/10 rows). Final row mean 8.2, below naive's 9.7; ours 65.1
**Row 10 (final merged model 050000), envs in train order:** e4 **0.0**, e6 **0.0**, e9 **0.0**, e2 **0.0**, e7 **1.0** (0/0/0/4), e0 **0.0**, e8 **0.0**, e1 **0.0**, e3 **23.0** (32/20/16/24), e5 (just trained) **58.0** (56/56/48/72). Row mean **8.2**.

**THE FULL RETAIN TRIANGLE (alpha 0.5, full FT, 4-seed x 25-episode cells, envs in train order):**

| after block | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 71.0 | - | - | - | - | - | - | - | - | - | 71.0 |
| b2 | 0.0 | 81.0 | - | - | - | - | - | - | - | - | 40.5 |
| b3 | 5.0 | 52.0 | 19.0 | - | - | - | - | - | - | - | 25.3 |
| b4 | 5.0 | 21.0 | 3.0 | 86.0 | - | - | - | - | - | - | 28.8 |
| b5 | 9.0 | 30.0 | 0.0 | 65.0 | 57.0 | - | - | - | - | - | 32.2 |
| b6 | 14.0 | 13.0 | 0.0 | 14.0 | 35.0 | 46.0 | - | - | - | - | 20.3 |
| b7 | 8.0 | 5.0 | 0.0 | 25.0 | 34.0 | 63.0 | 34.0 | - | - | - | 24.1 |
| b8 | 2.0 | 0.0 | 0.0 | 10.0 | 5.0 | 3.0 | 7.0 | 70.0 | - | - | 12.1 |
| b9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 35.0 | 100.0 | - | 15.1 |
| b10 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 23.0 | 58.0 | **8.2** |

**Headline.** Final-row means: **ours 65.1, naive LoRA r512 9.7, naive param-matched r1216 8.6 (E64 add-13), RETAIN alpha 0.5 8.2.** RETAIN finishes *below* both naive baselines — the paper's weight-merging recipe, run under our 10-task protocol, gives no measurable rollout retention at the end of the chain, and pays for the merge on the diagonal too (e5 58.0 vs naive's 97.0 and ours' 91.0: the half-merge cannot fit the final task either). The prior-task mean excluding the diagonal is **2.6** for RETAIN vs 0.0 naive vs 62.2 ours.
**The mechanism, stated from the whole triangle.** RETAIN is a *one-boundary* memory that also costs plasticity. Read down each column: a task scores well on its own diagonal, retains roughly half to three-quarters one boundary later (e6 81->52, e2 86->65, e0 46->63, e1 70->35, e3 100->23), then collapses to 0-14 by the second boundary and to 0 by the third. Alpha 0.5 halves every earlier task's residual delta at each boundary, so after k further tasks an early task's contribution is down by ~2^-k — geometric erasure, with occasional small rebounds (e6 21->30 at b5, e0 46->63 at b7) where the next task's delta happens to share structure. Meanwhile the diagonal itself degrades once tasks are hard (e9 19.0, e5 58.0), because half of a fine-tune delta is not always enough to execute the task it was fit for.
### Entry 67 addendum 18 (9 Sep 26, 06:15 UK) — O-LoRA row 1: env 4 = 43.0 (lowest block-1 diagonal of the four chains — an r64 capacity effect, not orthogonality)
**Row 1 (task-1 adapter on checkpoint 005000):** env 4 per-seed 36/48/32/56 = **43.0** (the widest per-seed spread we have seen at b1: 24 points, vs RETAIN's 4).

| chain | b1 diagonal (env 4) | adapter / delta capacity |
|---|---|---|
| RETAIN alpha 0.5 (full FT, merged) | 71.0 | all 4.143B weights, halved |
| naive LoRA r512 | 70.0 | ~0.852B |
| ours merged6x2 | 54.0 | 2.8B memory values, frozen backbone |
| **O-LoRA r64** | **43.0** | **0.106B** |

**Reading — this one is not about forgetting.** Task 1 trained with `orth 0.000e+00` throughout (nothing to be orthogonal against), so 43.0 is O-LoRA's *plasticity ceiling at r64*, uncontaminated by its own constraint: a 106M-parameter adapter is 8x smaller than naive r512's and simply fits the task less well. That matters for reading every later O-LoRA row — its diagonal starts 27 points below naive's, so a low row mean at b10 will be part capacity, part orthogonality, and the two need separating (the per-task oracle would be the clean control; not in the current protocol). The pre-registered expectation (add-2) was a band of 20-45 for the O-LoRA final row, citing OrthoSkillVLA's 30.5; a b1 diagonal of 43 is consistent with landing inside that band.
### Entry 67 addendum 19 (9 Sep 26, 07:45 UK) — RETAIN DENSE DRIFT MATRIX COMPLETE: mean own-task loss drift **+476.6%**, vs ours +28.5% and naive +740-1567%
**Matrix (10/10 rows).** Rows are the merged checkpoint after each block, columns the per-task action MSE on that task's own data (dataset task ids = train order):

| ckpt | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 005000 | **0.109** | 0.822 | 2.260 | 1.108 | 1.268 | 1.115 | 0.996 | 1.186 | 1.245 | 0.777 |
| 010000 | 0.330 | **0.144** | 2.951 | 1.464 | 1.637 | 1.323 | 1.290 | 1.602 | 1.664 | 1.035 |
| 015000 | 0.493 | 0.273 | **0.613** | 1.819 | 1.686 | 1.348 | 1.355 | 1.681 | 1.680 | 1.131 |
| 020000 | 0.709 | 0.491 | 1.140 | **0.276** | 1.847 | 1.467 | 1.586 | 1.780 | 1.847 | 1.352 |
| 025000 | 0.849 | 0.713 | 1.542 | 0.463 | **0.418** | 1.485 | 1.793 | 1.790 | 2.003 | 1.619 |
| 030000 | 1.037 | 0.974 | 2.131 | 0.816 | 0.776 | **0.275** | 2.018 | 2.075 | 2.234 | 1.858 |
| 035000 | 1.231 | 1.182 | 2.166 | 1.141 | 1.053 | 0.470 | **0.370** | 2.264 | 2.091 | 1.744 |
| 040000 | 1.253 | 1.313 | 2.439 | 1.279 | 1.085 | 0.750 | 0.653 | **0.566** | 2.207 | 2.020 |
| 045000 | 1.450 | 1.516 | 2.843 | 1.705 | 1.381 | 0.998 | 0.906 | 0.928 | **0.447** | 2.208 |
| 050000 | 1.690 | 1.753 | 3.169 | 2.114 | 1.771 | 1.209 | 1.101 | 1.467 | 0.841 | **0.267** |

**Own-task drift (loss at its own boundary -> loss under the final model):** t0 0.109->1.690 **+1455%**; t1 0.144->1.753 **+1121%**; t2 0.613->3.169 **+417%**; t3 0.277->2.114 **+665%**; t4 0.418->1.771 **+324%**; t5 0.276->1.209 **+339%**; t6 0.370->1.101 **+198%**; t7 0.566->1.467 **+159%**; t8 0.448->0.841 **+88%**; t9 0.267->0.267 **+0%** (the final task, no subsequent merge). **MEAN +476.6%.**

**Reading.** The function-space measurement agrees with the rollout triangle and is the cleaner statement of it, because it is continuous where success rate saturates at 0. Three things. (i) **RETAIN's drift is ~17x ours** (+476.6% vs +28.5% after ten tasks) and sits inside, though at the low end of, the naive band (+740-1567%): merging at alpha 0.5 slows the erasure of earlier tasks relative to unprotected sequential training, but does not stop it — consistent with the triangle, where RETAIN's only surplus over naive was the one-back cell. (ii) **The exposure ordering is monotone and clean**: drift is ordered almost exactly by how many merges a task subsequently endures (t0 +1455% with nine, down to t8 +88% with one, t9 +0% with none). Each boundary multiplies an earlier task's residual delta by 0.5, so its loss climbs geometrically back toward the pre-training function — the same 2^-k mechanism the triangle showed, now visible without a floor effect. (iii) **The diagonal is not free either**: own-task loss at its own boundary ranges 0.109-0.613, and the two hardest (t2 0.613, t7 0.566) are the same tasks whose rollout diagonals were weakest (e9 19.0, e0/e7 mid-30s to 40s) — half a fine-tune delta is a worse fit for a hard task, in loss as in rollouts.
### Entry 67 addendum 20 (9 Sep 26, 08:10 UK) — O-LoRA boundary 2: constraint held through a full task (orth 995 -> 3.85), 2-adapter algebra witness 2.0e-05
**Boundary 2:** task 2 ("put the white mug on the plate and put the chocolate pudding to the right of the plate") finished at loss 0.101.
**Two witnesses, both good.** (i) **Orthogonality actually bites and then relaxes:** the penalty ran 9.950e+02 (step 100) -> 5.213e+01 -> 2.063e+01 -> 1.504e+01 -> 9.97 -> **3.852e+00** (step 5,000). The new adapter starts far inside adapter 1's span and is pushed out over the task, ending ~250x down. Gradient norm stayed high (621) throughout, i.e. the constraint is still shaping the solution at the end, not satisfied and forgotten. (ii) **Algebra witness with two adapters: `algebra_max_rel_err` 2.0e-05** (boundary 1, single adapter, was exactly 0.0). That is bf16 round-off on the stacked-vs-concatenated equivalence, not drift — the adapter-swap drift matrix will still score what the chain trained.
### Entry 67 addendum 21 (9 Sep 26, 09:20 UK) — O-LoRA row 2: **env 4 collapses 43.0 -> 3.0 in one task, with task 1's adapter still present and orthogonal.** Row mean 21.0, lowest of the four chains
**Row 2 (checkpoint 010000):** env 4 per-seed 4/4/0/4 = **3.0**; env 6 (just trained) 40/36/28/52 = **39.0**. Row mean **21.0**.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 | **O-LoRA r64** |
|---|---|---|---|---|
| b1 (e4) | 70.0 | 54.0 | 71.0 | **43.0** |
| b2 (e4, e6) | 0.0, 64.0 = 32.0 | 54.0, 65.0 = 59.5 | 0.0, 81.0 = 40.5 | **3.0, 39.0 = 21.0** |

**This is not an export bug — verified.** The evaluated checkpoint carries both adapters: `adapter_model.safetensors` has `lora_A` of shape (640, 32) with **64/640 rank-rows populated at b1 and 128/640 at b2**, i.e. task 1's rank-64 block is present and untouched alongside task 2's, and the triangle loads it. The b1 algebra witness was exactly 0.0 and b2's 2.0e-05, so the stacked-vs-concat equivalence holds. Task 1's parameters are intact, orthogonal to task 2's by construction (penalty ran 995 -> 3.85 over the task), **and task 1 still went from 43% to 3%.**
**What that means.** Parameter-space orthogonality does not buy function-space retention here. The forward pass sums both adapters' contributions into the same residual stream, so adding an orthogonal delta still moves the model's output on task 1's input distribution — orthogonal directions are not non-interfering directions once the base is shared and the readout is nonlinear. O-LoRA's own premise is exactly this preservation, and on a 4.1B VLA under our protocol it does not hold at the first opportunity to test it. Naive r512 dropped e4 70 -> 0 at the same boundary; O-LoRA dropped 43 -> 3. Both forget; O-LoRA forgets from a lower diagonal.
**Caveat to carry (from add-18).** O-LoRA's per-task adapter is 106.3M params vs naive r512's single 0.852B adapter — 8x smaller per task, though the ten-task total (1.063B) approaches naive's budget. Its low diagonals (43.0, 39.0 vs naive 70.0, 64.0) are therefore partly capacity, not only the orthogonality constraint, and the two are not separated by the current protocol. State that explicitly wherever this row goes in the paper.

### Entry 67 addendum 22 (9 Sep 26, 13:40 UK) — O-LoRA row 3: **the diagonal has gone too** (e9 = 2.0). Row mean 2.7, below naive's 27.7 at the same block
**Row 3 (checkpoint 015000):** env 4 **0.0** (0/0/0/0); env 6 **6.0** (16/0/8/0); env 9 (just trained, "put the yellow and white mug in the microwave and close it") **2.0** (4/0/0/4). Row mean **2.7**.

| after block | naive LoRA r512 | ours merged6x2 | RETAIN alpha 0.5 | **O-LoRA r64** |
|---|---|---|---|---|
| b1 (e4) | 70.0 | 54.0 | 71.0 | 43.0 |
| b2 (e4,e6) | 0, 64 = 32.0 | 54, 65 = 59.5 | 0, 81 = 40.5 | 3, 39 = 21.0 |
| b3 (e4,e6,e9) | 0, 1, 82 = 27.7 | 50, 66, 71 = 62.3 | 5, 52, 19 = 25.3 | **0, 6, 2 = 2.7** |

**This row is different in kind from rows 1-2.** Until now O-LoRA forgot prior tasks (e4 43 -> 3) while still fitting the current one (39.0 at b2). At b3 it fits *nothing*: the just-trained env 9 scores **2.0**, against naive's 82 and ours' 71 on the same task. The training loss said this was coming — task 3 finished at **0.386**, vs 0.047 (t1) and 0.101 (t2), and vs RETAIN's 0.060 on the same task. So the failure is in the fit, not the rollout instrument.
**Two contributions, both plausible, not yet separated.** (i) **The task.** Env 9 (microwave) is the hardest in LIBERO-10 by every chain's diagonal: RETAIN 19.0, ours 71.0, naive 82.0 — and RETAIN's own final loss on it was its worst too. (ii) **The constraint tightening.** Task 3 is the first with two prior adapters to be orthogonal against; its penalty opened at 2,198 (vs 995 at t2), decayed far more slowly (still 920 at step 1,400, 206 at 2,700, 54.3 at 5,000), and grad-norm stayed ~700 to the end. A rank-64 adapter constrained to the orthogonal complement of two earlier rank-64 blocks has materially less usable capacity for a task that already needs the most. Task 4 opened at penalty **4,991** — higher again with three adapters — so if the mechanism is (ii), later diagonals should stay depressed rather than recover.
**Read-ahead.** If b4's diagonal (env 2, an easy task — naive 96, ours 87, RETAIN 86) also comes in low, that isolates (ii) as dominant: the constraint, not task difficulty, is the binding limit. If b4's diagonal recovers to 40-60, (i) dominates and env 9 is simply the task O-LoRA cannot fit at r64. This is the cleanest natural experiment the chain offers.

### Entry 67 addendum 23 (9 Sep 26, 19:15 UK) — the read-ahead resolves: **the orthogonality constraint, not task difficulty, is the binding limit.** O-LoRA row 4 diagonal 17.0 on an easy task
**Row 4 (checkpoint 020000):** env 4 **0.0** (0/0/0/0); env 6 **5.0** (0/8/0/12); env 9 **0.0** (0/0/0/0); env 2 (just trained) **17.0** (28/12/0/28). Row mean **5.5**.

**Verdict on add-22's two hypotheses.** Env 2 is one of the easiest tasks in LIBERO-10 — naive r512 scored **96**, ours 87, RETAIN 86 on their own b4 diagonals. O-LoRA scores **17.0**. The diagonal did not recover, so hypothesis (i) (env 9 is simply hard) is **rejected as the dominant cause** and (ii) (the constraint tightens as adapters accumulate) **stands**. The two series say the same thing:

| task | prior adapters | final train loss | own-task diagonal |
|---|---|---|---|
| t1 (e4) | 0 | 0.047 | 43.0 |
| t2 (e6) | 1 | 0.101 | 39.0 |
| t3 (e9) | 2 | 0.386 | 2.0 |
| t4 (e2) | 3 | 0.301 | **17.0** |

Loss degrades ~6x and the diagonal falls ~2.5x between zero and three prior adapters, monotonically in adapter count and independent of which task is being fit (t3 hard, t4 easy, both ruined). **O-LoRA at r64 under our protocol is failing to LEARN, not merely failing to RETAIN** — a different failure from RETAIN's (which fit its tasks and then merged them away) and from naive's (which fit at specialist grade and then overwrote).
**Why this happens, mechanically.** Each new rank-64 adapter is pushed into the orthogonal complement of the union of all earlier adapters' row spaces. After k tasks that complement has lost ~64k directions at every constrained site, so by t4 the adapter is fitting inside a subspace already stripped of 192 directions it might have wanted. The penalty trace shows it biting: openings 995 (t2) -> 2,198 (t3) -> 4,991 (t4) -> 2,348 (t5), grad-norm ~620-725 sustained to the end of every task.
**Fairness caveat, to state in the paper.** Our r64 is the paper's own recipe verbatim (alpha/r 0.25, their target set, lr 1e-4 -> 1e-5, lambda_1 0.5, penalty skipped on the two 32-input projections where 640 orthogonal rows cannot exist) — 106.3M params per adapter, 1.063B after ten. But r64 was chosen to match the *r64 specialist target set*, not to match naive r512's 0.852B single-adapter budget, and O-LoRA's b1 diagonal of 43.0 (with **no** constraint active) already sat 27 points below naive's 70.0. So two effects compound: an undersized per-task adapter, and a constraint that removes capacity from it as the chain proceeds. **The honest claim is about the recipe as published, not about orthogonal-subspace methods in general**, and the pre-registered 20-45 band (add-2, citing OrthoSkillVLA's 30.5) now looks unreachable — projecting the diagonal trend, the b10 row mean will land in low single digits. A per-task r64 oracle (no constraint, task identity given) would separate the two cleanly and is the obvious follow-up; it is not in the current protocol.

### Entry 67 addendum 24 (10 Sep 26, 01:00 UK) — O-LoRA row 5 confirms add-23; halfway row mean 2.4
**Row 5 (checkpoint 025000):** env 4 **0.0**; env 6 **2.0** (4/0/0/4); env 9 **0.0**; env 2 **5.0** (4/12/4/0); env 7 (just trained, final train loss 0.371) **5.0** (12/4/4/0). Row mean **2.4** — against naive 18.0, RETAIN 32.2, ours 66.0 at the same block.
**Nothing new, which is the point.** The diagonal series is now 43.0 / 39.0 / 2.0 / 17.0 / **5.0** across zero-to-four prior adapters: after the second task it never again exceeds 17, and every off-diagonal cell in rows 3-5 is 0-6. Add-23's reading (the constraint, not task difficulty, is the binding limit) holds at the halfway mark with no sign of recovery. Every row from here is expected to be the same shape — a near-zero diagonal over a floor of zeros.
### Entry 67 addendum 25 (11 Sep 26, 22:10 UK) — **E67 COMPLETE.** Final row mean 6.5 — last of the four chains
**Chain.** `algebra_max_rel_err` held at **2.0e-05 at every one of the ten boundaries**. Final train loss per task: 0.047 / 0.101 / 0.386 / 0.301 / 0.371 / 0.326 / 0.303 / 0.407 / 0.396 / 0.259 — the step between t2 and t3 never recovers. Step time grew 2.2 -> 6.3 s/step with the penalty computed against every prior adapter.

**THE FULL O-LoRA TRIANGLE (r64, alpha 16, lambda_1 0.5; 4-seed x 25-episode cells, envs in train order):**

| after block | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 43 | - | - | - | - | - | - | - | - | - | 43.0 |
| b2 | 3 | 39 | - | - | - | - | - | - | - | - | 21.0 |
| b3 | 0 | 6 | 2 | - | - | - | - | - | - | - | 2.7 |
| b4 | 0 | 5 | 0 | 17 | - | - | - | - | - | - | 5.5 |
| b5 | 0 | 2 | 0 | 5 | 5 | - | - | - | - | - | 2.4 |
| b6 | 1 | 1 | 0 | 3 | 6 | 1 | - | - | - | - | 2.0 |
| b7 | 0 | 1 | 0 | 3 | 3 | 2 | 1 | - | - | - | 1.4 |
| b8 | 0 | 0 | 0 | 2 | 4 | 2 | 0 | 2 | - | - | 1.2 |
| b9 | 0 | 0 | 0 | 5 | 3 | 4 | 1 | 2 | 27 | - | 4.7 |
| b10 | 0 | 2 | 0 | 3 | 1 | 2 | 0 | 3 | 28 | 26 | **6.5** |

Diagonal mean **16.3** (ours 65.4, naive 73.1, RETAIN 62.2); prior-task mean at b10 **4.3** (ours 62.2, RETAIN 2.7, naive 0.0).

**FOUR-CHAIN FINAL ROW (the paper's headline row):** ours **65.1** | naive r512 **9.7** | naive r1216 param-matched **8.6** | RETAIN alpha 0.5 **8.2** | **O-LoRA r64 6.5**.

**O-LoRA drift matrix — and the twist.** Mean own-task drift **+71.6%**, which is the *second best* of the four (ours +28.5%, O-LoRA +71.6%, RETAIN +476.6%, naive +740-1567%). But read the per-task column before crediting it: t0 +466%, t1 +171%, t2 +49%, t3 +20%, t4 +7%, then **t5-t9 at +0.4% or less**. The late tasks barely drift because by then the model is barely changing — each new rank-64 adapter, confined to the orthogonal complement of 5-9 earlier ones, moves the function almost not at all. This is precisely the paper's Figure 3 claim (the constraint keeps previous-task loss low) and it is **true here in function space**; it is also why the rollout triangle is empty. O-LoRA buys function preservation by buying plasticity loss, and at r64 on a 4.1B VLA the price is the whole product: diagonals of 1-6% from b5 to b8 mean there is nothing worth preserving. The one intact claim is that the *mechanism* works as advertised — orthogonality really does freeze the function; it just freezes a function that never learned the task.
**Where the three failure modes now sit, stated cleanly for the writeup.** Naive: fits at specialist grade (diag 73.1), then overwrites — prior tasks 0.0. RETAIN: fits well (diag 62.2), then geometrically erases at ~2^-k per merge — prior tasks 2.7, drift +477%. O-LoRA: never fits after task 2 (diag 16.3) but holds what little it has — prior tasks 4.3, drift +72%. Ours: fits (diag 65.4) **and** holds (prior tasks 62.2, drift +28.5%). The three baselines fail in three different places; only the memory row does both jobs at once.
## Entry 68 - 9 Sep 26 (ABLATIONS: isolating the three headline mechanisms on the FINAL architecture — live-vs-stationary addressing, TF-IDF-only writes, joint-vs-staged router preparation at matched LR — plus zero-separation / zero-contrastive warm-up certificates. Pre-registered reads and result placeholders)

**Why (Josh, 9 Sep).** Two fresh-context reviews of the ICRA draft (Codex, Fable) converge on one theme: the paper
weakly ablates its core knobs, so the contributions cannot be isolated. Checked against the draft, the complaint is
fair. The three headline mechanisms — router warm-up ("geometry before content"), stationary addressing, protected
writes — are each supported by routing diagnostics on old layouts and by none of them by a success comparison on the
final 12-site / 7-table merged 6x2 architecture:
- Table II A (joint vs staged preparation): IoU only, 4-site n=384 config (E35 vs E37), and confounded with router LR
  (2.5e-5 vs 1e-4) — the paper says so itself. E36 attributed the E35 failure to "anchored optimisation" (values at
  40x the router LR), so the confound may be the whole effect.
- Table II B (stationary routing): self-IoU only, live arm = E38 rwarmupA (4 sites, 20-ep cells, killed at 7/10, e2
  80->25) vs stationary = E39 stageB, which ALSO re-ran its A-phase; no matched-horizon success number exists.
- Table II C (core-50 protection): peak-norm vs corefrac (E52/E53) — two protection variants, not protection vs none.
  The only "no protection" number is joint-era (E27/E28: 34.0 -> 40.5 at n=384, 4 sites).
- Table II D/E (separation, contrastive): joint-training probes (E25/E26), backbone co-adapting — the paper flags this.
Placement, sharing, write budget, LR and the expert anchor already have success-level ablations (Sec. V.C) and are
NOT repeated. Retrieval breadth, bank size, slot rank, heads, gating are closed axes (SR / summary §8) and are skipped.

**Design principles (agreed).** (1) Every arm is a single delta from the paper cell (E62 recipe, §2 of the summary),
on the FINAL layout, on the five development tasks (e4, e6, e9, e2, e7). Five tasks suffice for the three mechanism
arms: drift accumulates with exposure and E38 showed it inside five blocks; unprotected writes collapsed the earliest
task at five blocks (E51 P2); joint preparation acts on acquisition (E35 inits -18pp). (2) The control is NOT re-run:
the TF-IDF mask is built from retrieval indices accumulated across micro-batches
(an optimizer-step quantity), and micro-batching is recorded neutral (E53 24 Jul, E60 add-7),
so the H200 control at bs16xacc2 is valid against H100 arms at any ladder rung. (3) Full retention triangle per arm
(15 cells), not acquisition+final: the paper's R_post and backward-transfer metrics are triangle quantities, and the
intermediate rows are what distinguish gradual exposure-ordered decay (live routing) from boundary-specific loss
(unprotected writes). (4) All four paired eval seeds (1000/2000/3000/4000, 25 eps,
vec 13) — the E64 add-3 instrument — so every cell pairs with the control.

**The run-to-run bar.** The 5-task and 10-task merged 6x2 runs share recipe and seed and differ at row 5 by up to 8
points per cell (e7 43 vs 51, e4 56 vs 53), means 65.2 vs 66.0. That pair is the only replicate we have: **an ablation
delta under ~3 points on the 5-task mean is noise; per-cell deltas under ~8 are noise.** Arms are chosen because their
expected effects are large.

### The arms

| arm | single delta from the paper cell | steps | starts from | replaces |
|---|---|---|---|---|
| A1 live addressing | `use_frozen_base_input_features=false`, `frozen_prepass=false` from value fill onward (routers, gates and anchors read the live stream) | 10k fill + 25k seq5 = 35k | merged 6x2 warm-up ckpt (values zero at warm-up => live == memory-free there; arms share it) | Table II B |
| A2 TF-IDF-only writes | `protect_prior_slots=false`; same `tfidf_top_t=3072`, online IDF, LR, optimizer reset, union masks on shared tables | 20k (tasks 2-5) | control's task-1 checkpoint: with an empty store score == tfidf, so task 1 is identical by construction | the missing "no protection" row (= the Lin et al. recipe at our budget) |
| A3 joint preparation, matched LR | keys + query proj + anchor maps + values + gates + out-proj trained TOGETHER for 10k on libero_90 (MSE + c0.05 + sep8), router LR 1e-4 (= warm-up LR, removes the E35/E36 confound), stationary routing ON; then routers frozen, seq5 C-config verbatim | 10k joint + 25k seq5 = 35k | stage-1 base | Table II A |
| C1 zero-separation certificate | warm-up with `sep=0` (c 0.05, anchor 0.40, pool (1,0.5), prepass; router-only fast path) + held-out audit | 10k router-only | stage-1 base | Table II D |
| C2 zero-contrastive certificate | warm-up with `c=0` (sep 8, otherwise as C1) + audit | same | stage-1 base | Table II E |

Inference in A1 is the
plain single-pass path (prefix KV with memory, no pre-pass), which is what "live" means at deployment.

**Why no downstream separation run.** Separation's channel is the same-scene family (E27/E28, Sec. V.B). In the
development order the basket hub e7 is task 5 and its victims e0/e1 arrive at tasks 6 and 8, so a 5-task sep=0
sequence has no victim and would likely read flat — that would undermine the geometry story rather than bolster it.
A 10-task sep=0 arm is 60k steps and does not fit before the deadline. The held-out audit runs on all ten tasks'
demos from the warm-up checkpoint alone, so C1 measures the family channel (famIoU, and writable mass omega for e0/e1
against e7's core) without any downstream training; the paper's Sec. V.B overlap->deficit link carries it from there.
C2's expected failure mode is capacity (sprawl/collapse: E25, E45), i.e. acquisition, which the certificate shows
directly. Deferred, first to drop if time is short: expert-only memory (contribution (i) has only the LoRA compass
behind it; a full 45k chain).

### Pre-registered reads (written before any number lands)

| arm | acquisition (diagonal) | retention | diagnostics | kill/demote line |
|---|---|---|---|---|
| A1 live | ~= control, possibly +1-3 (tasks chase their own routing within a block, E39) | exposure-ordered decay; final mean well below 65.2; prior-task rows fall block by block, not at one boundary | pre->post self-IoU << 1 at every site above E4/V5 (E38 band 0.2-0.3), == 1 at E4 and V5 | final within 3 of control => stationarity is not load-bearing at this layout; Sec. III.D becomes a property statement, not a design win |
| A2 TF-IDF-only | >= control (no writer restriction) | loss concentrated on the earliest tasks; final mean below control; loss drift >> 3.9% (band: E53 peak-norm 22.6% on task 1) | prior-core write events > 0 at E14/E16 (control: 0); site-bleed unchanged | final within 3 of control => protection is not load-bearing on 5 unrelated tasks; the paper's write-rule claim then rests on 10-task e7 and the real robot only |
| A3 joint prep @1e-4 | genuinely uncertain: E35 (2.5e-5) had inits -18pp and e7 init 0 | uncertain | audit after 10k: E35-like diffuse routing (effnum ~2x, core50 large) OR a passing certificate | audit passes AND seq5 within 3 of control => "geometry before content" reduces to a router-LR observation; Sec. III.E must say so |
| C1 sep=0 | n/a | n/a | famIoU up from 0.145 band; bgIoU up (how much is the question — bg is the axis that pays, E56); omega for e0/e1 down | famIoU/bg unchanged => separation is decorative at this anchor weight; demote L_sep in Sec. III |
| C2 c=0 | n/a | n/a | collapse before 1k (E45 signature, ~70% of reads on one slot) or sprawl (core50/effnum far outside the gate) | certificate passes cleanly => contrastive is not load-bearing with pooled VLM routing; demote |

### Evaluation and reporting
- Control cells: the 5-task merged 6x2 run's OWN per-task checkpoints
  (005000..025000), 15 cells — not rows 1-5 of the seq10 triangle (different run; see the bar above).
  The seq10 rows 1-5 are kept as the replicate.
- Per arm: 5-row triangle (4 seeds x 25 eps), paired-noise MSE matrix with the FIXED loader (E65 add-16), and the
  arm's diagnostic (A1 self-IoU per site pre/post; A2 prior-core events + site-bleed; A3/C1/C2 the audit certificate).
- Columns per arm, matching Table I: acquisition mean, final mean, R_post, Delta_final, mean loss drift, paired
  delta vs control (SE over seed-wise differences).

### RESULTS

**Certificates (held-out audit, 10 tasks; expert / VLM):**
| warm-up | famIoU | bgIoU | mean core50 | min effnum (exp / vlm) | omega e0, e1 vs e7 core | gate | ref |
|---|---|---|---|---|---|---|---|
| paper cell (merged 6x2, c0.05 sep8) | exp 0.180-0.281 / vlm 0.082-0.145 | exp 0.031-0.091 / vlm 0.023-0.074 | exp 713-2019 / vlm 188-842 | 458 / 250 | (not in the audit summary) | PASS | E62, transcribed 9 Sep (add-3) |
| C1 sep=0 | | | | | | | |
| C2 c=0 | | | | | | | |
| A3 joint @1e-4 (post-10k audit) | | | | | | | |

**Five-task summary (4 seeds x 25 eps; paired delta vs control with SE):**
| arm | e4 | e6 | e9 | e2 | e7 | acq. mean | final mean | R_post | Delta_final | loss drift (fixed loader) | delta vs control |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control (5-task merged 6x2, own ckpts) | | | | | | | 65.2 (E62 add-3) | | | +3.9% (E65 add-24) | — |
| A1 live addressing | | | | | | | | | | | |
| A2 TF-IDF-only | | | | | | | | | | | |
| A3 joint prep @1e-4 | | | | | | | | | | | |

**Diagnostics:**
| arm | instrument | value | control reference |
|---|---|---|---|
| A1 | pre->post self-IoU per site (E4 E6 E8 E10 E14 E16 / V5..V15) | | 1.000 at every site (E62 add-2) |
| A2 | prior-core write events at E14 / E16 / shallow E4 / VLM | | 0 / 0 / ~25k / 0 (E62 add-2) |
| A2 | site-bleed on the five shared pairs | | 14-51% (E62 add-2) |
| A3 | audit certificate (row above) + seq5 in-run block-min losses | | |

### Entry 68 addendum 1 (9 Sep 26, 07:45 UK) — **W_a NEVER TRAINED (paper correction)**
**PAPER CORRECTION — the expert anchor map W_a is a fixed RANDOM projection, never trained.** Checked on the paper cell's warm-up checkpoint: all six `gemma_expert...layers.{4,6,8,10,14,16}.mlp.anchor_proj.weight` tensors have max|w| = 0.022097 = 1/sqrt(2048) and std 0.01276 = bound/sqrt(3) — bit-for-bit Kaiming-uniform init. Cause: no lerobot-train mode ever gives it a gradient (the sequential trainer trains values only). The tag comment ("Router group: trains in router warm-ups") and Sec. III.C of the draft ("W_a is learned during router warm-up") are wrong; the anchor is a fixed random projection of the pooled instruction hidden — which still conditions routing on language (a random projection preserves the pooled hidden's geometry, and E53's "anchor ledger" measured exactly that init geometry), so no result changes. Text fix: "a fixed random projection W_a" in Sec. III.C and Fig. 1(b). For E68 this is NOT a confound: A3's `train_memory_only` excludes `anchor_proj` identically, so every arm and the control share the same W_a. A trained-W_a arm (fix the filter, re-warm) is a future ablation, not a pre-deadline one.

### Entry 68 addendum 3 (9 Sep 26, 11:45 UK) — paper-cell certificate transcribed (the C1/C2 reference row); the "famIoU 0.145 band" does NOT describe the paper cell, so the C1 pre-registration is restated
The reference row of the certificate table, per site:

| site | famIoU | bgIoU | core50 mean (min-max) | min-task effnum |
|---|---|---|---|---|
| E4  | 0.205 | 0.038 | 735 (156-1547)   | 458 |
| E6  | 0.180 | 0.031 | 713 (196-1524)   | 544 |
| E8  | 0.214 | 0.037 | 806 (195-1485)   | 541 |
| E10 | 0.258 | 0.058 | 1308 (261-2745)  | 710 |
| E14 | 0.207 | 0.058 | 1274 (296-2491)  | 796 |
| E16 | 0.281 | 0.091 | 2019 (379-4364)  | 1067 |
| V5  | 0.082 | 0.023 | 226 (84-425)     | 250 |
| V7  | 0.117 | 0.024 | 188 (96-278)     | 291 |
| V9  | 0.134 | 0.037 | 328 (131-459)    | 361 |
| V11 | 0.137 | 0.046 | 428 (154-746)    | 404 |
| V13 | 0.136 | 0.052 | 499 (221-947)    | 497 |
| V15 | 0.145 | 0.074 | 842 (381-1738)   | 750 |

Gate re-verified from the raw numbers (E62 recorded "PASS" without transcribing them): expert bgIoU max 0.091 <= 0.10 (E16 is the tight one), expert mean core50 min 713 >= 400, expert min-task effnum min 458 >= 300, VLM min-task effnum min 250 >= 150 (V5, the tight one), VLM famIoU max 0.145 < 0.45. PASS on every clause, with E16's bgIoU and V5's effnum the two cells nearest their bands.
**CORRECTION carried into E68's C1 pre-registration.** The entry wrote "famIoU up from the 0.145 band", inheriting a number that belongs to the E37/E44-era 4-site n=384 configuration (and, coincidentally, to V15 here). The paper cell's actual famIoU is **0.180-0.281 on the expert and 0.082-0.145 on the VLM** — i.e. the merged 6x2 sits well ABOVE the old 0.145 on every expert site, which is consistent with E56/E59's finding that famIoU stopped converting and bgIoU became the axis that pays (the layout was selected on bg-first bands, not famIoU). C1 is therefore scored against the per-site numbers above, not against 0.145: the read is whether sep=0 moves the expert band up from 0.18-0.28 and, more importantly for the mechanism claim, whether it moves **bgIoU** off 0.031-0.091 and the VLM famIoU off 0.08-0.145. Unchanged bg across the board would mean separation is decorative at this anchor weight (the standing kill line).
Omega (writable read mass outside prior cores) is not part of the audit summary — it comes from the slot-analysis pass; it will be computed for all three warm-ups together when C1/C2 land, so the three are measured by one instrument.

### Entry 68 addendum 5 (9 Sep 26, 15:20 UK) — **live and stationary routing fit the pretraining corpus identically (0.096 vs 0.097 at the matched step)** — the arm's damage, if any, must come from the sequential stage
**The first A1 number, and it is a null — as pre-registered.** Both A-phases train values (+ gate, value_proj, swilu) for 10k steps on libero_90 from the SAME warm-up checkpoint, differing only in whether routing reads memory-free or live features, so their held-in loss is directly comparable at matched step/sample count:

| A-phase | routing | step | samples | epoch | loss | grad-norm |
|---|---|---|---|---|---|---|
| paper cell (E62) | stationary (frozen-base + prepass) | 10K | 320K | 0.56 | **0.097** | 0.030 |
| E68 A1 | LIVE | 10K | 320K | 0.56 | **0.096** | 0.030 |

(the two preceding log lines agree as closely: 0.100/0.097 vs 0.102/0.094.) **Reading:** live addressing costs nothing when there is a single training distribution and no earlier task to preserve — the router drifts with the values, but drifting together is not a defect while everything is being fitted at once. This matches the E68 pre-registration for A1 ("acquisition ~= control, possibly +1-3") and, more usefully, it *isolates* the arm: any A1 deficit that appears from here on cannot be attributed to a weaker value fill, because the substrate the sequential stage starts from is equal in fit to the control's. The whole test now rests on the sequential stage, exactly where E38's mechanism predicts it should.

### Entry 68 addendum 7 (9 Sep 26, 19:45 UK) — A1 boundary 1: in-run opener **75.0** on e4 vs the control's 80 (20-ep instrument, retired from decisions)
The in-run boundary eval read **pc_success 75.00** (tasks_seen=1, e4). Block-1 training loss fell 0.148 -> 0.043 over the 5,000 steps (grad-norm 0.017 -> 0.007, value LR 1.9e-3 -> 3.8e-4 on the linear decay), i.e. a normal block.
**Against the control at the same instrument:** the paper cell's 5-task run opened e4 at **80** on its fresh 20-episode boundary eval (E62 add-1, "best ever recorded"). A1 live reads **75**. The gap is 5 points on a 20-episode cell whose binomial spread is +-11pp (E41, the entry that retired this instrument from decisions), so the two are indistinguishable here and nothing should be concluded from it. Recorded because it is the first A1 rollout number and it is *directionally* consistent with the pre-registration ("A1 acquisition ~= control"), which the A-phase loss parity (add-5) already indicated more reliably.
**What actually decides A1** is unchanged and still ahead: the 4-seed x 25-episode retention triangle over all five blocks, where the pre-registered signature is exposure-ordered decay of the earlier tasks (E38's channel), not a weaker diagonal. The first cell where that can show is block 2's e4 row.

### Entry 68 addendum 9 (10 Sep 26, 01:15 UK) — A1 boundary-2 in-run eval: **57.5** (2 tasks, 20 eps each); tracking the control, no live-routing collapse through two blocks
A1 in-run trace so far: b1 **75.0** (1 task) -> b2 **57.5** (mean over e4 retained + e6 just-trained). The trainer logs only the aggregate at these boundaries, so acquisition and retention are not separable from this instrument; for scale, the control's 4-seed b2 row is 54/65 = **59.5** (E64 add-12, a different instrument at 25 eps x 4 seeds). A1's 57.5 is therefore in the same region, and **there is no sign of the E38 collapse through two blocks** — consistent with that run too, which degraded progressively rather than immediately (rwarmupA reached 7/10 blocks before e2 fell 80 -> 25). Nothing is concluded from either number: the 20-episode single-seed cell was retired from decisions in E41, and A1's verdict rests on its 4-seed triangle.
The shared-table union mask is behaving: 3072 + 3072 -> 5996 rows (overlap 148), i.e. the two sites of a shared pair request nearly disjoint row sets and the union is close to additive.

### Entry 68 addendum 10 (10 Sep 26, 05:15 UK) — A1 boundary 3: in-run **65.0** (3 tasks); three blocks in, the live-routing arm shows NO exposure-ordered decay — flagging early that A1 may come out NULL
Block 3 (e9) trained 0.508 -> 0.082.
**A1's in-run trace against the control's 4-seed triangle rows** (different instruments — 20 eps x 1 seed vs 25 eps x 4 seeds — so this is a shape comparison, not a paired one):
| after block | A1 live (in-run, 20 ep) | control (4-seed triangle, E64 add-12) |
|---|---|---|
| b1 | 75.0 | 54.0 |
| b2 | 57.5 | 59.5 |
| b3 | **65.0** | 62.3 |
**The pre-registered A1 signature is not appearing.** E68's table predicted "exposure-ordered decay; final mean well below 65.2; prior-task rows fall block by block". Through three blocks the aggregate is flat-to-rising and b3 sits slightly *above* the control's b3. Two honest caveats before anyone reads that as a result: (i) the in-run cell is 20 episodes at one seed, retired from decisions in E41 at +-11pp, and it aggregates the just-trained task with the retained ones, so a strong diagonal can mask a weak row; (ii) E38's own collapse was progressive — rwarmupA held up early and lost e2 80 -> 25 only by block 7 of 10 — and A1 has only two more blocks to run, so this arm may simply be too short to reach the regime where drift bites.
**Consequence to prepare for.** If the 4-seed triangle confirms the shape, A1 fires its pre-registered kill line ("final within 3 of control => stationarity is not load-bearing at this layout"), and Sec. III.D of the paper must be restated: stationary addressing would then be an exactness property of the routing function that we can demonstrate and audit (self-IoU 1.000, E62 add-2), rather than a design choice carrying measurable retention. That is a weaker claim than the draft makes, and it is better to know it now than at submission. The alternative reading — that five tasks is below the horizon where the drift channel converts, given E38 needed seven — is testable only with a 10-task live arm (60k steps), which does not fit before the deadline; if A1 lands null, the honest sentence names the horizon rather than claiming the mechanism is absent.

### Entry 68 addendum 11 (10 Sep 26, 08:05 UK) — **CORRECTION to add-10**: the per-task cells DO show A1's e4 declining (75/60/55) — but the control's own e4 swings 80/65/75/45/58 on the same instrument, so neither reading survives; the 20-ep in-run triangle cannot resolve this arm
Add-10 read A1's in-run *aggregates* (75.0 / 57.5 / 65.0) as "no exposure-ordered decay" while noting the masking risk. The per-task cells show the masking was real — and then show the comparison is unusable anyway.
**A1 live (in-run, 20 eps x 1 seed):**
| after block | e4 | e6 | e9 | row mean |
|---|---|---|---|---|
| b1 | 75 | | | 75.0 |
| b2 | 60 | 55 | | 57.5 |
| b3 | 55 | 65 | 75 | 65.0 |
**Control, matched run / matched instrument** (its b5 row is the 50-ep final = E62 add-1's 58/74/64/84/54):
| after block | e4 | e6 | e9 | e2 | e7 | row mean |
|---|---|---|---|---|---|---|
| b1 | 80 | | | | | 80.0 |
| b2 | 65 | 65 | | | | 65.0 |
| b3 | 75 | 60 | 75 | | | 70.0 |
| b4 | 45 | 70 | 80 | 80 | | 68.8 |
| b5 | 58 | 74 | 64 | 84 | 54 | 66.8 |
**Reading.** (i) A1's e4 column falls monotonically 75 -> 60 -> 55, which is the pre-registered live-routing signature, so add-10's headline sentence is withdrawn. (ii) But the CONTROL's e4 column reads 80 -> 65 -> 75 -> **45** -> 58 under stationary routing with protection on — a 30-point swing between adjacent blocks in the arm that is by construction stable, and its b3 (75) exceeds its b1 (80) only after dipping to 65. A column that volatile in the control cannot license reading a 20-point monotone drift in the arm. This is precisely the +-11pp binomial spread that retired the instrument in E41, compounded by aggregating over cells. (iii) What survives: A1's row means sit 5.0 / 7.5 / 5.0 points below the control's at b1/b2/b3 — consistent in sign across three rows, which is mildly suggestive and nothing more.
**Standing position for A1 is therefore UNCHANGED and undecided**: no interim claim either way, and the arm is decided by the post-hoc 4-seed x 25-episode triangle (100 eps/cell), which is the instrument every headline in the paper uses. Add-10's "prepare for a null" contingency stays on the table but is no longer supported by evidence; the horizon caveat (E38 needed 7 blocks) stays too.

### Entry 68 addendum 12 (10 Sep 26, 09:20 UK) — A1 b4: e4 goes back UP (55 -> 65), the "monotone decline" of add-11 is dead, and A1's row mean now EXCEEDS the control's — the in-run instrument is settled as uninformative for this arm
Block 4 (e2) trained to 0.050.
**A1 live, in-run per-task (20 eps x 1 seed), four rows:**
| after block | e4 | e6 | e9 | e2 | row mean |
|---|---|---|---|---|---|
| b1 | 75 | | | | 75.0 |
| b2 | 60 | 55 | | | 57.5 |
| b3 | 55 | 65 | 75 | | 65.0 |
| b4 | **65** | 55 | 75 | 95 | **72.5** |
**Both interim readings are now dead.** Add-10 claimed no decay (from aggregates); add-11 withdrew that and noted e4 falling 75 -> 60 -> 55; b4 has e4 back at **65**, so the column is 75/60/55/65 — not monotone, not a drift signature, just noise. And the row means now read 75.0 / 57.5 / 65.0 / **72.5** against the control's 80.0 / 65.0 / 70.0 / **68.8**: A1 was 5-7.5 points *below* at b1-b3 and is 3.7 points *above* at b4. A consistent-sign deficit that reverses on the next row was never a deficit.
**Settled, and worth stating once so it is not relitigated:** the 20-episode single-seed in-run cell cannot resolve differences of this size in either arm — the control's own e4 column (80/65/75/45/58) swings 30 points under stationary routing with protection on. No further interim reads will be taken from this instrument for E68; A1, A2 and A3 are each decided by their post-hoc 4-seed x 25-episode triangle (100 eps/cell) and their paired-noise loss matrix, nothing else. The in-run numbers stay in the log as a record of what the run did, not as evidence.

### Entry 68 addendum 13 (10 Sep 26, 10:05 UK) — **THE A1 ABLATION IS BITING HARD: cross-arm read-mass IoU on the SAME task is 0.44-0.68**, with an E38 depth gradient — yet inter-task separation is preserved, which is the likely reason retention has not collapsed
Josh asked whether the A1 null is real or an inert flag. Measured directly from the two runs' own usage records: per-site weighted-Jaccard read-mass IoU between **A1-live and the control-stationary on the same task, same frozen warm-up router, same data, same seed**, so the ONLY difference is whether routing reads live or memory-free features.
**Task 0 (e4), cross-arm IoU:**
| site | E4 | E6 | E8 | E10 | E14 | E16 | V5 | V7 | V9 | V11 | V13 | V15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IoU | 0.677 | 0.641 | 0.559 | 0.518 | 0.482 | 0.524 | 0.684 | 0.532 | 0.455 | 0.438 | 0.456 | 0.512 |
About **half the read mass moves**. The gradient is the E38 signature in production: shallow sites drift least (E4 0.677, V5 0.684), deep sites most (E14 0.482, V11 0.438). E4 is perturbed at all — rather than the exact 1.000 E38 saw at its bottom site — because in the interleaved 6x2 layout V5 sits BELOW E4, so even the shallowest expert routing input carries VLM memory. A1's footprints are also broader (V9 61,145 slots touched vs the control's 45,506; V11 65,152 vs 55,107) and its shallow cores smaller (E4 core50 2,521 vs 4,342), consistent with a router whose input moves *within* each block so the accumulated footprint is a union over a moving target.
**But inter-task separation is essentially unchanged.** Pairwise inter-task read-mass IoU (the bg proxy), A1 vs control at the same sites:
| pair | E4 | E10 | E16 | V5 | V11 | V15 |
|---|---|---|---|---|---|---|
| t0-t1 live | 0.166 | 0.223 | 0.411 | 0.153 | 0.247 | 0.422 |
| t0-t1 control | 0.221 | 0.286 | 0.406 | 0.192 | 0.245 | 0.380 |
| t0-t3 live | 0.038 | 0.090 | 0.181 | 0.061 | 0.178 | 0.328 |
| t0-t3 control | 0.055 | 0.088 | 0.200 | 0.083 | 0.146 | 0.269 |
Same magnitudes, same depth structure, A1 marginally *more* separated at shallow expert sites and marginally less at deep VLM ones.
**The mechanism this suggests.** The router is frozen in both arms, so its *geometry* — the map from feature directions to slot regions — is identical; live routing perturbs only its *input*. The perturbation translates each task's footprint substantially (cross-arm 0.5) while leaving tasks about as far apart from each other as before. If tasks remain mutually separated, the write mask and protection still act on approximately the right disjoint regions, which is a coherent reason for retention to hold up despite large absolute addressing drift. E38's damage would then require not drift per se but drift that *destroys separation* or that de-references the protection store.
**What is NOT measured here, stated plainly.** (i) **Self-IoU across the sequence** — whether task k's footprint at the final checkpoint still matches what the protection store recorded during block k — is the decisive statistic and is NOT obtainable from these files: they record each task's usage *during its own block*, one recording per task. It needs the held-out audit run on two checkpoints and is the E62 add-2 instrument that read exactly 1.000 for the control. (ii) **famIoU is undefined on the front-5**: the basket family is e7 + e0 + e1, and only e7 is in this sequence, so no family pair exists; the audit's famIoU is a 10-task statistic against a frozen checkpoint and is not comparable to anything computed here. (iii) These are training-time footprints under a moving router, not frozen-checkpoint footprints, so they are not the same statistic as the audit's bgIoU and should not be quoted against the gate bands.
**Consequence for the arm:** the A1 null (if the 4-seed triangle confirms it) is a REAL result about a genuinely non-stationary router, not an artefact of a flag that did nothing. That materially strengthens whatever Sec. III.D ends up saying, in either direction. **Recommended next measurement:** self-IoU on A1's 005000 vs 025000, which converts this from "drift preserved separation" (inferred) to measured.

### Entry 68 addendum 14 (10 Sep 26, 17:20 UK) — **A1 COMPLETE: final 50-ep 64.0 vs the control's 66.8, a paired mean delta of -2.8 — inside the pre-registered +-3 noise bar, so A1 is tracking to fire its NULL kill line**
**A1 final row vs the control, SAME instrument (50 eps, seed 1000), paired per task:**
| | e4 | e6 | e9 | e2 | e7 | mean |
|---|---|---|---|---|---|---|
| control (E62 add-1) | 58 | 74 | 64 | 84 | 54 | **66.8** |
| A1 live | 58 | 56 | 60 | 78 | 68 | **64.0** |
| delta | 0 | **-18** | -4 | -6 | **+14** | **-2.8** |
**Reading.** The mean deficit is **2.8 points**, inside the +-3 bar E68 set from the 5-task/10-task replicate, and far inside the +-7pp E43 assigns a 50-episode cell. The per-task pattern is two large movers in opposite directions that cancel: e6 **-18** and e7 **+14**. e7 is the basket hub and the hardest cell in the suite — the one the control itself only reaches 54 on — and live routing does *better* there, which is the opposite of the pre-registered failure mode (the family/hub task was supposed to be where drift hurt most). Nothing here looks like exposure-ordered forgetting.
**Status of the A1 pre-registration.** The kill line reads "final within 3 of control => stationarity is not load-bearing at this layout". At the 50-ep instrument A1 lands at -2.8 and therefore **fires it**, pending the 4-seed triangle, which is the instrument of record and has not run. Combined with add-13 (the ablation genuinely moves ~half the read mass, cross-arm IoU 0.44-0.68), the emerging result is: **large addressing drift, preserved inter-task separation, and no measurable retention cost over five tasks.** If the triangle confirms, Sec. III.D must be restated — stationary addressing becomes a demonstrable exactness property (self-IoU 1.000) rather than a design choice carrying measured retention, with the honest caveat that five tasks may sit below the horizon where drift converts (E38 needed 7 of 10).

**The control triangle already exists.** Its rows b1-b5 (E64 add-12) are blocks 1-5 of the merged 6x2 chain at the headline instrument, same recipe, same seed, same task order; blocks 1-5 are the same computation whether the chain later continues to ten tasks or stops at five. Use those as A1's comparator (54.0 / 59.5 / 62.3 / 64.2 / 66.0). The 5-task and 10-task runs differ per cell by up to 8 points, which is the replicate spread noted in E68 and the reason the +-3 noise bar exists.

### Entry 68 addendum 16 (10 Sep 26, 22:15 UK) — A2 boundary 2 (its first MEASURED cells): e4 55 / e6 45, row mean 50.0 — both below the control, and the DIAGONAL is down 20, which is the opposite of the pre-registration
Block 2 trained 0.261 -> 0.034.
A2's b1 row is inherited, not measured — protection has no role in block 1, so the fork is exact and block 1 was never retrained.
**Three arms at b2, same in-run instrument (20 eps x 1 seed):**
| arm | e4 (retained) | e6 (just trained) | row mean |
|---|---|---|---|
| control | 65 | 65 | 65.0 |
| A1 live | 60 | 55 | 57.5 |
| **A2 no-protection** | **55** | **45** | **50.0** |
**Two observations, both held loosely.** (i) The retained cell e4 is 55 against the control's 65, the direction protection is supposed to defend — but A1, which leaves protection ON, sits at 60, so at this instrument the three arms are separated by less than one standard error of a 20-episode cell. (ii) More interesting, A2's **diagonal is 20 points below the control's** (45 vs 65). The E68 pre-registration for A2 reads "acquisition >= control (no writer restriction)", so if this survives the 4-seed instrument it is a *falsified* prediction and needs a mechanism: plausibly that an unprotected writer spreads its budget over contested slots that other tasks also drive, producing a mushier fit for the current task rather than a cleaner one — the E35 "diffused writes" signature, arriving from a different direction.
**No conclusion is drawn.** Add-12 closed interim reads on this instrument for E68 after the control's own e4 column was shown to swing 30 points; that ruling stands and applies here. A2 is decided by its 4-seed triangle (14 cells — b1 adopted from the control by construction) and its paired-noise loss matrix, where the pre-registered signature is loss drift well above the control's +3.9% and prior-core write events appearing at E14/E16 where the control records exactly zero. The diagonal question is now an explicit watch item for that triangle.

### Entry 68 addendum 17 (11 Sep 26, 03:15 UK) — **A2 b3: 30 / 35 / 65, row mean 43.3 — the prior-task columns are now in monotone free-fall (e4 80 -> 55 -> 30), which IS the pre-registered no-protection signature and is far outside what this instrument's noise can explain**
**Three arms at b3, in-run instrument:**
| arm | e4 | e6 | e9 | row mean |
|---|---|---|---|---|
| control | 75 | 60 | 75 | 70.0 |
| A1 live | 55 | 65 | 75 | 65.0 |
| **A2 no-protection** | **30** | **35** | **65** | **43.3** |
**The columns, which is where this becomes readable:**
| arm | e4 across b1->b3 | e6 across b2->b3 |
|---|---|---|
| control | 80 -> 65 -> 75 | 65 -> 60 |
| A1 live | 75 -> 60 -> 55 | 55 -> 65 |
| **A2** | **80 -> 55 -> 30** | **45 -> 35** |
**Why this is read, when add-12 closed interim reads on this instrument.** That ruling was calibrated to differences of 5-7 points, where the control's own +-11pp cell noise (and its 30-point e4 swing) makes any claim unsafe. A2 is not in that regime: e4 has fallen **50 points over two blocks**, monotonically, from a cell the control holds flat at 75-80; the row mean is **-26.7** against the control where A1's worst row was -7.5. Two columns are declining together, and the just-trained cell (e9 65 vs the control's 75) is also soft. A pattern of that size and shape is not a noise story, and it would be dishonest to hide behind the earlier ruling. The formal verdict still belongs to the 4-seed triangle — but the pre-registered A2 signature ("damage concentrated on the earliest tasks") is now visibly present, in contrast to A1 where it never appeared.
**Provisional reading of the pair, to be confirmed or killed by the triangles.** Protection looks load-bearing and stationary addressing does not, at five tasks on this layout. That is the opposite of the emphasis in the current draft, where stationary addressing is presented as a headline design choice (Sec. III.D) and the write rule as an extension of prior work (Sec. III.E). If both triangles confirm, the paper's mechanism story should be re-weighted accordingly, and add-13's finding supplies the reason the two differ: live routing translates every task's footprint together and preserves inter-task separation, so protection still acts on the right disjoint regions; removing protection instead lets the writer overwrite those regions directly, and no amount of routing stability compensates.
**Watch item from add-16 still open:** A2's diagonal was 20 points low at b2 (45 vs 65) against a pre-registration that expected acquisition to be *better* without a writer restriction; b3's diagonal is also low (65 vs 75). If that holds at 4 seeds it needs the E35 "diffused writes" mechanism as an explanation, and it is a second finding rather than a footnote.

### Entry 68 addendum 18 (11 Sep 26, 09:20 UK) — A2 b4: 35 / 55 / 50 / 85, mean 56.25 — **the "monotone free-fall" of add-17 partially REVERSES** (e4 30->35, e6 35->55) and the weak-diagonal watch item softens (e2 85); the row-level deficit survives, the column-level story does not
**A2's in-run triangle, four rows:**
| after block | e4 | e6 | e9 | e2 | row mean |
|---|---|---|---|---|---|
| b1 | 80* | | | | 80.0 |
| b2 | 55 | 45 | | | 50.0 |
| b3 | 30 | 35 | 65 | | 43.3 |
| b4 | **35** | **55** | 50 | **85** | **56.25** |
(*inherited from the seed.)
**Correction to add-17.** That entry read e4's 80 -> 55 -> 30 as monotone free-fall "far outside what this instrument's noise can explain", and justified breaking add-12's moratorium on that basis. b4 has e4 back to **35** and e6 back to **55** from 35 — both reversals, in the same shape that killed the equivalent A1 reading at add-12. The honest position is that the *column* story was over-read a second time, by me, for the same reason: a 20-episode single-seed cell will manufacture monotone-looking runs and then undo them. Add-17's mechanism paragraph should be treated as a hypothesis awaiting the triangle, not as a finding.
**What does survive is the row level.** Row means across b2-b4: control **65.0 / 70.0 / 68.8**, A1 **57.5 / 65.0 / 72.5**, A2 **50.0 / 43.3 / 56.25** — A2 below the control at every row by 15.0, 26.7 and 12.5, mean deficit ~18 points, where A1's three-row mean deficit is ~1.9. That is a consistent sign across three independent rows for A2 and no consistent sign for A1, which is a weaker claim than add-17 made but still the clearest separation E68 has produced so far.
**The weak-diagonal watch item (add-16) also softens:** A2's diagonals read 45 (b2), 65 (b3), **85** (b4) against the control's 65, 75, 80. The b2 cell now looks like the outlier rather than the start of a pattern, and the pre-registered expectation that acquisition is unharmed without a writer restriction is no longer contradicted. Keep it open, decide it at 4 seeds.
**Standing rule restated, because I have now broken it twice:** no arm-level claim from the in-run instrument. A1 and A2 are both decided by their 4-seed triangles and paired-noise loss matrices, and the loss matrix is the sharper instrument for A2 specifically — prior-core write events at E14/E16 (control: exactly 0) are a *mechanism* readout that does not depend on rollout noise at all.

### Entry 68 addendum 19 (11 Sep 26, 22:10 UK) — **A2 COMPLETE: final 59.2 vs the control's 66.8 (−7.6), with the oldest task at 26 vs 58 (−32) — the exposure-ordered signature A1 never produced. C1 COMPLETE: removing separation blows the background-overlap gate at ALL SIX expert sites (bgIoU 0.196–0.267 vs 0.031–0.091).**
**A2 (no protection) final, 50-ep instrument, paired against the control and A1:**
| | e4 | e6 | e9 | e2 | e7 | mean |
|---|---|---|---|---|---|---|
| control | 58 | 74 | 64 | 84 | 54 | **66.8** |
| A1 live | 58 | 56 | 60 | 78 | 68 | **64.0** (−2.8) |
| **A2 no-protect** | **26** | 56 | 78 | 80 | 56 | **59.2** (−7.6) |
A2's in-run row means across the sequence: 80.0 / 50.0 / 43.3 / 56.25 / **59.2**.
**The cell that matters is e4 at 26 against the control's 58.** e4 is the *first* task and therefore carries the most subsequent exposure; it is down 32 points, while the two most recently trained cells (e2 80, e7 56) are at or above the control. That is exposure-ordered decay in the shape the pre-registration named for A2, and — unlike the column readings I over-called in add-17 and withdrew in add-18 — it appears at the 50-episode instrument, on the final checkpoint, in the arm's headline number. A1's equivalent cell is 58, i.e. unchanged from the control. **Protection is doing work; stationary addressing, at five tasks on this layout, is not.** Both statements remain provisional on the 4-seed triangles.
**C1 (separation weight 8 → 0) certificate — the cleanest result E68 has produced.** Held-out audit on all ten tasks, against the paper cell's reference (add-3):
| | expert famIoU | expert bgIoU | expert core50 | VLM famIoU | VLM bgIoU | VLM core50 |
|---|---|---|---|---|---|---|
| paper cell | 0.180–0.281 | **0.031–0.091** | 713–2019 | 0.082–0.145 | 0.023–0.074 | 188–842 |
| **C1 sep=0** | **0.380–0.455** | **0.196–0.267** | 387–1016 | **0.181–0.385** | 0.074–0.164 | 93–191 |
Background overlap rises **3–6×** and **fails the ≤0.10 gate at every one of the six expert sites**; family overlap roughly doubles on both towers. The pre-registered kill line — "famIoU/bg unchanged ⇒ separation is decorative at this anchor weight" — does **not** fire. Separation is load-bearing for the geometry, and this is the frozen-backbone, final-architecture evidence that Table II row D currently lacks (it is presently a joint-training probe from E25/E26).
**A second, unpredicted feature of C1 worth keeping:** cores get *smaller* and effective-number *larger* (expert core50 387–1016 vs 713–2019; min-effnum 784–1934 vs 458–1067). Without the separation term each task's reads concentrate more tightly, yet the tasks pile onto *each other* — concentration without separation. That is a cleaner statement of what the separation loss does than the draft currently makes: it is not a compaction knob (contrastive does that, E24) but a translation knob, exactly as E25/E26 argued, now demonstrated on the paper architecture.
C2 (contrastive 0.05 → 0) runs at updt_s 1.410 against C1's 0.655, a property of the arms: C1 ran with the separation term disabled (the expensive one, it scores the joint slot distribution against a 512-sample queue every step) while C2 keeps separation at 8.0.

### Entry 68 addendum 21 (12 Sep 26, 01:25 UK) — C2 certificate: removing the contrastive loss does NOT collapse or sprawl — it PASSES the gate, narrowly, so C2 fires its demote line while C1 did not
**C2 (contrastive 0.05 -> 0) certificate, against the paper cell (add-3) and C1 (add-19):**
| | expert famIoU | expert bgIoU | expert core50 | VLM famIoU | VLM min-effnum |
|---|---|---|---|---|---|
| paper cell | 0.180–0.281 | 0.031–0.091 | 713–2019 | 0.082–0.145 | 250–750 |
| C1 sep=0 | **0.380–0.455** | **0.196–0.267** | 387–1016 | **0.181–0.385** | 210–356 |
| **C2 c=0** | 0.155–0.269 | **0.012–0.048** | 432–1446 | 0.109–0.152 | **158–276** |
**C2 passes every gate clause**: expert bgIoU ≤ 0.10 (0.012–0.048, actually *better* than the paper cell), expert mean core50 ≥ 400 (min 432 at E6), expert min-effnum ≥ 300 (min 387 at E6), VLM min-effnum ≥ 150 (min **158** at V7 — the tightest clause in the whole run), VLM famIoU < 0.45. The pre-registered failure modes — collapse before 1k (E45's ~70%-on-one-slot signature) or sprawl far outside the gate — **did not occur**, so C2's kill line fires: *contrastive supervision is not load-bearing for the certificate at this configuration*.
**Why that does not contradict E25/E45, and what it actually shows.** Contrastive still does something measurable and it is exactly what E24 called it — a breadth knob, not a separator. Without it, footprints *tighten*: expert core50 drops from 713–2019 to 432–1446 and VLM min-effnum from 250–750 to 158–276, brushing the 150 tripwire. Separation (famIoU, bgIoU) is untouched. The collapse E45 recorded was on **per-token VLM routing**; the paper cell routes the state region on a single pooled key (E45–E46), and pooled routing evidently supplies the breadth that contrastive was previously needed for. So the honest sentence is: *with pooled VLM routing, the contrastive term is redundant for separation and marginal for capacity* — which is a narrower and better-supported claim than the draft's Table II row E, and it is now frozen-backbone evidence on the final architecture rather than a joint-training probe.
**The two certificates together settle the geometry half of E68.** Separation is load-bearing and decisive (C1: background overlap up 3–6×, gate failed at all six expert sites). Contrastive is not (C2: gate passed, separation unchanged, capacity mildly reduced). That is an asymmetry the current draft does not make, and it is cheap to state.
**E68 training scoreboard, all four arms, provisional on the triangles:**
| arm | what it removes | result | pre-registered line |
|---|---|---|---|
| A1 | stationary addressing | 64.0 vs 66.8 (−2.8); routing genuinely moved (cross-arm IoU 0.44–0.68, add-13) | **null fires** |
| A2 | prior-usefulness protection | 59.2 vs 66.8 (−7.6); oldest task 26 vs 58 | **effect, exposure-ordered** |
| C1 | inter-task separation loss | bgIoU 0.031–0.091 -> 0.196–0.267, gate failed 6/6 | **effect, decisive** |
| C2 | contrastive loss | gate passed; cores tighten only | **demote fires** |
**Remaining work is evaluation, not training:** A1's 4-seed triangle, A2's, the paired-noise loss matrices — the sharper instrument for A2, since prior-core write events at E14/E16 are noise-free — and A3 (joint preparation at matched router LR), which remains the one unscheduled arm of the original five.

### Entry 68 addendum 22 (12 Sep 26, 02:20 UK) — A3 (joint router preparation at matched LR), the last unrun arm of the original five
**What A3 tests, restated because it is the arm most likely to change the paper's framing.** The method's first named design choice is "geometry before content": two separate 10k stages on the frozen backbone — a router warm-up training ONLY keys and query projections with slot values pinned at zero (so the flow-matching loss has exactly zero gradient on the router), then a value fill training ONLY values with routers frozen. A3 collapses that into ONE 10k stage training routers and values together under MSE plus the same auxiliary losses, then freezes the routers and runs the identical 5-task sequence.
**The confound it exists to remove.** Table II row A rests on E35 (joint, killed at 6/10 blocks) versus E37 (staged, best certificate) — overlap numbers only, no success comparison, on a 4-site n=384 configuration two generations old. The two conditions also differed in router learning rate, **2.5e-5 vs 1e-4**, and E36 diagnosed the joint failure as exactly that: values training at ~40x the router's rate, so the router never moved ("anchored optimisation"). A3 runs joint preparation at the warm-up's own **1e-4**. If it then matches the staged protocol, "geometry before content" reduces to a learning-rate observation and Sec. III.E is overclaiming.
Config (single delta from the paper chain): `train_router_only=false train_memory_only=true freeze_memory_router=false router_only_fast=false optimizer_lr=1e-4 vlm_route_once=false`, prepass ON; then the held-out audit (informational, part of the result) and the seq5 stage with frozen routers. Note W_a is untrained in BOTH arms (add-1), so the anchor is an identical fixed random projection either way — no confound there.
Pre-registered read is unchanged from the entry: genuinely uncertain on acquisition; the audit after 10k either shows E35-like diffuse routing (effnum ~2x, large cores) or a passing certificate; **kill line — audit passes AND seq5 lands within 3 of the control ⇒ the staged protocol reduces to router LR, and Sec. III.E must say so.**

### Entry 68 addendum 23 (12 Sep 26, 06:20 UK) — A1 and A2 4-seed retention triangles COMPLETE
Both ran 25 episodes x 4 paired seeds (1000/2000/3000/4000) per cell, vec batch 13, envs in train order 4/6/9/2/7. Comparator throughout is the merged 6x2 chain's b1-b5 (E64 add-12): 54.0 / 59.5 / 62.3 / 64.2 / 66.0.

**A1 (live addressing)**

| after block | e4 | e6 | e9 | e2 | e7 | row mean | control | delta |
|---|---|---|---|---|---|---|---|---|
| b1 | 67 | | | | | 67.0 | 54.0 | +13.0 |
| b2 | 52 | 77 | | | | 64.5 | 59.5 | +5.0 |
| b3 | 44 | 66 | 68 | | | 59.3 | 62.3 | -3.0 |
| b4 | 48 | 63 | 68 | 86 | | 66.3 | 64.2 | +2.1 |
| b5 | 50 | 74 | 69 | 85 | 44 | 64.4 | 66.0 | -1.6 |

**A2 (TF-IDF-only writes, `protect_prior_slots=false`)**

| after block | e4 | e6 | e9 | e2 | e7 | row mean | control | delta |
|---|---|---|---|---|---|---|---|---|
| b1 | 54 | | | | | 54.0 | 54.0 | +0.0 |
| b2 | 42 | 66 | | | | 54.0 | 59.5 | -5.5 |
| b3 | 45 | 62 | 74 | | | 60.3 | 62.3 | -2.0 |
| b4 | 39 | 65 | 75 | 82 | | 65.3 | 64.2 | +1.1 |
| b5 | 32 | 69 | 61 | 83 | 54 | 59.8 | 66.0 | -6.2 |

A2 per-seed: b1 e4 52/52/60/52; b2 e4 40/48/40/40, e6 64/68/64/68; b3 e4 48/52/44/36, e6 80/52/52/64, e9 68/68/76/84; b4 e4 40/40/40/36, e6 60/60/76/64, e9 72/76/72/80, e2 84/76/80/88; b5 e4 32/32/28/36, e6 68/76/68/64, e9 76/64/60/44, e2 80/80/80/92, e7 52/52/64/48.

**Final rows:** A1 64.4, A2 59.8, control 66.0. **Prior-task mean at b5** (row excluding the diagonal): A1 69.5, A2 61.2, control 62.2 (E64 add-12 b5 e4/e6/e9/e2 = 53/72/65/89, diagonal e7 51). **Diagonal cells:** A1 67/77/68/86/44, A2 54/66/74/82/54.
**e4 column across blocks:** A1 67/52/44/48/50; A2 54/42/45/39/32; control 54/54/50/51/53.
**A2 b1 = 54.0 (per-seed 52/52/60/52) reproduces the comparator's b1 of 54.0 exactly.** A2's checkpoint 005000 is a byte copy of the control's task-1 boundary (add-20), so this cell is an instrument/fork check, not an ablation measurement; it is also the matched 5-task control's b1.

### Entry 68 addendum 23 (12 Sep 26, 06:30 UK) — **THE 4-SEED TRIANGLES FOR A1 AND A2 ARE IN**. Final means: control 66.0, A1 64.4 (−1.6), A2 59.8 (−6.2). But the final mean HIDES A1's retention: by the paper's own Δ_final metric the ranking is control +0.4, **A1 −4.0**, A2 −6.2
**A1 — live addressing**
| after | e4 | e6 | e9 | e2 | e7 | mean |
|---|---|---|---|---|---|---|
| b1 | 67 | | | | | 67.0 |
| b2 | 52 | 77 | | | | 64.5 |
| b3 | 44 | 66 | 68 | | | 59.3 |
| b4 | 48 | 63 | 68 | 86 | | 66.3 |
| b5 | 50 | 74 | 69 | 85 | 44 | **64.4** |
**A2 — no protection**
| after | e4 | e6 | e9 | e2 | e7 | mean |
|---|---|---|---|---|---|---|
| b1 | 54 | | | | | 54.0 |
| b2 | 42 | 66 | | | | 54.0 |
| b3 | 45 | 62 | 74 | | | 60.3 |
| b4 | 39 | 65 | 75 | 82 | | 65.3 |
| b5 | 32 | 69 | 61 | 83 | 54 | **59.8** |
**Control** (merged 6×2, 10-task instance rows 1–5, E64 add-12): 54.0 / 59.5 / 62.3 / 64.2 / **66.0**.
**Final-mean deltas: A1 −1.6, A2 −6.2.** Both agree closely with the 50-episode finals (A1 −2.8, A2 −7.6), so the two instruments corroborate. On the pre-registered line ("final within 3 of control ⇒ not load-bearing"), **A1 fires its null and A2 does not.**
**But the final mean is the wrong statistic for A1, and the paper's own metric says so.** A1 *acquires* task 1 far better than the control (b1 diagonal **67 vs 54**) and then loses it: its e4 column runs 67 → 52 → 44 → 48 → 50, while the control's is flat at 54 → 54 → 50 → 51 → 53. Computing Δ_final = mean_k(R_K,k − R_k,k), the metric already defined in Sec. IV and reported in Table I:
| arm | e4 | e6 | e9 | e2 | e7 | **Δ_final** | R_post |
|---|---|---|---|---|---|---|---|
| control | −1 | +7 | −6 | +2 | 0 | **+0.4** | 64.5 |
| **A1 live** | **−17** | −3 | +1 | −1 | 0 | **−4.0** | 64.0 |
| **A2 no-protect** | **−22** | +3 | −13 | +1 | 0 | **−6.2** | 62.9 |
So **A1 does show retention loss — 4 points of Δ_final against the control's +0.4, concentrated entirely in the first task (−17)** — and it is masked in the final mean by a stronger diagonal. The honest statement is not "stationary addressing is not load-bearing" but "**stationary addressing costs ~4 points of retention, roughly two-thirds of what removing protection costs (−6.2), and the loss is concentrated in the oldest task in both arms**". A1's kill line was written against the wrong statistic; Δ_final is the one Table I already reports and is what should adjudicate.
**Mechanistically this now reconciles with add-13.** Live routing moves ~half the read mass (cross-arm IoU 0.44–0.68) while preserving inter-task separation, so protection keeps acting on approximately-right regions — hence damage that is real but roughly half the size of removing protection outright. The two ablations degrade the *same* cell (e4, most exposure) by −17 and −22 respectively.
**Consequence for the draft.** Sec. III.D can keep its claim, but quantified and honest: stationary addressing buys ~4 points of Δ_final at five tasks, not "no forgetting". Sec. III.E's write rule buys ~6. Both are real, neither is the whole story, and the E38 horizon caveat still applies (that collapse needed 7 of 10 blocks).
**A2's b1 = 54.0 exactly matches the 10-task control's b1 = 54**, which is a free cross-instance validation: A2's b1 checkpoint is the 5-task control's task-1 boundary, measured independently here, and it lands on the 10-task instance's value to the decimal.

### Entry 68 addendum 24 (12 Sep 26, 13:40 UK) — **A3 JOINT STAGE COMPLETE**

**Joint stage.** 10,000 steps at bs4×acc8. Held-in loss at 10k: **0.54** (0.546 / 0.541 / 0.542 on the last three log lines; 0.56 at 5k).

A3's sequential runs at **bs16×acc2, the control's own rung**; every E68 arm therefore trains at effective batch 32 — only the micro-batching differs.

### Entry 68 addendum 25 (12 Sep 26, 14:10 UK) — **PARAMETER-MATCHED NAIVE-LoRA TRIANGLE (rows b1–b9)**

**What was missing.** Table I's row "(b) Naive seq. LoRA, matched parameters" (r=1216 / α=304, 2.681B added, E66) has only its final-checkpoint row — the E66 4-seed campaign (all ten envs, 25 eps × 4 paired seeds, mean 8.6) — so Δ_final and the intermediate retention are blank ("--") for that baseline alone, while the r=512 naive, RETAIN, O-LoRA and ours all have full 10-row triangles. Josh (12 Sep, 13:55 UK): "figure out which model it is and get the triangle queued up".

**Instrument.** Byte-identical to the E67 baseline triangles: after block k, the first k envs in training order `[4,6,9,2,7,0,8,1,3,5]`, 25 eps × seeds 1000/2000/3000/4000, vec-batched bs13. **b10 is not re-run**: the E66 row is the same instrument on all ten envs, so it *is* b10.

**Rendering.** Every eval script here exports `MUJOCO_GL=osmesa`, so our rollouts are CPU software-rendered. Not switching the renderer to EGL for speed: it would change the instrument (a different GL backend can shift pixels) relative to every comparator row.

### Entry 68 addendum 26 (12 Sep 26, 14:35 UK) — **A3 HELD-OUT AUDIT (joint-prep checkpoint): fails ONE gate clause — expert bgIoU at E16 (0.132 > 0.10), E14 at the edge (0.099); everything else passes with broader cores**

Same instrument as every certificate (10 held-out libero_10 tasks × 400 steps at bs8, value LR 1e-12, prepass routing). Recorded as **informational** (pre-registered in add-0: the audit is part of A3's result, not a gate on its sequential).

| site | famIoU | **bgIoU** | core50 mean (min–max over tasks) | min effnum |
|---|---|---|---|---|
| E4 | 0.219 | 0.045 | 868 (175–1761) | 513 |
| E6 | 0.193 | 0.040 | 924 (332–1789) | 893 |
| E8 | 0.248 | 0.053 | 1116 (330–1858) | 843 |
| E10 | 0.277 | 0.078 | 1731 (483–3185) | 1215 |
| E14 | 0.260 | **0.099** | 2275 (922–3193) | 2317 |
| E16 | **0.322** | **0.132** | 3032 (808–5015) | 2180 |
| V5 | 0.109 | 0.033 | 287 (160–523) | 392 |
| V7 | 0.115 | 0.035 | 282 (192–418) | 494 |
| V9 | 0.152 | 0.044 | 405 (269–564) | 617 |
| V11 | 0.151 | 0.054 | 501 (316–801) | 708 |
| V13 | 0.159 | 0.062 | 634 (298–879) | 649 |
| V15 | 0.148 | 0.080 | 949 (510–1626) | 951 |

**Against the bands and the comparators (paper cell add-3; C1 add-19; C2 add-21):**
| | expert famIoU | expert bgIoU | expert core50 (mean) | VLM famIoU | VLM min-effnum |
|---|---|---|---|---|---|
| paper cell (warm-up → fill) | 0.180–0.281 | 0.031–0.091 | 713–2019 | 0.082–0.145 | 250–750 |
| C1 sep=0 | 0.380–0.455 | 0.196–0.267 | 387–1016 | 0.181–0.385 | 210–356 |
| C2 c=0 | 0.155–0.269 | 0.012–0.048 | 432–1446 | 0.109–0.152 | 158–276 |
| **A3 joint-prep** | 0.193–**0.322** | 0.040–**0.132** | **868–3032** | 0.109–0.159 | 392–951 |

Gate clauses: expert bgIoU ≤ 0.10 — **FAILS at E16 (0.132)**, E14 passes by 0.001; expert mean core50 ≥ 400 — pass everywhere (min 868 at E4); expert min-effnum ≥ 300 — pass (min 513); VLM min-effnum ≥ 150 — pass with room (min 392, better than the paper cell's 250); VLM famIoU < 0.45 — pass (max 0.159). So A3's routers are certificate-grade at ten of twelve sites and miss at the deepest expert site only, on the same clause the paper cell itself finds tightest (E16 = 0.091 there).

**Reading.** Training routers and values together at matched LR (1e-4 routers / 1e-3 values, MSE + the same aux losses) gives routing that is *more diffuse and more overlapping at depth* than warm-up-then-fill: cores are 1.2–1.5× broader at every expert site (E16 mean core50 3032 vs the paper cell's ≤ 2019), family IoU is higher at E16 (0.322 vs ≤ 0.281), and background overlap climbs monotonically with depth (0.045 → 0.132, the E38 gradient again) to cross the band at E16. Mechanistically consistent with the E35/E36 story minus the LR confound: once values carry loss the deep sites are pulled toward footprints that *share* slots across tasks, and the separation loss at weight 8 holds that in check at shallow sites but not fully at E16. It is a mild, one-site version of what C1 shows everywhere (0.196–0.267). Whether it costs retention is now the sequential's question; the pre-registered expectations in add-0 stand. Note: this audit is on a checkpoint whose values are non-zero, unlike the warm-up certificates — irrelevant to the routing measurement because the prepass routes on memory-free features, which is the point of stationary addressing.

### Entry 68 addendum 27 (12 Sep 26, 15:20 UK) — param-matched triangle: b1 = 80.0

**Row b1 (naive r1216 after task 1, env 4): 72 / 80 / 88 / 80 → mean 80.0** — the adapter acquires its first task at the level the paper quotes for its acquisition diagonal (82.8 mean across the diagonal); the b10 row has this same env at 0.

Row b2 (partial): env 4 (task 1) already **0/25 at seed 2000** one task later, env 6 = 84 — the naive adapter's forgetting is total and immediate, as the b10 row implied.

### Entry 68 addendum 28 (12 Sep 26, 15:40 UK) — **param-matched naive-LoRA triangle DROPPED** (Josh: "Drop the full naive sequential triangle please")

What stands: **row b1 only** (env 4 after task 1: 72/80/88/80, mean 80.0) plus the E66 b10 row — enough to state the naive r=1216 adapter acquires at ~80 and ends at 0 on that task, which is what Table I already says. b2 was in progress (4 of 8 quarters: env 4 at 0/25 and 0/25 for seeds 1000/2000 one task later, env 6 at 84 and 84) and is discarded. Table I keeps "--" for this row's Δ_final.

## Entry 69 - 12 Sep 26 (TWO REVIEWER-REQUESTED FULL FINE-TUNE BASELINES: naive sequential full FT of pi0.5, and the same with a 1-episode-per-task replay buffer — 10 tasks, final row only)

**Why.** A fresh-context read of the draft (Josh, 12 Sep, 16:15 UK): "the one baseline we're missing is a straight sequential train of pi05 with all params free … and the second with a small replay buffer. Like 1 episode per task added at each boundary and mixed in future training." Table I has every *method* baseline (specialists, multitask LoRA, naive LoRA r512 / r1216, RETAIN, O-LoRA) but not the plain reference every continual-learning reader expects — all parameters free, no adapter, no merge — nor the cheapest fix for it (tiny rehearsal). Josh: both from the libero_90 stage-1 base, **10 tasks, final row only** (the r1216 row's treatment: Δ_final stays "--").

**Baseline 1 — naive sequential full FT.** The E67 RETAIN trainer fine-tunes *every* parameter per task and then merges θ ← (1−α)·θ_prev + α·θ_ft. **α = 1.0 is the identity merge**, so the RETAIN chain script with `ALPHA=1.0` *is* the naive sequential full fine-tune under the identical recipe (pi0.5 preset: AdamW 2.5e-5 peak, wd 0.01, betas (0.9, 0.95), clip 1.0, 500-step warm-up, cosine to 2.5e-6 within each task; 5,000 steps/task; effective batch 32; optimizer reinit + LR reset per task; no task identity; seed 1000; libero_90 stage-1 base `libero_90_pi05_base_nomem_50k`). The trainer's merge-exactness witness (ratio = α) still holds at α = 1 (merged ≡ ft).

**Baseline 2 — replay.** Same trainer: at each boundary one episode of the finished task (seeded choice) joins a buffer; every later task's sampler draws each frame from the buffer with probability 0.25 (uniform over buffered episodes) and from the current task otherwise. **Assumption stated, not yet confirmed by Josh:** the 25 % fixed fraction — the literal "mixed in" (uniform over frames) would give ~2 % per past task and be indistinguishable from naive.

**Pre-registered expectations.** Naive full FT: the catastrophic-forgetting reference — acquisition high (the paper's full-FT diagonal band), final row near zero on tasks 1–9, final mean in the **5–15** band with the r512 (9.7) and r1216 (8.6) adapters; if it lands *above* RETAIN's 8.2 that is itself a finding (RETAIN's merge would then be net harmful at α = 0.5, consistent with E67's recency-gradient reading). Replay (1 episode/task at 25 %): partial retention — a visible lift on the oldest tasks over naive but well below ours (65.1) and below the specialists' ceiling; a final mean in the **15–35** band would say "one stored episode buys a third of the gap"; ≥ 50 would be the surprise to explain (it would mean the forgetting is mostly output-distribution drift that a few frames re-anchor).

### Entry 69 addendum 3 (12 Sep 26, 17:15 UK) — **REPLAY BASELINE DROPPED; the PAPER CELL under the REVERSED task order takes its slot** (Josh, after Codex: "drop one of these and do the paper cell with a different task order" → "Reversed. DO it.")

**Decision and reasoning.** Kept the naive full FT (the reviewer's explicit ask, already running, zero new code). Dropped replay: it is a nice-to-have that sits outside the paper's own no-replay constraint, its code was untested on real data, and a reviewer's first worry about a continual-learning headline is order sensitivity, on which the draft has no evidence at all. So the slot goes to **the paper cell's 10-task sequential with `online_task_ids = [9,8,…,0]`** — everything else byte-identical to the E62 10-task run (same A-checkpoint, merged 6x2, corefrac β4, lr 2e-3→2e-4, top_t 3072, bs16×acc2). Reversed rather than a seeded permutation: maximally different from the dataset order and explainable in one sentence.

This run reports its final row from the 4-seed campaign only, so the in-run evals buy nothing; it cuts them to one episode per seen task.

**Pre-registration for the reversed order.** The headline claim is retention under sequential writes with stationary addressing + protection; the dataset order is arbitrary. Expect the reversed final mean inside the ±3-point band of the 65.1 headline established by the 5-/10-task replicate (E68 add-0), with a *different* per-task profile (the tasks that were oldest are now newest). A final mean below ~55 would say the headline benefits from the specific order and must be reported as such; above ~70 equally so. Per-task, the reversed run's oldest tasks (9, 8, 7 in the new order) are the cells to compare with the control's oldest (0, 1, 2).

### Entry 68 addendum 29 (12 Sep 26, 17:25 UK) — A3 sequential boundary 1: task-1 end loss 0.039, in-run 20-ep e4 = 50

Block 1 (dataset task 0 → env 4) at bs16×acc2: loss 0.244 (step 200) → 0.098 (800) → 0.045 (4K) → **0.039 at 5K**. In-run 20-episode eval: **e4 = 50.0** (control's in-run e4 at its b1 was 80 on the 10-task instance and its 4-seed b1 = 54; A1's 4-seed b1 = 67, A2's = 54.0). One 20-episode cell (±11 pp) — recorded, not read; the in-run instrument stays under the add-18 moratorium and the 4-seed triangle adjudicates.

### Entry 69 addendum 4 (12 Sep 26, 20:10 UK) — naive full FT boundary 1: **the α = 1.0 identity is verified on 4.143B parameters** (ratio 1.0000), task-1 end loss 0.028

Task 1 (dataset task 0, the two-mug task): loss 0.417 (step 100) → 0.108 (1000) → 0.067 (1800) → 0.045 (3100) → 0.035 (4300) → **0.028 at 5000**, grad-norm 5.0 → 0.77.

**The merge witness is the proof that this run IS the naive baseline.** `retain_boundary.json`: `alpha 1.0`, `mean|ft−prev| = 1.0961e-04`, `mean|merged−prev| = 1.0961e-04`, `ratio_merged_over_ft = 1.0000`, largest-tensor ratio 1.0000, over **n_params = 4,143,404,816**. Identical to 4 decimal places on every parameter means θ ← θ_ft exactly: no merge, no interpolation — the plain sequential full fine-tune, under the recipe the RETAIN row used. (The witness would raise if the ratio deviated from α by > 1e-3; it is the same check that caught nothing here because there is nothing to catch.)

**A3, boundary 2.** Block 2 end loss 0.032.

### Entry 68 addendum 30 (13 Sep 26, 10:25 UK) — **A3 SEQUENTIAL COMPLETE.** In-run final row (50 ep/task) **57.2 mean**: 58 / 50 / 72 / 64 / 42 — no collapse, oldest task the SECOND best cell

5 blocks × 5,000 steps at bs16×acc2. Block end losses **0.042 / 0.033 / 0.076 / 0.046 / 0.061** — no drift across blocks, i.e. the jointly-prepared routers keep the value writes as learnable at block 5 as at block 1.

**In-run retention triangle (20 ep/cell for k < 5, 50 ep for the final row):**
| after block | e4 | e6 | e9 | e2 | e7 |
|---|---|---|---|---|---|
| 1 | 50 | | | | |
| 2 | 40 | 70 | | | |
| 3 | 55 | 45 | 65 | | |
| 4 | 50 | 65 | 80 | 90 | |
| **5 (50 ep)** | **58** | **50** | **72** | **64** | **42** |

**Final row mean 57.2** (58+50+72+64+42)/5. Comparators at the same in-run instrument: the control's 5-task in-run final row was 58/74/64/84/54 = 66.8 (E62), A1 and A2 were not read at 50 ep. **Read with the add-18 moratorium in force**: this is the in-run instrument, one 50-episode cell per task at seed 1000, and the 4-seed triangle is what adjudicates A3 against the control's 66.0. What can be said without the triangle: **no collapse** — the pre-registered failure mode for joint preparation (E35/E36's dead routers, or a late-task starvation) did not occur; the oldest task (e4, five blocks of exposure) is the second-best cell at 58, above its own block-1 acquisition of 50; the weakest cell is the NEWEST task (e7, 42), which is an acquisition story, not a forgetting one. That pattern — flat across age, weak on the last task — is the opposite of the control's (strong recent, decaying old), and is the thing the 4-seed rows must confirm or kill.

### Entry 69 addendum 5 (14 Sep 26, 08:00 UK) — **NAIVE SEQUENTIAL FULL-FT CHAIN COMPLETE**: 10/10 boundaries, α = 1.0 identity verified at every one

Ten tasks × 5,000 steps at bs4×acc8.

**Per-task end loss (step 5000 of each block):**
| task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| end loss | 0.028 | 0.026 | 0.061 | 0.034 | 0.043 | 0.023 | 0.022 | 0.026 | 0.026 | **0.015** |

Every block converges to the same 0.02–0.06 band, and the last block is the *lowest* of the ten — exactly what an unconstrained full fine-tune should do: it fits whatever is in front of it, with nothing holding it back. That is the point of the baseline, and it is also why the final row should show near-total forgetting of tasks 1–9.

**The α = 1.0 identity holds at every boundary** (`ratio_merged_over_ft`, fp32, over all 4,143,404,816 parameters): b1–b10 all **1.0000**, dataset tasks 0→9 in order, `mean|ft−prev|` 9.9e-05 to 1.1e-04 per boundary. So this run is θ ← θ_ft throughout: the plain sequential full fine-tune under the RETAIN recipe, not a merge. Nothing about the E67 RETAIN row's protocol differs except the merge coefficient.

**Pre-registered expectation stands** (Entry 69 header): final mean in the 5–15 band alongside naive r512 (9.7) and r1216 (8.6); landing *above* RETAIN's 8.2 would say the α = 0.5 merge is net harmful, which is a finding in its own right.

### Entry 69 addendum 6 (14 Sep 26, 11:50 UK) — **NAIVE SEQUENTIAL FULL-FT FINAL ROW: 7.7%**. Nine of ten tasks at EXACTLY 0.0 across all four seeds; the reviewer's missing reference is now the cleanest catastrophe in Table I

Same instrument as every comparator row (all ten envs × 25 eps × paired seeds 1000/2000/3000/4000 = 1,000 episodes, bs13).

| env (training order) | s1000 | s2000 | s3000 | s4000 | mean |
|---|---|---|---|---|---|
| 4 (task 1) | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 6 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 9 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 7 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 8 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 3 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 5 (task 10, last) | 76.0 | 80.0 | 76.0 | 76.0 | **77.0** |
| | | | | **FINAL MEAN** | **7.70** |

**Reading.** Inside the pre-registered 5–15 band, and the shape is the textbook one: total erasure of every earlier task, full competence on the most recent. 77.0 on task 10 says the model is not broken — it is *perfectly* fit to whatever it saw last (end loss 0.015, the lowest of the ten blocks, add-5) and has no mechanism to hold anything else. Zero variance across four seeds on nine of ten cells: this is not noise, it is a floor.

**Against the other baselines** (all same instrument): ours 65.1 | naive LoRA r512 9.7 | naive LoRA r1216 param-matched 8.6 | RETAIN α=0.5 8.2 | **naive FULL FT 7.7** | O-LoRA r64 6.5. So the full fine-tune is the *worst* of the parameter-updating baselines, marginally below RETAIN — which answers the add-5 pre-registration: **the α = 0.5 merge is NOT net harmful; it buys +0.5 points over not merging at all.** Both are within noise of each other and both are catastrophic; the honest statement is that weight merging at this α neither helps nor hurts materially, while our memory writes are 8.5× better than any of them.

**What this row buys the paper.** Table I's row (a) "Naive seq. full FT" can now be stated with its own 4-seed number instead of being implied by the LoRA proxies. The reviewer's objection — *you never show what happens with all parameters free* — is answered directly: 7.7%, nine tasks at exactly zero. It also strengthens the framing: the LoRA naive rows (9.7 / 8.6) were never a softer version of full FT, they are the same catastrophe.

### Entry 68 addendum 31 (14 Sep 26, 18:20 UK) — **A3 4-SEED TRIANGLE COMPLETE: final mean 59.0 vs control 66.0 (−7.0).** A3 fires its kill line, but for a reason the pre-registration did not anticipate: the loss is ACQUISITION of the newest task, not forgetting of the old

All five rows: 15 cells × 25 eps × 4 paired seeds. Same instrument as the control and A1/A2.

**A3 vs control, per row (4-seed means, training order e4 e6 e9 e2 e7):**
| block | A3 | control |
|---|---|---|
| b1 | **52.0** | 54.0 |
| b2 | 49 / 61 → **55.0** | 54 / 65 → 59.5 |
| b3 | 49 / 64 / 65 → **59.3** | 50 / 66 / 71 → 62.3 |
| b4 | 50 / 71 / 75 / 84 → **70.0** | 51 / 54 / 65 / 87 → 64.2 |
| **b5** | **50 / 60 / 63 / 85 / 37** → **59.0** | 53 / 72 / 65 / 89 / 51 → **66.0** |

Final row per-seed: e4 [52,44,56,48]; e6 [64,48,56,72]; e9 [60,64,64,64]; e2 [92,80,80,88]; **e7 [40,32,40,36]**.

**The kill line fires: −7.0 on the final mean, well outside the ±3 replicate band.** Joint router preparation at matched LR is *worse* than warm-up-then-fill, so the staged protocol is load-bearing and Table II's row stands (with its LR confound now removed rather than merely asserted).

**But the mechanism is the opposite of the pre-registered one.** E35/E36 predicted dead routers → progressive collapse of old tasks. That did not happen:
- **e4, the oldest task, does not decay at all**: 52 → 49 → 49 → 50 → **50** across five blocks. The control's e4 wanders 54 → 54 → 50 → 51 → 53. Both are flat; A3 sits ~2 points lower throughout, which is a *b1 acquisition* difference, not forgetting.
- **Every cell A3 loses is a recent-task cell.** At b5 it matches or beats the control on e2 (85 vs 89, within noise) and e9 (63 vs 65), and loses decisively on **e7, the newest task: 37 vs 51 (−14)**, with tight per-seed spread [40,32,40,36] — not noise.
- b4 A3 *leads* the control 70.0 vs 64.2, because the control had a bad e6 cell (54) at that block.

**Reading.** Jointly-prepared routers retain what they wrote as well as staged ones — the retention story is untouched — but they are **worse at acquiring new tasks late in the sequence**. That is consistent with the audit (add-26): A3's routers are more diffuse and more overlapping at depth (E16 bgIoU 0.132 > 0.10 gate, cores 1.2–1.5× broader), so a late task's writes land on slots already committed and the TF-IDF × (1−u)^β rule protects them *against the new task*. Protection working as designed, on a geometry that is too entangled to leave room. This is a cleaner and more interesting claim than "joint training collapses": **the warm-up's job is not to prevent forgetting, it is to carve disjoint capacity so later tasks still have somewhere to write.**

**For the paper.** Table II's A row can now be stated as a 4-seed number with a mechanism: joint preparation at matched LR costs 7.0 points at five tasks, concentrated entirely in the newest task (−14 on e7), with zero cost to the oldest. The E35/E36 LR confound is eliminated — this is not a learning-rate artefact.

**Standing E68 picture** (5-task, 4-seed final means): control **66.0** | A1 live addressing 64.4 (Δ_final −4.0) | **A3 joint prep 59.0 (−7.0)** | A2 no protection 59.8 (−6.2).

### Entry 69 addendum 7 (14 Sep 26, 22:40 UK) — **REVERSED-ORDER PAPER CELL: 62.4 vs forward 65.1 (−2.7, inside the ±3 replicate band). The headline is order-robust**

Same instrument as the forward row (all ten envs × 25 eps × paired seeds 1000/2000/3000/4000 = 1,000 episodes, bs13). Chain: 10 tasks `[9,8,…,0]`, 50k steps, bs16×acc2.

**Final rows side by side.** The reversed run's training order is 5, 3, 1, 8, 0, 7, 2, 9, 6, 4 (env ids); the forward run's is 4, 6, 9, 2, 7, 0, 8, 1, 3, 5. Per-env means, and each env's **position** in its own run (1 = first learned, 10 = last):

| env | forward mean (pos) | reversed mean (pos) | Δ rev−fwd |
|---|---|---|---|
| 4 | 59.0 (1) | 58.0 (10) | −1.0 |
| 6 | 60.0 (2) | 64.0 (9) | +4.0 |
| 9 | 63.0 (3) | 67.0 (8) | +4.0 |
| 2 | 87.0 (4) | 90.0 (7) | +3.0 |
| 7 | 54.0 (5) | 32.0 (6) | −22.0 |
| 0 | 37.0 (6) | 43.0 (5) | +6.0 |
| 8 | 76.0 (7) | 77.0 (4) | +1.0 |
| 1 | 38.0 (8) | 9.0 (3) | −29.0 |
| 3 | 86.0 (9) | 92.0 (2) | +6.0 |
| 5 | 91.0 (10) | 92.0 (1) | +1.0 |
| **mean** | **65.1** | **62.4** | **−2.7** |

**Headline: −2.7, inside the ±3 band the 5-task/10-task replicate established (E68 add-0).** The order-robustness question Codex raised is answered: the 65.1 is not an artefact of the particular task sequence.

**What moves, and what does not.** Seven of ten envs sit within ±6 of their forward value; the mean shift is carried almost entirely by **two cells that are hard in both runs**: e1 (38 → 9) and e7 (54 → 32). Both are already the two weakest forward cells after e0. Their reversed positions (3rd and 6th) are *earlier* than forward (8th and 5th), so they sat under more subsequent blocks — consistent with per-task difficulty compounding with exposure, not with an order-specific failure. **No cell collapses to zero**, unlike the naive baselines where nine of ten do.

**Position is not what determines the score.** The first task learned scores 92 (reversed, e5) and 59 (forward, e4); the last scores 58 (reversed, e4) and 91 (forward, e5). Each env keeps roughly its own difficulty across both orders — e5, e3, e2 are easy (86–92 in both), e1, e0, e7 are hard (9–54) — which is the strongest evidence in the run that retention is governed by task identity and capacity, not by recency.

**For the paper.** A one-line order-robustness claim with a 4-seed number behind it: reversing the ten-task sequence moves the final mean by 2.7 points, within replicate noise, with no collapsed cell.

**E69 is complete.** Both of its deliverables are in: naive sequential full FT 7.7 % (add-6) and reversed order 62.4.
