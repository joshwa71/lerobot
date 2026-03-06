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
- To test this, I added a new **routing compactness** loss on PQ marginals with CLI control via `routing_compactness_weight` (default `0.0`).
- New ablation family now being run at weights **0.5, 1, 2**:
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_compactness_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_compactness_0.5.sh`
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_compactness_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_compactness_1.sh`
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_compactness_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_compactness_2.sh`
  - with matching sequential scripts under `job_scripts/smolvla-memory/sequential/2_layer/routing_compactness_exp/`
- After further discussion, the more important anti-forgetting term is **routing separation**, not compactness alone:
  - compactness says each task should route locally
  - separation says different tasks should route to different regions
- So compactness is now being treated as a **secondary control ablation**, while the main next test is a routing-separation sweep at weights **0.5, 1, 2**:
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_separation_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_separation_0.5.sh`
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_separation_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_separation_1.sh`
  - `job_scripts/smolvla-memory/pretrain/2_layer/routing_separation_exp/pretrain_12_14_film_lora_2_sample_contrastive_1_routing_separation_2.sh`
  - with matching sequential scripts under `job_scripts/smolvla-memory/sequential/2_layer/routing_separation_exp/`
- Also fixed the LoRA corruption path so corruption is applied to the **adapter output before the shared gating/aggregation path**, which is a better match to the failure mode we want to test than corrupting the low-rank hidden activations.
