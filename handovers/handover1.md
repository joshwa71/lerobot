# Handover 1 — VLA Memory-Layer Continual Learning (pi05 / LIBERO)

**Written:** 2026-06-17 ~04:50 UTC
**Author:** Claude (session ending here)
**For:** a fresh-context Claude picking up exactly where this left off.

---

## 0. TL;DR / immediate next action

`probes5` (tmux) **finished at 03:27 on 17 Jun** — two 10k pretrains + two held-out audits are on disk. **Your first job: analyse them.** This is the decisive test of a hypothesis we spent the back half of the session building toward: *does fixing the separation-loss estimator (a new cross-batch queue) finally let prior-side routing separation reduce held-out interference WITHOUT collapsing per-task capacity?*

Run the analysis (see §6 for the exact comparison script pattern), comparing:
- `audit_heldout_standard_c0.1_rq512_10k` vs no-queue `audit_heldout_standard_c0.1_10k`
- `audit_heldout_standard_c0.05_sep0.5_rq512_10k` vs no-queue `audit_heldout_standard_c0.05_sep0.5_10k`
- against anchors `audit_heldout_control_40k` (broad) and `audit_heldout_c005_40k` (collapsed/failed).

**The gate (decides the next pretraining move):** GOOD = held-out **L14 famIoU ≤ ~0.28 AND core50 ≥ ~1,500 AND in-run query_intra_sim ≤ ~0.90**. If rq512 hits it → the queue unlocked separation; proceed to a **weight sweep on top of the queue** (e.g. sep 2.0 / contrastive 0.2, possibly 20k). If rq512 **still** doesn't move held-out famIoU → keep iterating **on the pretraining side** — the queue makes stronger separation pressure signal-not-noise, so push the weights harder, add similarity-weighted separation, and/or pretrain longer (§7). The solution stays on the pretraining side.

Then write **Entry 24** in the research log with the verdict.

---

## 1. The project in one paragraph

We're reducing **catastrophic forgetting in sequential (continual) task adaptation** of a VLA policy using **PKM-style memory layers**. Backbone is **π₀.₅ (pi05)**: gemma_2b VLM + gemma_300m action expert, 6.6B params, chunk_size 50. Memory layers (product-key memory, PKM) are attached to **expert layers [8,10,12,14]**, with **LoRA-rank-2 values**, `mem_n_keys=384` (→ slot table = 384² = 147,456 slots/layer), `mem_heads=4`, `mem_knn=36`. Queries are **language-conditioned via FiLM** (`all-mpnet-base-v2` embeddings). Protocol: **(1) joint pretrain** the whole model + memory on a held-in suite, then **(2) sequential adaptation** on a held-out suite where the **backbone + router (keys + query projection) are FROZEN and only the memory LoRA values are trained**, one task at a time, with **TF-IDF gradient masking** (`tfidf_top_t` = #slots updated per batch) limiting which slots get updated.

**Current regime (this session):** pretrain = **libero_90** (90 tasks, held-in), sequential = **libero_10 / LIBERO-Long** (10 tasks, held out of pretraining, the hardest suite — 6 of 10 tasks need *two* full pick-place cycles). 40k pretrain steps, 3000 sequential steps/task, 50 eval episodes/task.

**Hard constraints (decided, DO NOT relitigate or propose around):**
- **The anti-forgetting / separation solution MUST live on the PRETRAINING side** — i.e. shape the frozen router's prior so held-out tasks inherit compact, separated footprints. **Do NOT propose moving the anti-interference work to the sequential side** (no write-time plasticity decay, no sequential-side gradient masking schemes beyond the existing TF-IDF, etc.). Josh has ruled this out repeatedly and emphatically. The sequential phase only trains the LoRA values with the existing machinery; the *fix* is always in the prior.
- **Router (keys + query proj) stays FROZEN during sequential.** Training it slowly was tried earlier and *decimated* prior-task performance (frozen old queries × moved keys silently re-point old tasks' retrieval). Off the table.
- **No per-task parameters, no added parameters of any kind** at adaptation time.
- **Pretrain-task forgetting is acceptable** — pretraining exists only to give the frozen router + memory a good *prior* for clean sequential adaptation. Only the 10 sequential tasks must be protected from each other.
- Also off the table: EWC, replay, hard task-boundary slot allocation, sequential-side soft/hard plasticity decay.

**The goal property:** each sequential task should inherit (from the frozen router) a **compact, well-separated** slot footprint, with cross-task overlap only where it's benign or mutually beneficial — so later tasks' value updates don't overwrite earlier tasks' high-weight reads.

---

## 2. Key files & locations

- **Research log (READ THIS FIRST, Entries 18–23):** `/home/josh/lerobot/projects/research_log.md`. Dated entries, one per experiment cycle. This handover summarizes but the log has the full numbers.
- **Project memory / context docs:** `/home/josh/lerobot/projects/vla-memory.md` (design doc), `research_log.md` (sim), `realworld_research_log.md` (real-robot track, separate).
- **wandb parser:** `/home/josh/lerobot/scripts/parse_wandb.py` — offline `.wandb` reader (no wandb API needed). `WandbRun.from_wandb_dir(path).history_df()` etc.
- **Core code:**
  - `src/lerobot/policies/modules/memory_lite.py` — `HashingMemoryLite` (the PKM memory module): query proj, subkey scores, retrieval, routing losses, contrastive loss, **the new routing queue** (this session).
  - `src/lerobot/policies/modules/memory_config.py` — `MemoryLayerConfig` dataclass (all `--policy.memory_layer.*` knobs).
  - `src/lerobot/scripts/lerobot_train.py` — pretrain loop; flush hook at line ~184 (`_flush_staged_contrastive_queues`).
  - `src/lerobot/scripts/lerobot_sequential_train.py` — sequential loop; TF-IDF masking, online IDF, per-task memory-usage JSON dumps.
- **Run outputs:** `/home/josh/lerobot/outputs/train/<run_name>/` — each has `wandb/`, `checkpoints/<step>/pretrained_model/`, and (sequential/audit runs) `memory_by_task/memory_usage_task_{0..9}.json` + `eval/results.jsonl`.
- **Job scripts:** `/home/josh/lerobot/job_scripts/nebius/libero_90/probes/` — all the probe/audit/pipeline scripts from this session.
- **Logs of running jobs:** `/home/josh/lerobot/outputs/probe_logs/` (`*_runner.log`, per-stage `*.log`).

---

## 3. Environment & ops (how to run things)

- **Always use the conda env:** `source /home/josh/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-memory-updated`. (System python lacks pandas/wandb. Despite CLAUDE.md saying `uv run`, this fork's job scripts all use this conda env.)
- **GPU:** single local **H200 (143 GB)**. Check with `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`. Full pi05 run sits ~108 GB.
- **Long jobs run in detached tmux**, driven by a `run_*_seq.sh` runner that chains pretrains + audits sequentially (single GPU). Pattern: `tmux new-session -d -s <name> 'bash <runner>.sh'`. Logs to `outputs/probe_logs/`.
- **Monitoring pattern:** background a watcher that greps the per-stage log for a progress milestone OR error (`Traceback|RuntimeError|CUDA out of memory`), e.g. wait for `~150/10000` steps to confirm stable startup past the OOM/config window.
- **Step time:** ~3.6 s/step for a 10k pretrain → ~10–11h; an audit ~35 min. A 2-pretrain+2-audit batch ≈ 23.5h.
- **Scheduler gotcha (cost a wasted run earlier):** lerobot **auto-scales** the LR schedule when `steps < scheduler_decay_steps` (`optim/schedulers.py:111`). So a 10k run with `scheduler_decay_steps=40000` runs a *compressed* cosine (warmup 1000 / decay 10000). This is fine and consistent across all 10k probes (they're comparable to each other). But it means **you cannot `--resume` a 10k probe to 40k** — the rebuilt unscaled scheduler jumps LR back up (SGDR sawtooth). For a full run, launch FRESH with `steps == scheduler_decay_steps == 40000`.

---

## 4. The analytical method / workflow (IMPORTANT — how to evaluate a run)

This is the repeatable workflow used all session. Reproduce it; don't invent a new one.

### 4a. Three data sources per run
1. **wandb history** (`parse_wandb.py`) — in-run training metrics every 200 steps. Key keys:
   - `train/mse_loss` (the real optimization signal; `train/loss` includes aux terms with no/zero grad and is misleading — for negatives-only SupCon it even goes negative).
   - `eval/pc_success` (pretrain held-in, 4 eps), `eval/avg_pc_success_seen` + `eval/success/libero_10_overall` (sequential).
   - Memory health: `train/gate_mean` (pi05 saturates ~0.92–0.99 — memory dominates the residual, so overwrites are maximally destructive), `train/mem_usage_effnum_*` (effective # slots used, per layer; **per-batch** perplexity), `train/mem_used_frac_*`, `train/mem_usage_top1_share_*`.
   - Routing: `train/routing_intra_task_support_*` (per-task soft support, per layer — watch L14, it's the highest-overlap/highest-trust layer and resists compaction), `train/routing_inter_task_similarity_mean` (slot-level, lower=more separated), `train/query_intra_sim_mean` (**the collapse meter** — within-task query cosine; control ~0.79, collapsed ~0.99), `train/query_inter_sim_mean`.
2. **Per-task memory-usage JSONs** (`outputs/train/<run>/memory_by_task/memory_usage_task_{k}.json`) — per-task, per-layer, per-slot `{total_accesses, batch_accesses, total_updates, batch_updates}`. `batch_accesses` = #batches in that task's block where the slot was read (this is exactly the online-IDF document-frequency). The pretrain checkpoint also has an aggregate `checkpoints/<step>/pretrained_model/memory_usage.json` (no per-task; `total_accesses`/`batch_accesses` only).
3. **`eval/results.jsonl`** (sequential/audit) — one line per checkpoint: `{"step":N, "task_<envid>":success,...}`. Build the retention matrix from this.

### 4b. The metrics that actually matter (learned the hard way)
- **Per-task FOOTPRINT, at L14, from the JSONs** (PREFIX = `model.paligemma_with_expert.gemma_expert.model.layers.<L>`, key `task_<k>`):
  - `effnum` = `exp(entropy(normalized total_accesses))` — effective # slots the task reads.
  - `core50` = #slots holding 50% of read weight (sort desc, cumsum). **This is the capacity proxy.**
- **Pairwise weighted-read IoU** between tasks i,j: normalize each task's access vector to sum 1, `IoU = Σ min / Σ max`. This **equals the logged `memory_iou/all_modules_mean`** (validated) — it's READ overlap, the forgetting channel. Track the **basket family** (libero_10 tasks t4/t5/t7 = "put both X and Y in basket", near-duplicate instructions) — the worst-overlap cluster.
- **Read-through-overwrite:** fraction of task i's (normalized) read weight on slots that a LATER task j>i updated (`total_updates>0`). This is the *direct* forgetting measure. Old top_t=1536 run: 47–86% for early tasks. Timing-matched: a task collapses in eval at exactly the step a heavy overwriter trains.
- **Subkey decomposition (when diagnosing capacity collapse):** a slot = `i1*384 + i2`. Reshape a task's access vector to (384,384); marginal over axis 1 = half-1 subkey usage, axis 0 = half-2. `eff_keys/half = exp(entropy(marginal))`. Also the **joint/product ratio** = `effnum_slots / (eff_half1 * eff_half2)` — **lower = the two halves are "locked"** (same (i1,i2) co-occur), which is the query-collapse signature.

### 4c. The held-out routing AUDIT (the key cheap instrument)
Because a full sequential run is ~45h, we evaluate a *prior* cheaply by measuring how it routes the held-out tasks **before any adaptation**:
- Script: `job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh <checkpoint_dir> <audit_run_name>`.
- It streams the libero_10 demos through the **frozen** checkpoint via `lerobot-sequential-train` with `memory_value_lr=1e-12` (numerically inert), **no env** (`cfg.env=None` → no rollout eval), no checkpointing, 100 batches/task. Dumps the standard `memory_by_task/*.json` measuring the **pristine prior's** held-out footprints.
- ~35 min/checkpoint. Then compute §4b metrics on the resulting JSONs.
- **Caveat:** control routing is ~flat over training, so `control@40k` is a fair anchor for `probe@10k`. The compact/collapsed anchor is `c005@40k` (and `probeC@10k`).

### 4d. Analysis-script conventions
- Write throwaway analysis scripts to `/tmp/*.py`; run in the conda env.
- **Cache the parsed JSONs to `/tmp/*.npz`** (np.savez_compressed of per-layer (10, 147456) access arrays) — the JSONs are ~50 MB each and slow to parse; caching makes iteration fast.
- Load per layer into `acc[L]` shape `(10, 147456)`; compute effnum/core50/IoU from there.
- **Validate** the pipeline by checking computed mean pairwise IoU == logged `memory_iou/all_modules_mean` (it matched to 4 decimals — do this once per new run family).

---

## 5. What we discovered this session (the narrative arc)

### Entry 19 — the top_t=1536 sequential run (the run we started from)
- libero_90 pretrain (held-in 81.1% @40k, MSE 0.133) → libero_10 sequential at `tfidf_top_t=1536`. **Final avg 34.4%.**
- **Two failure modes:**
  1. **Plasticity ceiling:** diagonal (cold-start) avg only 39.8%; all six dual-pick-place tasks fit at 8–34% (train MSE plateaus 0.21–0.26 vs pretrain 0.133), the three one-cycle tasks fit 68–78%. Even *perfect* retention caps this run at ~40%. This is the bigger lever for absolute performance and is **untouched** by all the routing work below.
  2. **Read-time overwrite, family-clustered:** basket family read IoU 0.23–0.34 (vs ~0.09 background); top_t=1536 made every task write 47K–112K unique slots → 47–86% of early tasks' read weight overwritten by later tasks; gate≈0.98 makes each overwrite maximal. Timing-matched collapses (t5 30→4 and t0 18→4 exactly when t7 trains).
- **Mechanism autopsies:** (a) **TF-IDF/online-IDF is functionally a no-op at this scale** — `log((B+1)/(DF+1))` compresses a 60:1 DF ratio to a 3× score penalty while TF varies ~86×; idf_exponent=2 would leave 97% of damaging writes intact (measured). Don't sweep idf_exponent. (b) **Hard write-veto is zero-sum** under broad routing — protecting predecessors blocks 80–96% of the new task's own writes. (c) Gradient-dilution rejected as the diagonal driver (concentration uncorrelated with fit).

### Entry 20 — first probes (locality vs SupCon)
- **Probe L (locality weight 0.25→1.0):** FAILED — at 4× weight L14 support barely moved (locality loss is ~0.2% of the objective; MSE outbids it). Locality-as-a-loss is a dead lever; retired.
- **Probe C (sample SupCon contrastive 0.01→0.05 + `contrastive_negatives_only=true` + queue 512):** looked like a **breakthrough** — held-out family IoU 0.349→0.190 (−46%), footprints ~7× smaller, and crucially it moved the **slot-level** metrics (not just query-space, unlike all prior attempts). We graduated it to a full 40k.

### Entry 21 — the SupCon recipe at full scale = DISASTER (the pivotal lesson)
- Full 40k SupCon-negonly pretrain → libero_10 sequential at top_t=512. **Killed at task 6; cold-start avg 14% vs old 31.2%.**
- **Forgetting was SOLVED** (read IoU 0.0165 vs 0.107; read-through 0–6% vs 59–86%) **but plasticity was DESTROYED** (per-task min MSE 1.5–2.6× worse; effnum ~500 vs ~2700).
- **Root cause: over-compaction.** `negatives_only=true` removed the intra-task uniformity term → the SupCon positive pull collapsed within-task queries (`query_intra_sim` 0.99) → routing collapsed to ~7× smaller footprints → per-task capacity died in the frozen-backbone regime. **Capacity and interference are the same quantity (routing breadth) read two ways; we crushed both.**
- **Key reframe:** the audit *had* the capacity number (core50 351) and we read "7× smaller" as purely good. The audit needs a **capacity gate**, not just an interference gate. Also confirmed **top_t was NOT the cause** (per-batch read breadth collapsed 5520→784 at L14; top_t=512 was matched-to-generous for the collapsed prior — write coverage was actually *better* than the old run's 28%).

### Entry 22 — 2-knob isolation (negonly dose vs structure)
- **P1 (negonly 0.025):** still capacity-dead (core50 696, query_intra 0.98 — dose-insensitive; negonly has no anti-collapse term at any weight). negonly branch **dead**.
- **P2 (standard SupCon 0.05, negatives_only=false):** capacity preserved (core50 1465, query_intra 0.91) **but ~zero separation** (famIoU 0.338 ≈ control 0.349). Standard SupCon's same-task-in-denominator protects capacity but neutralizes the inter-task push at this weight.
- **Bracketed:** negonly couples capacity+separation (bad frontier); standard decouples but under-separates.

### Entry 22 follow-up (probes3) — pushing the capacity-safe variant
- **P3 (standard 0.1):** worse — lower capacity (core50 1162) AND worse-than-control famIoU (0.381). Cranking standard contrastive walks toward collapse.
- **P4 (standard 0.05 + `routing_inter_task_separation` 0.25→0.5):** best capacity of any contrastive run (core50 1752, query_intra 0.91) but **still no held-out separation** (famIoU 0.342 ≈ control). Direct slot-space separation at 2× default bought nothing on held-out tasks.
- **Provisional (then-)conclusion:** prior-side separation can't decouple — the only thing that ever separated held-out families was the *collapse* (an artifact of tiny footprints barely overlapping), and that kills capacity. Generalization gap: separation losses push the 90 *seen* tasks apart (P4 in-run seen-task sim 0.102 vs control 0.143) but that does NOT transfer to making *held-out* near-duplicate instructions route apart under a frozen map.

### Entry 23 — the estimator bug that reopened everything (current frontier)
- **Josh's catch:** the **separation/locality loss has NO cross-batch queue** (only the *contrastive* loss does). It operates on the current micro-batch only. With batch 32 over 90 tasks (random sampler): **~27 distinct tasks/batch, ~1 sample/task**, and any *specific* task pair co-occurs only ~(27/90)² ≈ **9% of steps**. So the separation loss has been pushing apart **noisy, single-observation, sparsely-covered** per-task histograms. The Entry-22 "can't decouple" conclusion may be premature — separation never had a fair estimator.
- **The math distinction (why separation ≠ "SupCon negonly=false"):** SupCon acts on per-sample **query vectors** in continuous k_dim *before* the key lookup and carries an intra-task **pull** (collapse risk); the routing separation loss acts on per-task **slot-occupancy histograms** *after* retrieval and has **no intra-task term** (intra is the separate locality knob). So separation can in principle reduce overlap by **translating** broad footprints to disjoint regions (no shrinkage) — the decoupling we want — *if* it has a clean per-task estimate.
- **Fix implemented (the cross-batch routing queue) — see §8 for code detail.**
- **probes5 launched** = exact rerun of P3/P4 with the only delta being `routing_query_queue=512`. **This is the clean A/B test of the whole hypothesis. It is now COMPLETE and awaiting analysis (§0).**

---

## 6. Current state & exact next-step commands

`probes5` finished 03:27 on 17 Jun. Artifacts:
- Pretrains: `outputs/train/libero_90_pi05_8_10_12_14_probe10k_standard_c0.1_rq512/` and `..._standard_c0.05_sep0.5_rq512/` (10k checkpoints + wandb).
- Audits: `outputs/train/audit_heldout_standard_c0.1_rq512_10k/` and `audit_heldout_standard_c0.05_sep0.5_rq512_10k/` (each has `memory_by_task/*.json`).

**Comparison anchors already on disk** (audits): `audit_heldout_control_40k` (broad), `audit_heldout_c005_40k` (collapsed/failed), `audit_heldout_standard_c0.1_10k` + `audit_heldout_standard_c0.05_sep0.5_10k` (the no-queue versions = the direct A/B).

**To analyse** (reuse the session's pattern): write `/tmp/audit_probes5.py` modeled on the prior audit scripts — load each audit's `memory_by_task` JSONs into cached npz `(10,147456)` per layer (L8/L12/L14), compute per-task **core50** and **effnum** (mean over 10 tasks), the **pairwise weighted-read IoU** matrix → **family mean (t4/t5,t4/t7,t5/t7)** and background mean, and the **subkey eff_keys/half + joint/prod ratio**. Print a table with rows: control@40k, c005@40k(failed), no-queue P3/P4, rq512 P3/P4. Also pull in-run `query_intra_sim_mean`, `routing_inter_task_similarity_mean`, `mem_usage_effnum_mean`, `mse_loss` @10k from each pretrain's wandb (via `parse_wandb.py`).

**Reference numbers to beat (held-out L14):** control core50 2643 / famIoU 0.349; failed-negonly core50 511 / famIoU 0.133; **no-queue P4 core50 1752 / famIoU 0.342**. **GOOD = famIoU ≤ ~0.28 with core50 ≥ ~1,500.** The whole question: did the queue move famIoU down from ~0.342 while holding core50? Watch for the **shrinkage shortcut** (famIoU drops only because joint/prod ratio falls / core50 shrinks — that's compaction, not the win).

Then **decide** per §7 and write **Entry 24**.

---

## 7. Decision tree after probes5

- **If rq512 P4 (or P3) clears the gate (famIoU ≤ ~0.28, core50 ≥ ~1,500):** prior-side separation works once fairly estimated. Next: a **weight sweep on top of the queue** — Josh's earlier proposal was **sep 2.0 + contrastive 0.05** and **sep 0.5 + contrastive 0.2**, and consider **20k steps** (10k may under-train the router; cheap to test). Then graduate the winner to a **fresh 40k** (NOT a resume — §3 scheduler gotcha) → 40k held-out audit (with capacity gate) → **a 1-task ~500-step plasticity probe** (cheap final capacity check) → full sequential. **Set `tfidf_top_t` from the winning prior's measured per-batch L14 effnum** (target ~70–90% write coverage), NOT a carryover — a broad-but-separated prior will need ~1536 again (Entry 22 note (d)).
- **If rq512 still doesn't move held-out famIoU (now with a fair estimator):** the queue removes the "estimator was too noisy" excuse, which makes *stronger* pretraining-side separation pressure meaningful (it was noise-limited before). Stay on the pretraining side and exploit that:
  - **(a) Weight sweep on top of the queue** — push `routing_inter_task_separation` well past 0.5 and/or contrastive up (Josh's earlier proposal: sep 2.0 / contrastive 0.05, and sep 0.5 / contrastive 0.2). With a clean gradient these should now bite instead of separating noise.
  - **(b) Similarity-weighted separation** — weight each task-pair's separation term by the language-embedding similarity of their instructions, so the budget concentrates on lookalike families (the basket cluster) rather than already-distinct pairs. The transferable property we actually need.
  - **(c) Longer pretraining** (10k→20k→40k) so the frozen router has more steps to learn a map that separates near-duplicates; and **(d)** tune the queue itself (depth, per-token vs aggregated granularity).
  - The router stays frozen and the fix stays in the prior. **Do not move the anti-forgetting work to the sequential side — explicitly off the table (§1).**
- **Separate axis — the plasticity / current-task-fit ceiling (NOT the forgetting problem):** the dual-cycle tasks cap the diagonal at ~40% regardless of interference. Josh's own pre-registered levers for this (Entry 22): more steps/task (3000→5000; MSE still falling at block ends), higher memory_value LR (floor 1e-4→2e-4, then peak 1e-3→2e-3), and lora_rank 2→4 (expensive). This is about fitting the current task better, distinct from the separation/forgetting work above.

---

## 8. The code change this session (routing queue) — detail

**What:** a cross-batch FIFO that gives the separation loss a well-estimated, all-task-covering per-task reference distribution (fixing the ~1-sample/9%-coverage problem in §5/Entry 23).

**Config:** `MemoryLayerConfig.routing_query_queue: int = 0` (in **samples**; 0 = off = original behavior). CLI: `--policy.memory_layer.routing_query_queue=512`.

**memory_lite.py additions:**
- `__init__`: routing-queue buffers (`_routing_queue_q` = per-token detached queries `(cap_rows, heads, k_dim)`, `_routing_queue_labels`, ptr/count, `_pending_routing_batches`). cap_rows = `routing_query_queue * tokens_per_sample`, sized lazily.
- `_ensure_routing_queue / _get_routing_queue / _stage_routing_queries / _enqueue_routing_queries` — mirror the contrastive-queue methods. **Highest granularity = per-token** queries; **global FIFO** (Josh's choices). Staging guarded by `_is_checkpoint_recompute()` (no double-enqueue under grad checkpointing).
- `flush_staged_contrastive_queries` — **extended to flush the routing queue too** (the existing flush hook at `lerobot_train.py:184` already iterates all submodules every step). **BUG FIXED:** the original had an early `return` when no *contrastive* entries were pending, which silently skipped the routing flush whenever contrastive was off — restructured to flush both independently.
- `forward`: stages per-token routing queries after the routing-loss block (when `routing_query_queue>0`).
- `_compute_routing_losses`: branches to **`_routing_losses_queued`** when `routing_query_queue>0` (else byte-identical old compact path).
- **`_routing_losses_queued`** (new, vectorized): builds **dense** per-task histograms over the full 147,456-slot space (numerically identical to the old compact path; cosine/entropy over zeros = no-ops). Current batch = **differentiable** (carries gradient via batched `scatter_add`). Queue = recomputed against **CURRENT keys** under `no_grad`, chunked (4096 rows), into **detached** per-task reference histograms covering all recently-seen tasks. Separation = push current (diff) away from references (detached), j≠i, as **one `einsum('ihs,jhs->ij')/heads`** with an i==j mask. Locality + global-balance stay on current histograms.
  - **Why recompute-vs-current-keys (not store frozen histograms):** the separation loss *moves* the keys, so frozen references would lag/oscillate. Storing per-token queries + recomputing is the right call (this is the "store kvs and recompute" Josh asked for; no grad to the stored queries, gradient via current batch only).

**Verification done:**
- Isolated **smoke test** (tiny module): queue populates to cap; **a single-task batch still gets a separation loss via references** (the coverage fix); gradients reach query_proj + keys; checkpoint-recompute guard holds; **queue-off path numerically identical** to old; **vectorized einsum identical** to the per-pair loop. (This is what caught the flush bug.)
- **Measured performance on the real 7B run: NO overhead** — 3.62 s/step with full 512-queue vs ~4.0 s/step no-queue. The routing loss is a rounding error next to the transformer.
- **Known remaining micro-inefficiency (not fixed, not a bottleneck):** the per-chunk task-label remapping uses `.tolist()` + a Python comprehension → a few host↔device syncs/step, fully masked by the transformer cost. Clean fix if ever needed (bigger queue / smaller model): `torch.unique(..., return_inverse=True)` + GPU-side masking. Safe to apply between runs (running processes won't pick up edits; re-smoke-test after).

**Git:** changes are uncommitted working-tree edits to `memory_lite.py` + `memory_config.py` (this repo is the lerobot fork; CLAUDE.md says commit/push only when asked — they haven't been). `ruff` is not on PATH; the module imports/runs clean.

---

## 9. Mental models worth keeping (so you don't relearn them)

1. **Routing breadth is one axis with two faces.** Broad routing = high capacity + high interference (control prior, 34%, interference-limited). Compact routing = low interference + low capacity (negonly prior, ~12%, capacity-limited). The target is the middle; the whole game is reaching it *without* the two being forced to move together.
2. **Capacity that matters is state-conditional.** A task can have a 2,900-slot aggregate footprint but if `query_intra_sim≈0.99` it pulls nearly the same ~36-slot mixture every step → the memory is ~a per-task constant, not a state→action map. Aggregate footprint *overstates* usable capacity when queries are collapsed. (effnum/core50 are aggregate; pair them with query_intra_sim.)
3. **Slot collapse is multiplicative.** A slot = (subkey1, subkey2). Query collapse both concentrates each half (~2×) AND correlates the two halves (joint/prod ratio drops) → slot effnum falls ~5× from a ~2× per-half drop. Watch the joint/prod ratio to distinguish "translated, still broad" from "shrunk/locked."
4. **Seen-task separation ≠ held-out separation.** Every separation lever moves the 90 seen tasks apart; the test is always whether it transfers to held-out near-duplicate families under the frozen router. Only the held-out audit answers this.
5. **`top_t` binds relative to read breadth.** It only "reduces capacity" if per-batch effective reads > top_t. Set it from the prior's measured per-batch L14 effnum (~70–90% coverage), never as a fixed carryover.
6. **The audit is necessary but not sufficient.** It tells you the prior is in the healthy routing band; it does NOT prove sequential success. Always gate on capacity AND interference jointly, and follow a passing audit with a cheap plasticity probe before a 45h sequential.

---

## 10. Open threads / cautions

- **Don't `--resume` 10k probes to 40k** (scheduler auto-scale, §3). Fresh runs only for 40k.
- **Don't sweep idf_exponent or hard write-vetoes** — both measured as ~no-ops / zero-sum (Entry 19/21).
- **Don't reintroduce `negatives_only=true`** — capacity-dead at any weight (Entry 22).
- **`train/loss` is misleading** for SupCon runs (negative / includes zero-grad aux terms). Use `train/mse_loss`.
- The realworld track (`realworld_research_log.md`) is separate; this session was sim-only.
- Memory headroom is fine (~108 GB / 143 GB) but the dense-histogram routing-queue path adds ~2 GB; if a future config raises `mem_n_keys` a lot, revisit the dense (147k) histograms in `_routing_losses_queued`.
