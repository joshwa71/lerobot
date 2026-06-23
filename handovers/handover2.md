# Handover 2 — VLA Memory-Layer Continual Learning (pi05 / LIBERO)

**Written:** 2026-06-19 ~14:35 UTC
**For:** the next Claude. Builds on **handover1.md** (still valid for: project framing §1, file map §2, ops/env §3, the analytical workflow §4, the held-out audit §4c, and the routing-queue code §8). This covers only what changed since — research-log **Entries 24–26**.

---

## 0. Immediate state / first action

A **graduation run is LIVE** in tmux `sep5_full` (log `outputs/sep5_full.log`), launched 19 Jun ~14:28, ETA **~3.7 days** (~23 Jun). It's a two-stage script (40k libero_90 pretrain → libero_10 sequential), confirmed stepping cleanly.

**Your jobs, in order:**
1. **Mid-flight (after stage 1, ~44h):** run the held-out audit (`probes/audit_heldout_routing.sh <40k_ckpt> audit_heldout_sep5_40k`) on the pretrain `last` checkpoint → confirm the sep5 prior **held under full LR decay** (target: L14 famIoU ≈ 0.264, core50 ≈ 2679, not eroded toward control 0.349/2643). Runnable while stage 2 trains. Also check held-in eval @20k/40k vs control 76.4/81.1.
2. **When sequential finishes:** build the **retention matrix** from `eval/results.jsonl` (per-task init→final). This is the real test.

Script: `job_scripts/nebius/libero_90/combined/pi05_libero_10_4_layer_film_lora2_knn36_40k_c0.05_sep5_noloc_rq512_topt1536.sh`.

---

## 1. The arc since handover1 (Entries 24–26)

handover1 ended at "analyze probes5 (rq512)". The thread from there:

- **Entry 24 — probes5 verdict + a metric trap.** The rq512 queue (Entry 23) did NOT move held-out famIoU at sep ≤ 0.5 (P3'/P4' ≈ control). BUT we diagnosed why the in-run separation metric *looked* worse with the queue: it's an **estimator artifact** — the no-queue path logs sparse-vs-sparse in-batch cosine (under-reads overlap, flatters separation); the queue logs sparse-vs-dense all-task cosine (honest, higher). **Never compare `routing_inter_task_similarity` across queue on/off — use the held-out audit.** A loss-magnitude audit showed: contrastive is the big aux term + the compaction knob; separation is real (~30% of MSE at sep0.5) but barely responded to 0.25→0.5 → "too weak, not saturated."
- **Entry 25 — sep=2.0 DECOUPLES (the turning point).** Pushing sep 0.5→2.0 (rq512, c0.05) dropped held-out famIoU 0.349→**0.311** *while capacity rose* (core50 1501→2368), zero fit cost, no shrink signatures. **First prior to reduce held-out interference without collapsing capacity.** Launched a 4-probe batch (P7–P10) to isolate the knobs.
- **Entry 26 — sep curve + graduation (today).** Full sweep: famIoU falls monotonically 0.350→0.311→0.309→**0.264** as sep 0.5→2→3→5, capacity *rises* (core50 →2679 ≥ control), no turnover, zero fit cost — **sep=5 (P9) CLEARS the gate** (famIoU ≤ 0.28 AND core50 ≥ 1500) via **translation, not shrinkage** (j/p ratio rises 0.36→0.39, effnum→control). Graduated it (see §0).

---

## 2. Decisions & mental models locked this session

1. **The breadth axis is NOT one-dimensional** (revises Entry 21). Capacity and interference *can* be decoupled — with a clean estimator (rq512) + strong separation, footprints **translate apart while staying broad**, lowering overlap *and* raising capacity. The win mechanism is translation; the failure mode to guard against is the shrink-to-disjoint shortcut (watch core50/effnum down + j/p ratio down).
2. **Division of labor (clean): contrastive = intra-task compaction (footprint *size*); separation = inter-task translation (*position*). Both required.** P7 (contrastive=0) sprawled (core50 7700, famIoU 0.482, worse than control) — without compaction, footprints are too big to keep disjoint. Contrastive has a Goldilocks: too much (negonly / 0.1) → collapse; none → sprawl; **0.05 is the pocket.**
3. **Locality is dead** — P10 (loc=0) ≡ P6 (loc=0.25) on every metric. Dropped permanently from the recipe.
4. **The query uses the hidden state, not just language** (corrects handover1/Entry-19 "FiLM-on-language" shorthand): `q = proj(x)·(1+γ(lang)) + β(lang)`. The scene (objects on the table) provides routing discrimination, so the "irreducible overlap floor" is softer than first claimed.
5. **famIoU is held-out↔held-out** — the libero_10 basket family (dataset task_index 4/5/7, near-identical "put both X and Y in basket" strings). Cause routes through libero_90's single-object basket primitives (frozen language-conditioned router maps lookalikes to the shared basin). Residual family overlap (~0.26, ~3× bg) is increasingly **genuine compositional sharing** — chasing it to zero would fight useful reuse.
6. **The winning recipe:** `c0.05 + sep5.0 + locality-off + rq512`, knn36/topk36, lora_rank2, layers [8,10,12,14], n_keys384. (Note: contrastive read as **0.05**, not the "0.5" Josh typed for probes 8–10 — 0.5 would over-compact; 0.05 made a clean factorial and the results confirmed it.)

---

## 3. What to expect from sep5_full

- **Interference: much improved.** The prior overlaps far less than the Entry-19 baseline (held-out famIoU 0.264 vs 0.349, bg 0.087 vs 0.127). Expect the retention matrix to show **no catastrophic early-task collapse** — specifically watch the t5→t7 cliff that hit at step 24000 in Entry 19, and read-through-overwrite for t0–t5.
- **Absolute performance: still capped by the plasticity ceiling (~40%), UNCHANGED.** This whole thread attacked *interference*, not the dual-cycle diagonal ceiling (Entry 19: six tasks need two full pick-places, fit only 8–34% even when current, rank-2 + frozen backbone + OOD composition). So the win shows as **retention** (final closer to peak), not a higher peak. Don't expect >~45% final; the bottleneck moves to plasticity.
- **top_t=1536 is deliberate** (broad-but-separated prior → right write budget per Entry 22(d); safer than Entry-19 because overlap is lower). Re-derive from per-batch L14 effnum only if overwrite climbs.

---

## 4. Next experiments (priority after sep5_full)

1. **If retention is good but absolute lands ~40%:** the bottleneck is now plasticity, not interference. Levers (sequential-only, cheap, Entry 19/22): **steps/task 3000→5000**, **value_lr floor 1e-4→2e-4** (then peak 1e-3→2e-3), and **lora_rank 2→4** (memory-heavy). These are now safer to push because the compact/separated prior leaks far less write-pressure into other tasks' cores.
2. **Reserve separation levers** (if you want famIoU lower without fighting genuine sharing): **per-layer separation weighted onto L14** (the resistant layer), and **hidden-state vs language reweighting** in the query fusion (so lookalike-language tasks separate by scene). Both pretraining-side.
3. **20k-vs-40k question** (Josh's): is router-training duration a bottleneck for separation/fit? Cheap to probe if the 40k prior looks under-trained.
4. The sep curve hadn't turned over at sep=5 — sep~8 might push famIoU lower, but diminishing returns vs the genuine-sharing floor; not a priority.

---

## 5. Gotchas / don'ts (in addition to handover1 §3, §10)

- **Don't read in-run separation across queue on/off as comparable** (Entry 24 estimator artifact). Held-out audit is ground truth.
- **Contrastive is load-bearing — don't drop it to 0** (P7 sprawl). Keep 0.05. Locality, by contrast, is safe to omit.
- **effnum/core50 are AGGREGATE — pair with `query_intra_sim`** (handover1 mental-model #2). sep5's query_intra is ~0.91 (healthy, not the 0.99 collapse), so its broad footprint is real capacity, but confirm with the 1-task plasticity probe if in doubt.
- **Scheduler auto-scale** still bites: don't `--resume` a 10k probe to 40k; launch fresh with steps==decay==40000 (the graduation script does this correctly).
- **Disk:** cleaned to ~769G free (dead-probe *checkpoints* deleted; all `wandb/` + all `audit_heldout_*` retained). The sep5_full run will write ~100G+ (40k pretrain ckpts + sequential per-task ckpts).
- **Code:** the routing-queue edits (`memory_lite.py`, `memory_config.py`) are still **uncommitted** working-tree changes (handover1 §8). Validated across probes 6–10 + the live run.

---

**One-line status:** sep=5 prior cleared the held-out interference gate for the first time (famIoU 0.264 / core50 2679, via translation); it's graduating to a full 40k + libero_10 sequential right now — the open question is whether the retention win survives full training, and the plasticity ceiling is the next frontier.
