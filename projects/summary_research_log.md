# Summary of the VLA Memory Research Log (Entries 0–67, February to September 2026)

This document is the one-stop reference for `research_log.md` (67 numbered entries plus addenda, roughly 9,000 lines). It is written so that a reader who has not opened the main log can find every headline number, the paper configuration, every baseline, and the history of every hyperparameter, and then go to the exact entry for the full context.

**How references work.** `E62` means the section headed `## Entry 62` in `research_log.md`; `E62 add-3` means `### Entry 62 addendum 3` under it. The Standing Reference at the top of the log (added 2 August) is cited as `SR`. To jump to an entry: `grep -n "^## Entry 62" lerobot/projects/research_log.md`. Numbers are quoted as they stand after all corrections in the log; where a number was later corrected, the corrected value is given and the correction is cited (see §11, the errata register).

**Reading order for a new session.** §2 (paper cell spec), §3 (paper cell results), §4 (baselines), then §12 (status). §5 is the per-hyperparameter history for when a specific knob is in question. §6 is the chronological narrative for the "why".

---

## 1. The problem and the fixed constraints

The method attaches sparse product-key memory modules to selected MLP sites of a vision-language-action policy, so that new tasks can be learned one after another by updating only memory values, with a frozen backbone, no task identity at inference, and no catastrophic forgetting. The prototype ran on SmolVLA (E0–E17), then moved to π₀.₅ (pi05: gemma-2B VLM plus 300M action expert, 6.6B parameters) on LIBERO (E18 onward), and in the final weeks was ported to a real WidowX AI arm (E65).

Each memory module replaces an MLP with `MLP(x) + Memory(x)`. Retrieval is product-key: a query is split into two halves, each half finds its top-k sub-keys, the Cartesian product forms candidate slots, and the final top-k slots are read with softmax weights. Slot values are rank-2 LoRA transforms of the hidden state (E0, E4), which is what "value" means for almost the whole log.

Constraints fixed early and held throughout (SR, E19):

- Router (keys and query projection) frozen during sequential adaptation. Key training online was tried once and silently re-pointed earlier tasks' retrieval (E19).
- No per-task parameters, no task identity at inference.
- EWC, replay, and hard task-boundary slot allocation are off the table.
- "Train on exactly the same data": no observation-space augmentation. Hidden-state noise on the value path's input was the one allowed exception (E58), later dropped from the recipe (E62 add-7).
- Forgetting of pretraining tasks is out of scope; only the sequential tasks are protected from one another.

Two write-side mechanisms recur. TF-IDF masking restricts each optimizer step's gradient to the top-t slots by term-frequency score (E1, E4). Protection multiplies that score by `(1−u)^β`, where `u` is a slot's usefulness to prior tasks (E27); the `corefrac` normalisation (E41 build, E52–E53 default) puts whole prior-task cores at `u=1` so they score zero and are structurally excluded from the write mask.

Environment mapping (SR): libero_10 dataset task_index → env is {0:4, 1:6, 2:9, 3:2, 4:7, 5:0, 6:8, 7:1, 8:3, 9:5}; train order is task_index ascending. "Front-5" = envs 4, 6, 9, 2, 7. The basket family (near-identical scenes) is envs 7 (soup+cheese, the hub), 0 (soup+sauce), 1 (cheese+butter).

---

## 2. The simulation paper cell: full specification with log references

The paper configuration is the **merged 6×2** model (E62). Every setting below is stated with the entry where it was chosen or last justified. Chain script: `joint_merged6x2_e468101416_v579111315_prepass_full_chain.sh` (E62); common bodies `joint_rwarmup_common.sh` and `joint_aphase_seq5_common.sh`. The 10-task sequential run is `libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k` (E64 add-12); the 5-task run has the same name with `seq5`.

### 2.1 Backbone and Stage 1

| setting | value | reference |
|---|---|---|
| Base model | pi05 (PaliGemma 2B VLM + 300M gemma action expert, 6.6B params, chunk 50), hub snapshot `9e55186` | E18, E30 |
| Stage 1 | full fine-tune of raw pi05 on libero_90 (no memory), 50k steps, bs32, gradient checkpointing, warmup 4k / decay 50k, pi05 default LR, bf16. Checkpoint `libero_90_pi05_base_nomem_50k` | E31 (recipe), E33 (protocol), E34 (result) |
| Stage-1 zero-shot on libero_10 | 10.6% mean; 0–2% on the collapse-prone tasks (the no-adaptation floor) | E34 |
| Substrate value | +10.6 even to full fine-tuning (67.6 → 78.2) | E60 add-9 |

### 2.2 Memory architecture

| setting | value | reference |
|---|---|---|
| Expert (action expert) sites | MLP layers [4, 6, 8, 10, 14, 16] | E62 (layout), E60 (deep 14/16 sites), E61 add-6 (share criterion) |
| Expert sharing | `share_groups` [[4,6],[8,10]] (one key/value table per pair); layers 14 and 16 solo | E61 (mechanism), E61 add-3/7 (shared-deep kills e7), E62 |
| VLM (paligemma LM) sites | MLP layers [5, 7, 9, 11, 13, 15] | E59 add-6 (L7 optimum, V5 viable), E60 (V5 cleared), E62 |
| VLM sharing | `vlm_share_groups` [[5,7],[9,11],[13,15]] | E61 add-6, E62 |
| Totals | 12 sites, 7 tables, 2.6837B value parameters measured (quoted as 2.8B in the log's tables; 12.5% under the 3.2B paper budget) | E62, E66 |
| Bank size | `mem_n_keys` = 256 per table (65,536 slots), both towers | E44 (right-sized from 384), E46 (scaling law), E48 (sizing rule: keep per-half subkey demand under ~55–60% of n; 192 and 128 dead) |
| Value type / rank | `value_type=lora`, `lora_rank=2`, `vlm_lora_rank=2` | E0/E4 (LoRA values), E33 (expert rank 4 gave +2, closed), E50 (VLM rank 4 gave −10, closed) |
| Heads | `mem_heads=4` | E1 (unchanged since) |
| Expert knn | `mem_knn=36`, `routing_loss_topk=36` | E15–E17 (36 with top_t 1536), E16 (alignment rule) |
| VLM knn | 16 (topk aligned to 16) | E44 (initial), E46 (alignment codified), E49 (36 lost; closed at 16) |
| Gate | kept (`mem_gated=true`); SwiLU-gated output projection | E40–E41 (removal tested, null, retired) |
| Per-slot affine bias | off (`lora_slot_bias=false`) | E40–E41 (null) |
| VLM text span | memory attached to the last 200 prefix positions (instruction + state-as-text), pad positions masked via `token_mask` | E44 (build), E44 add (pad-contamination fix) |
| VLM router key | `vlm_router_pool=anchored`, weights (1.0, 0.5): state region routed on one per-sample key = 1.0·nrm(pooled instruction hidden) + 0.5·nrm(pooled state hidden), instruction tokens per-token; value path per-position | E45 (build), E46 (winner) |
| Route-once | `vlm_route_once=true` downstream of warm-up (compact path; numerically identical with a frozen router); `false` at warm-up | E46 (build), E47 (dedup collapse), E47/E48 (flag) |
| Expert router conditioning | expert text anchor: `expert_anchor_pool=text`, `expert_anchor_weight=0.40`; FiLM off (`lang_to_query=false`, no mpnet embedder). Anchor at expert layer j pools LM layer j | E53 (built, w0.5 over-compacts), E54 (U-curve; B = w0.40 + sep8 passes), E55–E56 (B = the router), E59 add-2 (anchor-source pairing) |
| Stationary routing | `use_frozen_base_input_features=true` (query and gate read memory-free backbone features) + `frozen_prepass=true` (one memory-free forward per batch serves every routing input, both towers) | E38 (build, expert), E46 (extended to VLM), E59 (pre-pass) |
| Value-input noise | off | E58 (won on B), E62 add-7 (lost 4/4 seeds on merged 6×2; dropped) |

### 2.3 Router warm-up (geometry before content)

| setting | value | reference |
|---|---|---|
| Mode | `train_router_only=true` + `router_only_fast=true`: only keys and query projections train; values pinned at zero so the MSE gradient on the router is exactly zero | E36 (build), E37 (result), E49 add (fast path) |
| Steps / LR | 10k compressed schedule (lerobot auto-scales warmup/decay to 10k), router LR 1e-4, bs32 | E20 add-2 (schedule gotcha), E36 |
| Contrastive | sample SupCon, weight 0.05, `negatives_only=false`, `contrastive_query_queue=512` | E19 (queue), E20–E22 (negonly dead), E24 (c is the compaction knob), E26, E54 P2 (c-up null; axis closed) |
| Inter-task separation | weight 8.0 on the joint slot distribution, `routing_query_queue=512` | E7 (joint-slot fix), E23 (cross-batch queue), E25–E26 (sep 5 clears), E54 (sep 8) |
| Locality / global balance | 0 / 0 | E26 (locality dead), E13 (global balance SmolVLA-only) |
| Loss accounting | broadcast (deployment-mass-weighted): `vlm_route_once=false` at warm-up | E47 (principle), E48 (validated) |
| Pre-pass on | `PREPASS=true` | E59 add-1 |
| Downstream override rule | every freeze-mode flag set at warm-up must be explicitly disabled downstream (`train_router_only=false`, `router_only_fast=false`, `vlm_route_once=true`) | E37 add, E49 add |

### 2.4 Audit and gate

| setting | value | reference |
|---|---|---|
| Instrument | held-out routing audit: stream libero_10 demos through the frozen warm-up checkpoint (memory value LR 1e-12), dump per-task slot usage, compute famIoU / bgIoU / core50 / effnum; ~35 min; audit forced to the compact route-once path, `AUDIT_BS=8×400` for 12-module layouts | E19–E20 (build), E48 (bs override), E60 |
| Gate (bg-first) | expert bgIoU ≤ 0.10, mean core50 ≥ 400, min-task effnum ≥ 300; VLM min-task effnum ≥ 150; famIoU informational (backstop ≥ 0.45 only) | E21 (capacity gate), E54 (core50 relaxed to 400), E56 (bg-first), E59 add-2/3 (standing rule), E60 (first production use) |
| Paper cell certificate | gate PASS (no numbers transcribed in E62; the bigsearch cert on the same sites is E60 add) | E62 |

### 2.5 A-phase (values on the pretraining suite)

| setting | value | reference |
|---|---|---|
| Mode | `train_memory_only=true` + `freeze_memory_router=true`: values, gate, value_proj, swilu train; keys/query/backbone frozen; frozen-base routing on | E33 (flag), E37 (router frozen decision), E38 |
| Data / steps | libero_90, 10k steps; aux losses left on as telemetry | E37 |
| Batch ladder | bs32 → bs16×acc2 → bs16×acc2+ckpt (merged 6×2 ran after "two expected ladder demotions") | E54 U4, E59 add-1, E62 |

### 2.6 Sequential adaptation (the "C-config plus deltas")

| setting | value | reference |
|---|---|---|
| Trainable | values only (2.42–2.68B / 6.6B); router, keys, gate, backbone bitwise frozen (checkpoint diffs prove it) | E39, E53 |
| Steps per task | 5,000 | E29–E30 (3k → 5k), E41 (7k dead) |
| Value LR | `memory_value_lr` 2e-3 → 2e-4, linear per block ("lr2x") | E40–E42 (lr2x on staged), E50–E51 (1×/2×/4× inverted U; closed at 2×) |
| Write budget | `tfidf_top_t=3072` per site; online IDF (denom 1, exponent 1, raw weighting) | E42 (3072 = coverage gain), E51 P2/P9 (top-p 19.2, mass top-p 37.6; 3072 interior optimum) |
| Shared-table masks | per-site top-t with UNION merge on the shared table | E61 add |
| Protection | `protect_prior_slots=true`, `protect_beta=4`, `protect_mode=rank`, `protect_u_norm=corefrac`, `protect_hard_u=0`; store folded after each task; max-sync across shared-group members | E27–E28 (β4), E30 (β8 over-protects), E41 (corefrac built), E52–E53 (corefrac default; peak-norm retired), E61 add (max-sync) |
| Optimizer | AdamW, reinit each task, grad clip 1.0 | E18, E58 add-4 |
| Batch | bs16×acc2 (effective 32) | E62 add-1 (updt_s 0.587–0.598) |
| Seed / order | seed 1000; task_index 0..9 in order (5-task = 0..4) | SR |
| Checkpoints | per-task `pretrained_model` + `sequential_state.pt` (protection store + IDF accumulators); `--resume_sequential` | E54 U5 |
| In-run eval | 20 episodes at boundaries (retired from decisions), 50 episodes final (`eval_final_episodes=50`) | E40, E41 |
| Headline instrument | 25 episodes × 4 paired seeds (1000/2000/3000/4000), vec batch 13, `eval_seeds_campaign.py`; retention triangle = the same at every boundary | E60 add-3/4, E64 add-3 |

### 2.7 Real-world (WidowX AI) deltas from the sim cell

Same layout, sharing, bank, rank, knn, gate, protection and LR (E65). Differences: stage-1 = raw pi05 on `realworld_pretrain_v5` (15 tasks, 753 episodes), 50k steps, bs8×acc4 no-ckpt (E65); warm-up losses contrastive 1.0, separation 1.0, expert anchor 0.30, VLM pool (1.0, 1.0) (E65 add-10; arms 1–3 at c0.05/sep8, c0.10/sep4, c0.30/sep2 all hard-failed capacity, add-4/7/9); VLM min-effnum tripwire overridden after review (add-12, vindicated add-22); `tfidf_top_t=1536` (add-19/20; 3072 saturated the mask on low-diversity batches, add-14/15). Five sequential tasks: pool ids 0, 10, 16, 7, 1 = mustard-basket, push white lego, stack yellow bricks, screwdriver-tub, red bow-plate (split v5, E65).

---

## 3. Simulation paper cell results

### 3.1 Five-task (front-5: envs 4, 6, 9, 2, 7)

| instrument | e4 | e6 | e9 | e2 | e7 | mean | reference |
|---|---|---|---|---|---|---|---|
| 50 ep, seed 1000 (in-run final) | 58 | 74 | 64 | 84 | 54 | **66.8** | E62 add-1 |
| 25 ep × 4 seeds (headline) | 56±5.7 | 73±6.0 | 70±10.6 | 84±4.6 | 43±8.9 | **65.2** | E62 add-3 |

Paired deltas vs bigsearch-12 (4.8B): +1.6/+0.8/+1.6/−1.6 (statistical tie at 58% of its parameters). +4.6 over interleave-8, +6.2 over the r32 specialist oracle (E62 add-3). Fresh 20-ep opener e4 = 80, best ever recorded (E62 add-1). Training cost updt_s 0.587–0.598 at bs16×acc2, the fastest production config (E62 add-1).

### 3.2 Ten-task (all libero_10, train order e4, e6, e9, e2, e7, e0, e8, e1, e3, e5)

| env | task | 50 ep seed 1000 (E63) | 4-seed mean ± sd (E63 add-2) | r32 specialist | delta (4-seed) |
|---|---|---|---|---|---|
| e4 | two mugs | 52 | 59.0 ± 14.4 | 46.0 | +13.0 |
| e6 | mug+pudding | 76 | 60.0 ± 8.0 | 49.0 | +11.0 |
| e9 | mug+microwave | 64 | 63.0 ± 6.8 | 61.0 | +2.0 |
| e2 | stove+moka | 90 | 87.0 ± 5.0 | 80.0 | +7.0 |
| e7 | soup+cheese | 54 | 54.0 ± 4.0 | 59.0 | −5.0 |
| e0 | soup+sauce | 48 | 37.0 ± 10.0 | 46.0 | −9.0 |
| e8 | both mokas | 74 | 76.0 ± 5.7 | 67.0 | +9.0 |
| e1 | cheese+butter | 42 | 38.0 ± 9.5 | 62.0 | −24.0 |
| e3 | bowl+drawer | 86 | 86.0 ± 5.2 | 84.0 | +2.0 |
| e5 | book+caddy | 92 | 91.0 ± 2.0 | 83.0 | +8.0 |
| | **mean** | **67.8** | **65.1** | 63.7 | +1.4 |

Per-seed means 60.4 / 66.8 / 66.8 / 66.4 (E63 add-2). Front-5 64.6 vs back-5 65.6 at seeds (tasks that sat under five subsequent blocks are not degraded). The basket family is the entire deficit against specialists: e7/e0/e1 sum to −38, the other seven to +52 (E63 add-2). Against the re-provisioned r512 specialists (74.9) the cell is −9.8; log-linear interpolation places it at a specialist rank of roughly 45 (E64 add-8).

### 3.3 Rollout retention triangle, ten tasks, 4-seed (E64 add-12)

After block k, the k envs seen so far; each cell 25 episodes × 4 seeds.

| after block | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 54 | | | | | | | | | | 54.0 |
| b2 | 54 | 65 | | | | | | | | | 59.5 |
| b3 | 50 | 66 | 71 | | | | | | | | 62.3 |
| b4 | 51 | 54 | 65 | 87 | | | | | | | 64.2 |
| b5 | 53 | 72 | 65 | 89 | 51 | | | | | | 66.0 |
| b6 | 54 | 65 | 64 | 83 | 40 | 40 | | | | | 57.7 |
| b7 | 51 | 64 | 73 | 89 | 45 | 48 | 71 | | | | 63.0 |
| b8 | 47 | 67 | 60 | 88 | 46 | 46 | 76 | 41 | | | 58.9 |
| b9 | 53 | 70 | 66 | 87 | 48 | 44 | 77 | 43 | 83 | | 63.4 |
| b10 | 59 | 60 | 63 | 87 | 54 | 37 | 76 | 38 | 86 | 91 | 65.1 |

Diagonal mean 65.4; prior-task mean per row 54–70 throughout. Compare the naive triangle in §4.4 (prior-task mean 0 from block 2 on).

### 3.4 Function-space forgetting (paired-noise MSE matrix, corrected loader)

The matrix instrument had a loader bug for shared-table checkpoints until 30 August (two of seven tables were frozen at the first checkpoint's values; E65 add-16). Corrected numbers, just-trained → final:

| run | per-task drift | mean | reference (fixed) | first published (wrong) |
|---|---|---|---|---|
| merged 6×2, 5 tasks | +6.9 / +5.4 / +4.0 / +3.2 / +0.0 % | **+3.9%** | E65 add-24 | +1.2% (E62 add-2) |
| merged 6×2, 10 tasks | +22.7 / +20.2 / +24.9 / +43.3 / +117.3 / +40.5 / +9.2 / +4.5 / +2.3 / +0.0 % | **+28.5%** | E65 add-25 | +6.5% (E63 add-3/4) |
| interleave-8, 5 tasks (dedicated tables, unaffected) | +3.6 / +3.0 / +2.2 / +1.3 / 0.0 % | +2.0% | E59 add-3 | — |
| naive seq-LoRA r256, 5 tasks | +1567 / +1452 / +1383 / +740 / 0 % | — | E58 add-6 | — |

The corrected sentence for the paper: drift grows with the number of subsequent tasks but stays roughly 30× below the naive baseline and does not convert to success loss at this scale (front-5 64.6 vs back-5 65.6); "zero forgetting" is not the right phrase at ten tasks (E65 add-25). The basket hub e7 at +117% is by far the worst cell. Merged 6×2 is slightly leakier than dedicated-table interleave (E65 add-24).

### 3.5 Battery on the paper cell

- Jitter/OOD grid (chunk error, final checkpoint): 5-task e4 clean 0.0149, e7 clean 0.0238 (below the e7 specialist's 0.0330); best OOD grid of any substrate (E62 add-2). 10-task: e4 0.0170, e2 0.0298, e1 0.0223, e7 0.0628 (e7 the brittlest cell, E63 add-3).
- Harvest-bank rescore (e7 off-trail function vs the specialist): spec/succ far-quartile D 0.332 (interleave 0.344, B 0.482), demo anchor 0.0155 (E62 add-2).
- Prior-core write events: solo E14/E16 = 0 at every victim at both 5 and 10 tasks; VLM banks all zero; the one leak is shallow E4 (~25k events into e9's core, no growth with exposure, no rollout cost) (E62 add-2, E63 add-3).
- Site-bleed on the five shared pairs 14–51% at both 5 and 10 tasks (co-writing is per-block, not cumulative) (E62 add-2, E63 add-3).
- Noise arm on this substrate: 69.2 at seed 1000 (E62 add-6), 60.6 at 4 seeds, negative at 4/4 paired seeds; e7 tied (45±5 vs 43±9); dropped (E62 add-7).

---

## 4. All baseline results

Unless stated, every row is 25 episodes × 4 paired seeds (1000/2000/3000/4000) on final checkpoints. Every LoRA row and every memory row trains from the libero_90 stage-1 finetune; the only raw-pi05 row is "FT-fresh" (E64). Data budgets: full-FT rows use all ten tasks' demos jointly; specialists are per-task models with task identity at test; multitask-LoRA is one adapter on all tasks jointly; memory and naive rows are sequential (E60 add-7 decision, E61 add-7 caption).

### 4.1 Five-task table (front-5)

| row | e4 | e6 | e9 | e2 | e7 | mean | reference |
|---|---|---|---|---|---|---|---|
| Full FT from libero_90 substrate (all-10 data) | 78 | 78 | 79 | 90 | 66 | **78.2** | E60 add-9 |
| Full FT from raw pi05 (all-10 data) | 32 | 81 | 73 | 98 | 54 | 67.6 | E60 add-8 |
| **Merged 6×2 (ours, 2.8B, sequential)** | 56 | 73 | 70 | 84 | 43 | **65.2** | E62 add-3 |
| Bigsearch-12 (ours, 4.8B) | 53 | 70 | 69 | 81 | 50 | 64.6 | E60 add-4 |
| Interleave-8 (ours, 3.2B) | 50 | 70 | 58 | 86 | 39 | 60.6 | E60 add-4 |
| Merged 6×2 + value-input noise 0.5× | 51 | 67 | 57 | 83 | 45 | 60.6 | E62 add-7 |
| Specialists r32 (5 models, task ID) | 46 | 49 | 61 | 80 | 59 | 59.0 | E60 add-4 |
| Shared-pairs (ours, 1.6B) | 48 | 67 | 62 | 83 | 32±0 | 58.4 | E61 add-7 |
| Multitask-LoRA r32, 1k steps/task | 41 | 47 | 39 | 78 | 52 | 51.4 | E61 add-4 (row), E64 (under-provisioned) |
| Naive sequential LoRA r256 | 0 | 0 | 0 | 55 | 35 | 18.0 | E61 add-4 |

Front-5 means of the specialist rank ladder: r64 66.4, r128 68.8, r512 70.8 (E64 add-16/14/7). Zero-shot stage-1 floor 10.6 single-seed (E34).

Single-seed 50-episode anchors that predate the 4-seed instrument (keep as history only): specialists 63.2 (E56), multitask-LoRA 49.2 = the old must-line (E44), naive r256 17.6 (E58 add-6), joint FT on libero_90+libero_10 72.6 (E31; a different cell from both full-FT rows above, E60 add-6), B 53.2 (E55), absmax 53.6 (E54).

### 4.2 Ten-task table

| row | all-10 | front-5 | back-5 | reference |
|---|---|---|---|---|
| Full FT from libero_90 substrate | **79.7** | 78.2 | 81.2 | E62 add-10 |
| Multitask-LoRA r512/a128, 50k steps | 77.1 | — | — | E64 add-5 |
| Specialists r512/a128 | 74.9 | 70.8 | 79.0 | E64 add-7 |
| Specialists r128/a32 | 69.5 | 68.8 | 70.2 | E64 add-14 |
| Specialists r64/a16 | 68.8 | 66.4 | 71.2 | E64 add-16 |
| Full FT from raw pi05 | 67.8 | 67.6 | 68.0 | E62 add-10 |
| **Merged 6×2 (ours)** | **65.1** | 64.6 | 65.6 | E63 add-2 |
| Specialists r32/a8 | 63.7 | 59.0 | 68.4 | E62 add-9 |
| Multitask-LoRA r32, 10k steps (1k/task) | 53.2 | — | — | E62 add-10 |
| Naive sequential LoRA r512/a128 | 9.7 | 0.0 | 19.4 | E64 add-10 |
| Naive sequential LoRA r1216/a304 (parameter-matched, 2.681B, + vision tower) | 8.6 ± 0.5 | 0.0 | — | E66 add-2 |
| RETAIN (full FT + 0.5/0.5 weight interpolation per task) | running | | | E67 |
| O-LoRA (r64 adapter per task, frozen-but-active, L1 orthogonality λ₁=0.5) | queued | | | E67 |

Per-env, train order e4/e6/e9/e2/e7/e0/e8/e1/e3/e5:

| row | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 |
|---|---|---|---|---|---|---|---|---|---|---|
| FT-l90 | 78 | 78 | 79 | 90 | 66 | 80 | 82 | 60 | 89 | 95 |
| FT-fresh | 32 | 81 | 73 | 98 | 54 | 57 | 69 | 39 | 80 | 95 |
| Multitask r512 | 67 | 77 | 93 | 98 | 62 | 71 | 81 | 36 | 90 | 96 |
| Spec r512 | 52 | 71 | 78 | 85 | 68 | 84 | 70 | 61 | 86 | 94 |
| Spec r128 | 72 | 70 | 64 | 76 | 62 | 62 | 67 | 46 | 87 | 89 |
| Spec r64 | 60 | 60 | 75 | 84 | 53 | 69 | 75 | 52 | 77 | 83 |
| Spec r32 | 46 | 49 | 61 | 80 | 59 | 46 | 67 | 62 | 84 | 83 |
| **Merged 6×2** | 59 | 60 | 63 | 87 | 54 | 37 | 76 | 38 | 86 | 91 |
| Multitask r32 | 35 | 47 | 49 | 62 | 46 | 47 | 46 | 52 | 81 | 67 |
| Naive r512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 |
| Naive r1216 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 86 |

(References as in the table above; FT back-5 per-env from E62 add-7; multitask r32 per-env E62 add-10; spec per-env E64 add-16 ladder table.)

Notes on the LoRA rows (E64, E64 add-1): rank 512 was chosen before any result as the smallest power of two above the per-site per-token bottleneck (2 × heads × knn = 288 expert / 128 VLM); alpha/r = 0.25 held everywhere; every specialist and naive row is 5,000 steps per task, bs16×acc2, LR 1e-4 → 1e-5. Specialist gain is about +2.8 per rank doubling (E64 add-8). The 5-task multitask row (r32, 1k steps/task) is under-provisioned and was not rerun (E64: footnote or twin later). The parameter-matched control is over-complete on 362 of 416 target matrices and costs 3.8× the wall-clock per sample (E66, E66 add-1).

### 4.3 Real-world table (WidowX AI, five tasks; function-space only, no rollouts exist)

Diagonal drift of paired flow-matching MSE, just-trained → final (E65 add-23):

| row | t0 mustard | t1 lego | t2 bricks | t3 screwdriver | t4 bow | mean |
|---|---|---|---|---|---|---|
| **Ours, top_t 1536 (paper cell)** | +5.9 | +4.7 | +1.4 | +1.6 | +0.0 | **+2.7%** |
| Ours, top_t 3072 | +35.5 | +15.8 | +4.0 | +6.6 | +0.0 | +12.4% |
| Naive sequential LoRA r64/a16 | +3950 | +2963 | +3348 | +1521 | +0.0 | +2356% |

Own-task MSE relative to the r64 specialists (one adapter per task, task ID given): 0.88 / 0.97 / 0.88 / 0.79 / 0.82, i.e. the memory model fits every task better than that task's dedicated specialist (E65 add-22). This is a fit statement, not a success statement (E56 precedent: matched function still lost rollouts on 3 of 5 sim tasks). Task 1's small VLM footprint is a data property, not a capacity limit (its specialist is the second-worst cell) (E65 add-22).

### 4.4 Naive sequential LoRA r512 retention triangle, ten tasks, 4-seed (E64 add-11)

| after block | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | row mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 70 | | | | | | | | | | 70.0 |
| b2 | 0 | 64 | | | | | | | | | 32.0 |
| b3 | 0 | 1 | 82 | | | | | | | | 27.7 |
| b4 | 0 | 0 | 0 | 96 | | | | | | | 24.0 |
| b5 | 0 | 0 | 0 | 32 | 58 | | | | | | 18.0 |
| b6 | 0 | 0 | 0 | 0 | 0 | 73 | | | | | 12.2 |
| b7 | 0 | 0 | 0 | 2 | 0 | 7 | 67 | | | | 10.9 |
| b8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 | | | 4.8 |
| b9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 86 | | 9.6 |
| b10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 | 9.7 |

Diagonal mean 73.1 (the naive adapter fits each task at specialist grade, then loses it after one block).

### 4.5 LoRA specialist function anchors (chunk error on demo states, single model per task)

r32 specialists: e4 58 / 0.0204, e6 44 / 0.020, e9 70 / 0.0675, e7 60 / 0.0330, e2 84 / 0.0308 (E43, E55 add, E56). The e7 arbiter: the specialist converts at the same on-demo function the memory model has (0.0330 vs 0.0321) and still rolls 60 vs 20, so e7's wall is off-trail conversion, not fit (E55 add). Compass cells on e4 (E44, E51 P5): expert-only LoRA 14 / 0.229, VLM-only 40 / 0.030, attention-only 26 / 0.106.

---

## 5. Trajectory per hyperparameter: what was tried, what happened, where

Each subsection lists the settings in the order they were tried, the outcome, and the entry. "Seq" = final sequential success; SmolVLA numbers (E0–E17) are 4 tasks and not comparable to pi05 numbers (10 or 5 tasks).

### 5.1 Model, regime, and protocol

| stage | setting | outcome | reference |
|---|---|---|---|
| SmolVLA-500M, pretrain 35 tasks, sequential libero_spatial 6/7/8/9 | | best 57.5% (knn36, top_t 1536) | E0–E17 |
| pi05, pretrain libero_minus_goal (30 tasks, 30k), sequential libero_goal ×10 | joint pretrain | 45.8% (top_t 512), 43.0% (top_t 1536) | E18, E19 |
| pi05, pretrain libero_90 (40k), sequential libero_10 ×10 | joint pretrain | control 34.4%; diagonal caps ~40% | E19 |
| Staged: stage-1 base fine-tune, then memory on frozen base with the joint losses | routing losses lose their actuator; killed at 6/10 | E33–E35 |
| Router warm-up (values pinned at zero) on frozen base | famIoU 0.145, best certificate; A-phase then sequential | E36–E37 |
| Warm-up + A-phase + sequential without stationary routing | routing drift kills it (self-IoU 0.2–0.3 above L8) | E38 |
| Same with frozen-base routing | first flat forgetting matrix; the staged protocol is the thesis from here | E38–E39, E40 |
| Joint track ("r2244 + snapshot routing") | parked as off-thesis | E40 |

### 5.2 Memory placement (layers) and sharing

| setting | outcome | reference |
|---|---|---|
| SmolVLA expert [14,15] | initial | E1 |
| SmolVLA [12,14] | seq 34.5 with sep 0.25 | E2–E8 |
| SmolVLA [10,12,14] | seq 44.0 (+10, capacity) | E9 |
| SmolVLA [8,10,12,14] | seq ~55 (E11 read; the clean 4-layer control in E13 read 46.0) | E10–E13 |
| pi05 expert [8,10,12,14] | the joint-era and early staged substrate | E18–E49 |
| Drop L14, rank 4 on [8,10,12] | rejected: the last-layer hub role migrates to L12 and amplifies | E31–E32 |
| Move expert memory up (L15–17) | dead: separability flat, crowding worse | E36 |
| VLM text-span memory at LM [15,16] added | largest fit mover of the staged era; 40.0 on plain config | E44–E49 |
| Compact layer-max: expert [9–12] + VLM [13–16] | 44.8 plain | E50–E51 P7 |
| Spread: expert [2,4,6,8] + VLM [10,12,14,16] | best fits, full drift tax with peak-norm (41.2); 47.6 with corefrac; 53.2 with the anchored router (B) | E51 P8, E53, E55 |
| Attempt-A audit (spread) | L2 famIoU 0.212; expert should not go below ~L8 | E51 P7 |
| Absmax: expert [4–9] + VLM [10–16], 13 modules, 5.37B | 53.6, zero give-back; failed the capacity gate at every expert layer | E53–E54 |
| Interleave: expert [6,8,10,12] + VLM [7,9,11,13] (needs `frozen_prepass`) | 57.6 / 60.6 at seeds; e7 38 | E59 |
| Bigsearch: expert [4,6,8,10,12,14,16] + VLM [5,7,9,11,13], 4.8B | 59.6 / 64.6 at seeds; e7 58 | E60 |
| Shared pairs on the interleave sites (1.6B) | 56.8 / 58.4; e7 22 → 32±0 (shared deep table kills the depth lever) | E61, E61 add-3/7 |
| Share-criterion probe (router-input similarity) | inverted by calibration; adopted as a veto within the expert tower plus the depth rule | E61 add-6 |
| Merged 6×2 (paper cell) | 66.8 / 65.2 | E62 |
| Attention-side memory | killed by compass (attn-only LoRA 0.106, plateaued) | E51 P5 |
| Image-span VLM routing at [15,16] | 28.8; brittle to image noise; parked | E49 add, E50 |
| VLM depth geometry | instruction-anchor separation improves downward to L7, V-curve minimum at L7; V3/V4 excluded, V5 viable | E49 add, E59 add-6, E60 add |
| Lesion map on the 12-site layout | designed, never run (demoted to paper evidence) | E60, E61 add-5 |

### 5.3 Bank size (`mem_n_keys`)

| n | slots | outcome | reference |
|---|---|---|---|
| 384 | 147,456 | SmolVLA and joint pi05 era | E1–E43 |
| 544 | ~296k | rank-1 double-slot test, no follow-up | E4 |
| 256 | 65,536 | right-sized to measured usage; famIoU 0.145 both; standard since | E44, E46 |
| 128 | 16,384 | famIoU +47%, per-query-draw floor; dead | E46–E47 |
| 192 | 36,864 | pigeonhole knee; dead. Rule: per-half subkey demand under ~55–60% of n | E47–E48 |

### 5.4 Retrieval breadth (`mem_knn`) and loss alignment

| setting | outcome | reference |
|---|---|---|
| 16 (routing_loss_topk 16) | SmolVLA baseline 45.5 | E1–E14 |
| 8 / 12 | worse (42.5 / 43.5); concentration, not cleaner separation | E14 |
| 24 (topk 16) | 49.5 | E15 |
| 24 / 36 / 48 aligned (topk = knn) | 51.5 / 46.0 / 46.5; alignment is a real gain; higher knn write-limited at top_t 512 | E16 |
| 36 with top_t 1536 | 57.5, best SmolVLA | E17 |
| VLM knn 16 vs 36 | 36's apparent advantage was a dedup artifact; 16 wins at rollout; closed | E46–E49 |
| Rule | deployed knn must equal the topk the keys were trained with | E16, E46 |

### 5.5 Value type and rank

| setting | outcome | reference |
|---|---|---|
| Static value vectors | near-zero interference, ~30% plateau | E0 |
| LoRA slots rank 2–4 | capacity up, interference back | E0, E4 |
| Rank 1 with 2× slots | test only | E4 |
| Rank 8 [12,14], rank 4 [10,12,14] | OOM then accumulation; rank 4 confounded by a smaller contrastive pool | E10–E11 |
| Per-layer rank [2,2,2,4] / [4,4,4] | P1 passes, P2 rejected | E31–E32 |
| [2,2,4,4] | 46.5 (+2); forgetting per unit exposure rank-invariant; closed | E32–E33 |
| VLM rank 4 (graft) | 30.4 vs 40.0; closed | E49–E50 |
| Per-slot affine bias (`lora_slot_bias`) | learned but null; retired | E40–E41 |
| Uniform n128/r4 and n192/r4 | dead via bank size | E46–E48 |

### 5.6 Write budget (`tfidf_top_t`) and adaptive coverage

| setting | outcome | reference |
|---|---|---|
| 512 | SmolVLA default | E8–E16 |
| 768 / 1024 / 1536 | knn24 family flat or worse; knn36 family monotone to 57.5 at 1536 | E17 |
| 2048 / 3072 planned on SmolVLA | superseded by the regime change | E17–E18 |
| pi05 512 vs 1536 (libero_goal) | 45.8 vs 43.0; raising budget at high read overlap is net-negative | E18–E19 |
| 512 with the compact SupCon prior | near no-op because reads collapsed; coverage rule adopted (set top_t from per-batch effnum, ~70–90%) | E21–E22(d) |
| 1536 (C-config) | standard from E26 to E49 | E26, E30 |
| 3072 | +5.6 (37.6 vs 32.0) via coverage, not concentration; standard from E50 | E42, E50 |
| Top-p count rule 0.9 | 19.2, worst staged result on the best losses; the fixed budget was load-bearing protection | E50, E51 P2–P3 |
| Mass top-p clipped [3072, 5120] | 37.6 (effectively static 5120); dose-response 19.2 < 37.6 < 46.0; axis closed | E51 P6, P9 |
| Real-world 3072 → 1536 | mask saturation on low-diversity batches bypassed corefrac; 1536 restores it (drift 10.9% → 2.7%, fit tax +4.5%) | E65 add-14/15/19/20 |

### 5.7 Value learning rate and steps per task

| setting | outcome | reference |
|---|---|---|
| 1e-3 → 1e-4 per block | joint-era default | E18 |
| 3,000 → 5,000 steps | +4 (44.5), retention improved; standard since | E29–E30 |
| 2e-3 joint era | best fresh diagonal, worst retention (−9.5); later read as the drift tax | E30, E38 add |
| 7,000 steps (staged) | blocks converge by ~3k; dead | E40–E41 |
| 2e-3 staged (lr2x) | only init mover; give-back at 1× amplitude was mostly draws | E40–E42 |
| 4e-3 → 2e-4 + 3072 (expert-only substrate) | 42.4; composes in function space; conversion slope ~+2pp per −10% loss | E43 |
| 1× / 2× / 4× on the VLM substrate + 3072 | 40.0 / 46.0 / 40.4, inverted U; 4× + budget-v3 43.6; closed at 2× | E50–E51 P1/P8 |
| Router LR at warm-up | 2.5e-5 (base group) → 1e-4 | E36 |

### 5.8 Batch size

bs32 throughout; bs64 via accumulation exonerated and retired (parity at 2× cost, E41–E42); bs16×acc2 / bs8×acc4 are VRAM ladder rungs, numerically neutral (E53 update 24 Jul, E60 add-7).

### 5.9 Protection (write-side)

| mechanism | outcome | reference |
|---|---|---|
| TF-IDF top-t only | controls writes, not reads | E1, E8 |
| Weighted TF, saturating TF, IDF exponent 2, mass-weighted DF, pretrain IDF seeding | measured no-ops or harmful | E4–E5, E9, E19 |
| Hard veto of prior-read slots | zero-sum: starves the writer | E19 |
| Prior-usefulness rank mode, β=4 (peak-norm) | +6.5 (40.5); cleans incidental channels; e7 stays 0 | E27–E28 |
| β=8 | over-protects (38.5) | E29–E30 |
| Grad-scale (soft) mode | Adam invariance (caught pre-launch), then momentum-tail leak (~90% passthrough); fixed version works but taxes the last writer (+49% block-min) | E41–E42, E44 |
| Freeze top-5k A-phase slots (`protect_seed_path`, hard_u 0.9) | mechanically perfect, strategically null (39.2 vs 42.4) | E42 add, E43–E44 |
| Budget mode v2 | NaN 200 steps after the first boundary (momentum-boost compounding) | E51 P4 |
| Budget mode v3 (conserved proportional) | core overwrite eliminated (0 events), e9 +22, but 43.6 < 46.0 (writer tax) | E51 P8 |
| `protect_u_norm=corefrac` in rank mode | 51.6, zero writer tax, flat matrix; default since | E41 (build), E52–E53 |
| Decaying protection store | rejected as insufficiently general | E50 |
| Under sharing: max-sync vs noisy-OR | max kept; debate closed (differ 1.6–3.8%) | E61 add, E61 add-3 |

### 5.10 Routing losses at pretrain / warm-up

**Contrastive.** Centroid vs sample (E2, no winner); weights 0.5/1/2 non-monotonic (E2); 0.01 (E18–E19); 0.05 with `negatives_only` and queue 512: best certificate but capacity collapse (E20–E21); 0.025 negonly still dead (E22); standard 0.05: capacity safe, no separation (E22); standard 0.1: compaction, the c-axis is a breadth knob (E24); c=0: sprawl, contrastive is load-bearing (E25); 0.15 at anchor 0.35 and 0.5: null (E54 P2, C; axis closed); VLM sweep 0.0125/0.05/0.5: breadth monotone in c (E45). Real-world 1.0 (E65 add-10). Final: 0.05 standard, queue 512.

**Inter-task separation.** Half-marginal loss 0.15/0.25/0.35: no gain, wrong distribution (E6–E7); joint-slot 0.25: 16.5 → 34.5 (E8); 0.5 (E22–E24, no transfer); cross-batch queue rq512 makes the estimator honest (E23–E24); 2.0: first decoupled win (E25); 3.0/5.0: monotone, 5 clears the gate via translation (E26); 5 in the full run: interference halved, score flat (E27); 8 at w0.35 fails, 8 at w0 passes, 8 alone buys nothing at rollout (E54 P1/P3, E56). Final: 8.0. Real-world 1.0.

**Locality / compactness.** Introduced E5–E7; 0.25 carried through E19; 1.0 fails (E20); 0 ≡ 0.25 (E25–E26); dead.

**Global balance.** 0.05/0.1/0.2 on SmolVLA: only 0.05 helped and broadened footprints (E12–E13); never used on pi05.

**Dropout and output corruption.** Dropout 0.05–0.2 (E2), 0.1 (E14–E15): wrong mechanism. Corruption 0.05–0.2 (E8–E9), 0.05 (E14–E15), planned 0.05–0.15 on knn36 (E17): degrades pretrain quality; dead. Distinct from value-input noise (E58).

### 5.11 Router conditioning

| setting | outcome | reference |
|---|---|---|
| FiLM on mpnet language embedding (`lang_to_query`, `fuse_method=film`) | the SmolVLA and joint-era router | E1 |
| FiLM found near-inert at inference (γ≈0, scene term 17–21× larger) | basket collision is scene-genuine | E28 |
| Language-only routing | doubles the family collision; parked | E36 |
| Nonlinear query head (`query_proj_layers=2`) | null on frozen features | E34 |
| Expert text anchor w=0.5, FiLM off | cleanest famIoU/bg ever, but core50 ~400 (over-compaction) | E53, E53 PM |
| Anchor sweep 0.10/0.25/0.35/0.40/0.50 (spread) | famIoU U-curve; viable window empty under the core50 ≥ 800 gate | E53 arm-3, E54 U1 |
| w0.35 + sep8 / w0.35 + c0.15 | both fail | E54 U1/U2 |
| w0 + sep8 (P3) | passes; 47.2 at rollout | E54 U2, E56 |
| **w0.40 + sep8, FiLM off (B)** | passes in the absmax band; 53.2; the router | E54 U3–U4, E55 |
| w0.5 + c0.15 (C) | fails core floor by one layer | E54 U3 |
| No language at all (P4) | catastrophic sprawl: language conditioning is load-bearing at warm-up | E54 U3 |
| Real-world anchor 0.30 | with pool (1,1) | E65 add-10 |
| VLM per-token routing on the text span | fails at every dose (digit vocabulary) | E44–E45 |
| VLM pooled key: (1,0) / (1,0.5) / state-only | (1,0.5) wins on fit; anchored variants hit famIoU 0.145 | E45–E46 |
| Route-once loss dedup | collapses the palette to ~2 draws; broadcast losses restored | E47–E48 |
| VLM image-region keys (g2, a=1, b=0.5) | certifies, rolls 28.8; parked | E49 add, E50 |

### 5.12 Stationarity and gate

Routing drift discovered (self-IoU 1.0 at L8, 0.2–0.3 above) and frozen-base dual-path routing built (E38); extended to the VLM tower after a non-stationarity at L16 (E46); `frozen_prepass` lifts the placement guard at ~equal training cost (E59, E59 add-3). Gate: saturated at 0.98–0.99 in every joint pi05 run (E18–E19); settles at 0.63–0.75 on a frozen base (E34–E35); removal (`mem_gated=false`) tested and retired (E40–E41).

### 5.13 Sharing and value-input noise

Sharing: keys+values per adjacent pair, per-site heads (E61); all four pairs shared on the interleave sites: e7 crater (E61 add-3); veto rule from the similarity probe (E61 add-6); merged 6×2 with solo deep tables restores e7 (E62). Noise: dose 1× / 0.5× on B, both safe, 0.5× best (54.4, e7 +6 twice) (E58 add-3); 0.5× recalibrated on merged 6×2: 69.2 single-seed then 60.6 at seeds, dropped (E62 add-4/6/7); the 10-task noise run was cancelled (E63 add-1).

### 5.14 Evaluation instrument

4 episodes (E8) → 50 episodes (E18) → 20-ep intermediates + 50-ep finals (E28) → 20-ep cells retired at ±11pp (E41) → 50-ep finals ±7pp (E43) → 25 ep × 4 paired seeds after a 14-point single-seed swing on e2 (E60 add-3/4) → retention triangle at every boundary (E64 add-3). Chunk error adopted as the function-space gate (E41), valid within a substrate only (E50). Harvest-bank off-trail instrument (E57); D-vs-specialist valid only while below the specialist (E62 add-5, retracted E63 add-5: it agreed with the seeds).

### 5.15 Frontier ladder (single-seed 50-ep finals, front-5, in order)

stageB 32.0 (E39) → softprotect 41.2 (E42) → comp 46.0 (E50) → compact+corefrac 51.6 (E53) → absmax 53.6 (E54) → B 53.2 (E55) → dose05x 54.4 (E58 add-3) → interleave 57.6 (E59 add-3) → bigsearch 59.6 (E60 add-2) → merged 6×2 66.8 (E62 add-1). At the 4-seed instrument: interleave 60.6 → bigsearch 64.6 → merged 6×2 65.2 (E60 add-4, E62 add-3).

---

## 6. Chronological narrative

### 6.1 Phase A: SmolVLA, four sequential tasks (E0–E17, February to April 2026)

**Setting.** SmolVLA-500M, pretrain on 35 LIBERO tasks, then sequentially adapt on four `libero_spatial` tasks (6, 7, 8, 9). Metric: average success on seen tasks after all four blocks.

**Starting point (E0).** Static value vectors gave near-zero interference but plateaued around 30% per task. LoRA-valued slots raised capacity but brought interference back. Query-contrastive losses and output corruption gave little.

**The routing problem (E1–E7).** Every task read heavily from a shared hot core. TF-IDF stopped most writes into that core, but reads still overlapped 60–70% (E1). Sweeps over contrastive type/weight and dropout (E2), a cross-batch contrastive queue (E3), and locality and separation losses (E5–E6) followed. E6 found the logged "separation" metric was a mean pairwise similarity. E7 found the real defect: the routing losses were computed on the two product-key half-marginals, not the joint slot distribution, so they could be satisfied by shuffling tail mass. The loss was rewritten on the joint top-M×M candidate distribution and a NaN gradient bug fixed.

**The fix works (E8–E9).** Joint-slot separation at 0.25 took sequential success from 16.5% to 34.5%, resolved the late-layer collapse, and made write overlap near zero (E8). The remaining interference was read-time. A third memory layer lifted the result to 44% (E9). Corruption noise hurt.

**Capacity and breadth (E10–E17).** Gradient checkpointing and accumulation were added (E10); the rank-4 comparison was confounded by a smaller contrastive pool (E11). Four layers reached ~55% (E11). Global balance helped only at 0.05 (E12–E13). Lowering `knn` hurt, 24 helped (E14–E15), aligning the routing loss to the deployed `knn` gave 51.5% (E16), and `knn=36` with `top_t=1536` reached **57.5%**, the best SmolVLA result, with the oldest task still degrading (E17).

**What this phase established.** Read-time interference, not write overlap, binds once TF-IDF is in place. Routing separation must act on the joint slot distribution. Write budget must scale with read breadth. Capacity (layers) matters as much as interference.

### 6.2 Phase B: pi05 on LIBERO, the joint-pretrain era (E18–E33, June to early July 2026)

**Regime change (E18).** pi05, ten held-out tasks trained sequentially: first `libero_goal` with a 30-task pretrain (45.8%), then from E19 `libero_10` with a 90-task pretrain (34.4%). The memory gate saturated at 0.98–0.99, and the basket family routed to the same slot regions under a frozen router.

**The two-factor model (E18–E19).** Final success = f(basin depth, overwrite exposure). The diagonal (fresh fit) capped at ~40% because six of ten tasks need two pick-place cycles. E19 showed TF-IDF is functionally a term-frequency mask at this scale, that hard protection is zero-sum, and that the pretrain-diversity lever was spent.

**Instrument built: the held-out routing audit (E20).** Streaming the held-out demos through a frozen checkpoint yields famIoU, bgIoU, core50 and effnum in ~35 minutes; these certificates predicted full-pipeline behaviour repeatedly.

**Over-compaction and the breadth axis (E20–E26).** The SupCon negatives-only package produced the best certificate yet and destroyed plasticity (E21). E22–E24 bracketed the contrastive frontier, gave the separation loss a cross-batch queue, and diagnosed a metric artifact. Separation at 2.0 finally reduced overlap without shrinkage (E25), and the curve was monotone to 5.0, which cleared the gate (E26). Contrastive (compaction) and separation (translation) are distinct and both required; locality is dead.

**Separation alone does not convert (E27).** The sep-5 prior halved sequential read overlap and produced 34.0% vs 34.4%. Forgetting became narrow and catastrophic on the basket hub (env 7). This led to prior-usefulness write protection.

**Protection and the scene-routing finding (E28).** β=4 gave +6.5pp (40.5%); env 7 remained at zero. Probes showed the FiLM language pathway near-inert and the scene term dominating the query 17–21×; env 7's collision is scene-genuine.

**Plasticity levers and the non-memory ceiling (E29–E31).** 5k steps became the best at 44.5%; 2× LR fitted better but backfired on retention; β=8 over-protected (E30). A plain pi05 joint fine-tune on libero_90 plus libero_10 scored 72.6% (E31).

**Rank (E32–E33).** Per-layer rank; dropping L14 rejected (E32); `[2,2,4,4]` graduated to 46.5% (+2); forgetting per unit exposure rank-invariant; the bottleneck measured as ~14pp fit plus ~12pp retention, both rank-insensitive. Decision: the staged protocol (E33).

### 6.3 Phase C: the staged protocol, stationarity, and forgetting solved (E34–E43, July 2026)

**Stage 1 and the frozen-base probe (E34–E35).** Stage-1 zero-shot on libero_10 was 10.6%, so every sequential fit ever measured was memory adaptation. Memory on the frozen base gave natural gates (0.63–0.75) but the routing losses lost their actuator (backbone co-adaptation), the audit failed separation, and the chain was killed at 6/10 (E35).

**Resolved by measurement (E36).** Frozen features are separable (~98% linear probe) but crowded (inter-task cosine ~0.9 vs ~0.46 after joint training). The router failure was anchored optimisation (values at 40× the router LR). Josh reframed the contribution around the staged setting. Router warm-up (values pinned at zero) was built.

**Warm-up clears every gate (E37).** famIoU 0.145 in three hours on a frozen backbone, capacity above the best joint prior, the basket family separated for the first time.

**Routing drift and the stationarity fix (E38).** Value updates below a router move its queries; self-IoU 1.0 at L8 and 0.2–0.3 above. Frozen-base routing (query and gate read memory-free features via a lazy fork / dual pass) fixed it. The addendum notes this channel had been live in every sequential run of the project.

**Forgetting solved (E39).** First flat MSE forgetting matrix (+0.0 to +1.7%), self-IoU exactly 1.0000, only slot values change between blocks. The gap to the joint substrate was rollout-fit conversion (inits 35 vs 48).

**The conversion-gap programme (E40–E43).** Joint track parked. Four arms (affine slots without gate, 2× LR, 7k steps, top_t 3072) and five probes falsified two mechanisms and demoted a third; the denoised-chunk error became the function-space gate (E41). Soft protection was found defective twice due to Adam (E41–E42). Batch 64 retired; top_t 3072 a real coverage gain (E42). The LoRA specialist baseline: e4 58 at chunk 0.018 vs the memory's 0.153, so the deficit was fit (E43). Two heuristics: the breadth law and the read-write product (E43 add).

### 6.4 Phase D: VLM memory and the placement axis (E44–E50, mid to late July 2026)

**The compass (E44).** Expert-only adapters fail e4 (14 / 0.229); VLM-only nearly reproduce the full result (40 / 0.030). Placement is the currency. Span attribution found state-as-text tokens carry up to half the loss sensitivity. Build: text-span VLM memory on LM layers [15,16], last 200 prefix positions; a pad-contamination bug was fixed the same day.

**Pooled routing (E45–E46).** Per-token routing on the state span failed at every dose (shared digit vocabulary). Routing the state region on one anchored per-sample key gave famIoU ~0.15 and the largest e4 fit gain yet (0.153 → 0.099). Route-once shipped; frozen-base routing extended to the VLM tower.

**Protocol lessons (E47–E49).** Route-once loss dedup collapsed the palette; broadcast losses restored it (E47–E48). Bank-scaling law and the subkey pigeonhole rule (E48). The three graduated routers formed a palette-constancy axis; arm 1′ won at 40.0 on plain config; VLM knn closed at 16 (E49).

**Composition, failed substrates, jitter (E50).** Arm 1′ + 2× LR + top_t 3072 reached 46.0. VLM rank 4 (30.4) and image-span (28.8) both improved on-demo function and rolled worse; the jitter probe and Josh's amplification model attributed the image-span failure to slot transforms at image positions. Rules: chunk error ranks within a substrate only; rank axis closed on both towers; image stack parked.

### 6.5 Phase E: layers, protection modes, and the climb to 53.6 (E51–E56, 21–30 July 2026)

**E51 (nine parts).** LR inverted U (1× 40.0, 2× 46.0, 4× 40.4). Budget protection v2 NaN'd, v3 shipped with multi-step Adam smokes. Count top-p 0.9 gave 19.2 (the fixed write budget was load-bearing protection). Attention-side memory killed by compass. Compact layer-max reached 44.8 plain. Budget-v3 eliminated core overwrite but landed 43.6 (amplitude closed at 2×). Mass top-p 37.6 closed the coverage axis: extra budget reaches into contested banks, extra layers buy new separated addressing.

**Corefrac and the frontier (E52–E53).** The fold-in produced the best fits ever and the first rising diagonal of the stationary era (e4 +22.6%); the single-delta `corefrac` patch gave **51.6%**, past the multitask-LoRA must-line, with zero core events and no writer tax.

**Anchors, gates, absmax (E53–E54).** The expert text anchor at w0.5 over-compacted (core50 ~400); the sweep found no window under the core50 ≥ 800 floor (E53 arm-3). Then the backfilled absmax run (that very router class) landed **53.6% with zero give-back**, falsifying the per-layer floor; bgIoU 0.026 vs 0.08 was the likely active ingredient (E54). AAAI dropped for ICRA. Probe pair on spread: sep 8 alone weak; anchor-weight U-curve; w0 + sep8 passes; w0.40 + sep8 passes in the absmax band; no-language routing sprawls (E54 U1–U3). B graduated; its first run died of a logind `RemoveIPC` sweep; linger, connection multiplexing and sequential resume were built (E54 U5).

**B and the oracle (E55–E56).** B reached **53.2%**, tied with absmax at 60% of its parameters, flat matrix, project-best chunks on 4/5 tasks (E55). P3 landed 47.2: the anchor's shoulder cleanup was the whole gain; bgIoU is the axis that pays (E56). The five specialists completed (63.2 single-seed); B matched every specialist's function and beat two; the residual was rollout conversion, above all on e7 (E56).

### 6.6 Phase F: off-trail measurement, placement depth, sharing, and the paper cell (E57–E63, 31 July to 18 August 2026)

**The off-trail instrument (E57).** Harvesting the policy's own rollouts and re-denoising on harvested states: retrieval stays on the written footprint everywhere, there are no routing discontinuities, and the deficit is a competence radius of the value content. e7 is decided at the second policy call. The jitter shell probed ~10× nearer than real excursions.

**Value-input noise (E58).** Calibrated from the harvest bank; free on fit and retention; moved the far-region function toward the specialist with an inverted U in dose; e7 +6 in both arms (54.4 at half dose). The naive sequential LoRA baseline was built: r256 collapsed to 17.6% with three tasks at 0% (E58 add-6). First real spot preemption; Nebius CLI installed; PEFT resume built (E58 add-5).

**Frozen pre-pass and interleaving (E59).** One memory-free forward per batch lifts the placement guard. Interleave (expert [6,8,10,12] + VLM [7,9,11,13]) failed its famIoU gate, was overridden on the bg-first rationale, and landed **57.6%** with e7 38; famIoU declared dead as a gate axis; the harvest rescore showed deep placement widened the competence radius by 29% (E59 add-2/3/5). A geometry probe found a V-curve with minimum at LM layer 7 (E59 add-6).

**Go-big and the multi-seed instrument (E60).** Bigsearch-12 landed **59.6%** with e7 58. A 14-point single-seed swing on e2 forced the 4-seed instrument: bigsearch **64.6** vs the specialist oracle **59.0**, positive at 4/4 seeds (E60 add-3/4). Full fine-tune baselines: 67.6 from raw pi05, **78.2** from the libero_90 substrate (E60 add-8/9).

**Shared tables (E61).** Halving value parameters kept every task but e7 (38 → 22, then 32±0 at seeds): depth-specialised content is real (E61 add-3/7). Site-bleed 17–43% and a first matrix breach (+7.7%) (E61 add-4). The share-criterion probe was inverted by calibration and adopted as a veto (E61 add-6).

**The paper cell (E62).** Merged 6×2: **66.8** at seed 1000, **65.2** at four seeds, tied with bigsearch at 58% of its parameters, every gate passed (E62 add-1/2/3). The noise arm read 69.2 then 60.6 at seeds and was dropped (E62 add-6/7). The weekend queue completed the 10-task baseline rows (E62 add-5/9/10).

**Ten tasks (E63).** **67.8** at seed 1000, **65.1** at four seeds vs the 10-task r32 oracle 63.7; front-5 64.6 vs back-5 65.6; the basket family is the entire deficit (E63, E63 add-2). The function-drift matrix read +6.5% (E63 add-3/4), later corrected to +28.5% (E65 add-25).

### 6.7 Phase G: baseline re-provisioning, the real robot, and the instrument correction (E64–E67, 18 August to 7 September 2026)

**Uniform LoRA baselines (E64).** The multitask-LoRA rows had been the E43 breadth probe promoted unrevised. The active-parameter ladder was written down (per token the method is a rank-288/128 adapter per site; per step ~0.23B; total 2.8B). Rank 512 was chosen a priori; all rows rerun from scratch: multitask-10 **77.1**, ten specialists **74.9**, naive 10-task **9.7** (E64 add-5/7/10). Ladder points r64 (68.8) and r128 (69.5) added (E64 add-14/16). Two 4-seed retention triangles measured (E64 add-11/12).

**Real world, WidowX AI (E65).** Twenty tasks inventoried; a task-geometry probe ranked held-out subsets; split v5 chose five sequential tasks. Stage-1, warm-up, and audit ran; the gate hard-failed capacity at three loss settings; the fourth arm passed every expert clause and failed only the VLM min-effnum tripwire on one task, shown to be a data property, so the gate was overridden (E65 add-4 to add-12). The sequential run showed +10.9% mean drift, root-caused to write-mask saturation (E65 add-14/15). At top_t 1536 drift fell to **+2.7%** (E65 add-19/20/21). Rank-64 specialists and a naive baseline completed the real-world table: ours +2.7%, naive **+2356%**; the memory model beat every specialist's own-task MSE (E65 add-22/23). No real-robot rollouts exist.

**The matrix loader bug (E65 add-16/17/24/25).** Comparing two instruments exposed that the MSE-matrix scripts missed the two shared storages saved under `_storage_shared_from`. After the fix, the 5-task merged 6×2 drift is +3.9% (published +1.2%) and the 10-task drift +28.5% (published +6.5%). Dedicated-table runs and the other instruments were unaffected.

**Parameter-matched control (E66).** A naive sequential LoRA with the same added parameters as the memory (2.68B) forces r=1216 over 416 matrices including the vision tower, over-complete on 362 of them; it cost 3.8× the wall-clock per sample and scored **8.6** at four seeds, nine of ten prior tasks at exactly 0 (E66 add-1/2).

**External continual baselines (E67).** RETAIN (full fine-tune per task, then 0.5/0.5 weight interpolation, their continual setting) and O-LoRA (rank-64 adapter per task, earlier adapters frozen but active, official L1 orthogonality penalty λ₁=0.5) were implemented under `scripts/baselines/` without touching the memory code, smoke-tested twice (draccus `from __future__ import annotations` trap; stderr capture; merge witness), and launched 15:21 UK on 7 September. RETAIN lands ~22:30 UK on 8 September, O-LoRA ~10 September; each gets the rollout triangle and the 10×10 loss matrix. Pre-registered bands: RETAIN 15–40, O-LoRA 20–45; either above ~55 needs explaining (E67, E67 add-1/2).

---

## 7. The final recipe (as of 7 September 2026)

Pipeline, in order (full parameter tables in §2):

1. **Stage 1.** Full fine-tune of raw pi05 on the pretraining suite (libero_90, or the real-world 15-task pool), no memory, 50k steps (E31, E33, E65).
2. **Router warm-up.** Attach memory; train only keys and query projections with values pinned at zero, 10k steps, routing losses only: sample contrastive 0.05 (queue 512) plus inter-task separation 8.0 with the cross-batch routing queue (512); expert router anchored to the pooled LM instruction hidden at weight 0.40, FiLM off; VLM routing on a pooled key (1.0 × instruction + 0.5 × state) for the state region, per-token for instruction tokens; broadcast loss accounting; frozen pre-pass (E36–E37, E45–E48, E54, E59). Real-world: contrastive 1.0, separation 1.0, anchor 0.30, pool (1,1) (E65 add-10).
3. **Audit and gate.** Held-out routing audit; gate on bgIoU ≤ 0.10, mean core50 ≥ 400, min effnum ≥ 300, VLM min effnum ≥ 150; famIoU informational (E20, E54, E56, E59).
4. **A-phase.** Values only (routers frozen), 10k steps on the pretraining suite, frozen-base routing on (E37–E38).
5. **Sequential.** Values only, 5,000 steps per task, value LR 2e-3 → 2e-4 per block, `top_t` 3072 (1536 on the real robot), protection β=4 in rank mode with `corefrac`, frozen-base routing, per-task checkpoints with resumable cross-task state (E30, E42, E50, E53, E54 U5, E65 add-20).

Layout: expert MLP sites [4,6,8,10,14,16] with (4,6) and (8,10) sharing one table each and 14, 16 solo; VLM sites [5,7,9,11,13,15], all three adjacent pairs sharing; every bank n=256 (65,536 slots), rank-2 LoRA values, expert knn 36, VLM knn 16, four heads. Seven tables, ~2.8B value parameters, 12 sites (E62).

Evaluation: 25 episodes × 4 paired seeds per task for every headline number; retention triangle at every boundary; paired-noise MSE forgetting matrix with the fixed loader; jitter grid; slot autopsy; harvest-bank rescore where a specialist reference exists (E60 add-4, E64 add-3, E65 add-16).

---

## 8. Closed axes and dead ends

- **Value type:** static vectors (capacity ceiling) → rank-2 LoRA slots. Rank axis closed on both towers (expert +2 at E33, VLM −10 at E50). Per-slot affine bias and gate removal: no effect (E41).
- **Routing losses:** locality dead (E26); global balance only mildly useful on SmolVLA (E13); negatives-only contrastive collapses at any dose (E21–E22); standard contrastive is a breadth knob, never a family separator (E24, E54); separation is the only famIoU mover, and famIoU stopped converting to points after E27 and is dead as a gate axis (E56, E59 add-3). Background IoU is the certificate axis that pays (E56).
- **Retrieval breadth:** knn closed at 36 (expert, E17) and 16 (VLM, E49). Bank size: 384 and 256 fine, 192 and 128 dead (E48).
- **Write budget:** top_t 3072 is the interior optimum on LIBERO (E51 P9), 1536 on the real robot (E65 add-20); count top-p (19.2, E51 P2) and mass top-p (37.6, E51 P9) dead; batch 64 retired (E42).
- **Amplitude:** value LR 2× optimal (E51 P1/P8); steps saturate at 5k in the staged regime (E41).
- **Protection:** rank mode with corefrac is default (E53); grad-scale and budget modes work mechanically but tax writers (E44, E51 P8); β=4 optimal (E30); freezing top pretrain slots null-to-negative (E44); decaying stores rejected (E50); noisy-OR under sharing moot (E61 add-3).
- **Substrates:** frozen-base staged protocol without router warm-up, dead (E35); image-span VLM routing parked (E50); attention-side memory killed (E51 P5); sharing deep expert tables kills the depth-dependent task (E61); value-input noise wins on dedicated tables and loses on shared ones, dropped (E62 add-7).
- **Router conditioning:** per-token state-text routing dead (E45); no-language expert routing sprawls (E54 U3); expert anchor needs weight ~0.4 (E54); the VLM anchor works at every weight (E53 arm-3); language-only routing parked (E36); nonlinear query head null (E34).
- **Instruments:** 20-episode intermediate cells retired (E41); single-seed 50-episode finals for pre-registration scoring only (E62 add-7); chunk error ranks within a substrate only (E50); D-vs-specialist retained as an arbiter (E63 add-5).
- **Real-world:** arm-5 (anchor/language probe) disabled after top_t 1536 fixed the drift (E65 add-21).

---

## 9. Methodological rules that emerged

1. Pre-register reads and kill lines in the script header before launching; score them on landing (E29 onward; E43 owns a pre-registration built on a retired instrument).
2. Certificates (3-hour warm-up plus 35-minute audit) predicted full-pipeline behaviour repeatedly and are worth ~30 GPU-hours each; gate on background IoU first, then capacity floors (E49, E56, E59).
3. Headline claims cite the 4-seed instrument only; no mechanism claim enters the log on one seed (E60 add-4, E62 add-7).
4. Instruments must be validated against a second instrument or a known anchor (the matrix loader bug survived several entries until two instruments disagreed, E65 add-16; the harvest instrument was validated against the known chunk numbers, E57).
5. Any change to the protection path needs a multi-step Adam integration test; smokes persist alongside instruments in `scripts/vla_analysis/` (E41, E51 P4).
6. Infrastructure on the preemptible VM: transient `systemd-run` units rather than tmux (E54 U4), atomic checkpoints and resume by default (E54 U5, E58 add-5), sweep optimizer states before each new chain (E62 add-8), verify pushes landed after a session restart (E63 add-5), cross-check an unreachable VM against an independent host before concluding preemption (E58 add-2), monitors report artifact state rather than the last matching log line (E63 add-5), never rely on `systemctl is-active`'s exit code (E61 add-6), `git add -f` for `job_scripts/` (E40 update 14 Jul), every freeze-mode flag must be overridden downstream (E37 add).
7. Cross-substrate comparisons use 50-episode finals plus the jitter battery, never chunk error alone (E50); cross-run block-minimum losses from the log's `loss` field are invalid because it carries auxiliary-loss telemetry (E55).
8. `HF_HUB_OFFLINE=1` in every runner (hub 429s masquerade as tokenizer corruption, E53); two chunk probes cannot share the GPU (E53).

---

## 10. Standing heuristics (mechanism-level claims the paper can lean on)

- **Two-factor forgetting:** final ≈ f(basin depth, overwrite exposure); collapse needs a shallow basin and high exposure (E18).
- **Incidental vs genuine overlap:** write protection cleans incidental channels cheaply and cannot fix genuine same-scene contention (E19, E27–E28).
- **Breadth law:** the chunk error needed for a given success rises with the breadth of what the model trained on; success is a threshold in (task, support) (E43 add).
- **Read-write product with a conditionality factor:** usable adaptation ≈ learning signal × read participation × state-conditionality of addressing (E43 add, E48–E49).
- **Layers-versus-budget collision principle:** extra write budget reaches into shared contested banks; extra layers buy new banks with separated addressing; capacity converts iff it arrives with new separated addressing (E51 P9).
- **Three-way dissociation at depth:** placement matters (E59–E60), capacity does not (E59 add-4), content identity at depth matters (E61 add-3); share where consumption contexts are interchangeable, dedicate at depth (E61 add-5, E62).
- **Amplification model:** slot transforms applied directly at image positions amplify dependence on image features; off-demo visual drift moves the function proportionally more (E50).
- **Drift converts only at the threshold:** function drift is real and exposure-ordered at fit level yet does not convert to success loss below the cliff (E42, E52, E63 add-3).

---

## 11. Errata register (numbers and claims corrected inside the log)

| original claim | where | correction | where |
|---|---|---|---|
| E20 "SupCon package is the breakthrough" | E20 | separates but over-compacts; capacity collapse | E21 |
| E22 "separation cannot decouple" | E22 | estimator artifact plus too-low weight; sep 2.0 decouples | E23–E25 |
| E28 "up-weight scene over language" | E28 | scene already dominates 17–21×; option dead | E28 Part C |
| Rank multiplies destructiveness | E32 | fractional damage ~rank-invariant to first order | E32 appendix, E33 |
| E35 "staged protocol dead" | E35 | reversed: the staged setting is the thesis; the fix is the warm-up | E36–E37 |
| E38 "dilution fixed, per-batch effnum ≈ joint" | E38 | mean masked e9's 8973 | E39 |
| E39/E40 "e6 60→25 is eval noise" | E39–E40 | real in 4/4 (E41), then mostly init-draw artifact (E42) | E41, E42 |
| E41 hypothesis #0 (amplitude = conversion mechanism) | E41 | demoted to a ~10% fit improvement plus draws | E41 |
| E42 addendum "generalist freeze demoted" | E42 add | shallow freeze re-admitted; then measured null | E42 add, E44 |
| "Frozen backbone is the binding constraint" / "value-path capacity is not where the offset lives" | E33, E41 | falsified by the LoRA specialist (chunk 0.02 / 58 on e4) | E43 add |
| "Expert text-anchor is a decisive low-layer rescue" | E53 | it rescues famIoU/bg by cutting capacity ~4× at w0.5 | E53 PM |
| "Anchored-expert viable window is empty" | E53 arm-3 | a property of the core50 ≥ 800 gate, not the router family | E54 |
| Small cores correlate with the win | E54 | co-product; bgIoU is the causal knob | E56 |
| E62 add-5 duplicate numbering | E62 | two addenda 5 (13 Aug queue; 14 Aug noise landing = duplicate of add-6) | E63 add-5 |
| "Noise composes with sharing; D-vs-specialist retired" | E62 add-5 (dup) / add-6 | falsified at 4 seeds; D retained | E62 add-7, E63 add-5 |
| E62 5-task drift +1.2% | E62 add-2 | +3.9% (loader bug) | E65 add-16, add-24 |
| E63 10-task drift +6.5% and the add-4 grid | E63 add-3/4 | +28.5%; grid superseded | E65 add-25 |
| "Merged 6×2 indistinguishable from dedicated-table interleave" | E62 add-2 | void; slightly leakier | E65 add-24 |
| E63 block-8 e7 dip attributed to basket interference | E63 | noise-driven; withdrawn | E63 |
| E64 "per-token equivalent rank ~2" | E64 | per-site bottleneck 288 / 128; r512 chosen | E64 add-1 |
| E65 "top_t cannot remove the family channel; expect t0 +12–18%" | E65 add-15/19 | t0 +5.6%; coverage, not routing separation, was binding | E65 add-20 |
| E65 "E63 published +6.9/+6.9/…" column | E65 add-25 | the E63 add-3 table reads +12.4/+7.4/+4.7/+8.4/+18.6/+7.7/+3.9/+1.2/+0.9/+0.0 (mean +6.5); the fixed column is authoritative either way | E63 add-3 vs E65 add-25 |
| E60 add-6 "72.6 is the deleted full-FT" | E60 add-6 | 72.6 was the E31 joint 90+10 fine-tune; the all-10 full-FT cells are new (67.6 / 78.2) | E60 add-6/8/9 |

---

## 12. Status and open items (7 September 2026)

- **Paper configuration:** merged 6×2, no noise, corefrac β=4, top_t 3072, 2× LR (§2). Five-task 65.2 and ten-task 65.1 at the 4-seed instrument (§3). Target venue ICRA (E54).
- **Running (launched 15:21 UK, 7 September, E67):** RETAIN and O-LoRA under our protocol; RETAIN lands ~22:30 UK 8 September, O-LoRA ~10 September; each gets the rollout triangle and the 10×10 loss matrix. Pre-registered bands: RETAIN 15–40, O-LoRA 20–45. Watch items: eval rows slowing RETAIN training, RETAIN's 82-second periodic saves, the O-LoRA penalty trajectory (λ₁ = 0.5 as in the official code, not re-tuned).
- **Corrections to carry into the writeup:** function-drift numbers for shared-table runs are +3.9% (5 tasks) and +28.5% (10 tasks) (E65 add-24/25); the 5-task multitask-LoRA row (r32, 1k steps/task) is under-provisioned and needs a rank-512 twin or a footnote (E64); every LoRA and memory row trains from the libero_90 stage-1, the raw-pi05 row is FT-fresh only (E64); the table caption must state data budgets (E60 add-7).
- **The named residual:** the basket family (e7 soup+cheese, e0 soup+sauce, e1 cheese+butter). Routing cannot separate it (scene-genuine, E28), protection cannot fully shield the hub (E27–E28), it is the entire deficit against specialists at ten tasks (E63 add-2), and e7 is the worst drift cell at +117% (E65 add-25). Two mechanisms in one family: e7 is forgetting-damaged, e1 is fit/conversion-limited (E63 add-3).
- **Real robot:** every real-world number is function-space; drift-to-success conversion is uncalibrated without rollouts (E65 add-23). The real-world chain otherwise reproduced the simulation recipe end to end, including the write-mask saturation fix at top_t 1536.
- **Deferred:** lesion map on the 12-site layout (E60, E61 add-5); training-seed replicates; protection-off ablation on the paper cell; zero-shot and B 4-seed rows; a rank-512 twin of the 5-task multitask row; RETAIN at alpha 0.8 (their LIBERO single-task value, ~31h train + ~28h eval, E67); fresh sharepairs-e7 harvest (E61 add-4).

---

## 13. Where things live

- **Log and summary:** `lerobot/projects/research_log.md`, `lerobot/projects/summary_research_log.md`.
- **Code:** `src/lerobot/policies/modules/{memory_config.py, memory_lite.py}`, `src/lerobot/policies/pi05/modeling_pi05.py`, `src/lerobot/scripts/lerobot_sequential_train.py`; external baselines under `scripts/baselines/` (E67); instruments and smokes under `scripts/vla_analysis/` (real-world under `scripts/vla_analysis/realworld/`); ops watchers under `scripts/ops/`; job scripts under `job_scripts/nebius/{libero_90/staged, baselines, realworld}/` (gitignored, `git add -f`).
- **Headline artifacts on the VM:** 4-seed campaign JSONs `outputs/analysis/e60/seeds_*.json` (all sim rows incl. the two retention triangles `seeds_tri_*`), `outputs/analysis/e62/`, `e63/`, `e65_rematrix/` (fixed matrices), `realworld/e65/` (real-world table), `e67/` (RETAIN/O-LoRA rows as they land), `e56_offtrail/` (harvest bank), `_run_artifacts/<run>/` (memory_by_task + evals + wandb for archived runs).
- **Checkpoints:** the stage-1 base `libero_90_pi05_base_nomem_50k` and the merged 6×2 A-phase stay on the VM; completed sim-era runs (bigsearch, sharepairs, interleave, merged 6×2 5-task, 10-task, vnoise, warm-ups) are in cold storage at `/media/josh/Backup/memory-models` (E53 "Important", E54 teardown, E60 add-10, E65 add-18); ask Josh to rsync back for analysis. Optimizer states were not archived; weights plus `sequential_state.pt` suffice for matrix re-runs.
- **VM operations:** `phddev/CLAUDE.md` §9 (preemptible VM contract, `systemd-run`, linger, git local-first).
