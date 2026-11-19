# Research Log 2 - VLA Memory on SmolVLA

## Executive summary (state at 18 Nov 26)

**Problem.** Learn new tasks one after another without forgetting earlier ones: frozen backbone, no task identity at inference, no replay, no per-task parameters.

**Method.** Sparse product-key memory modules sit beside selected MLPs (MLP(x) + Memory(x)) in the action expert and the VLM language model. Each slot value is a rank-2 LoRA transform of the hidden state; sequential adaptation updates only slot values. Three mechanisms keep tasks apart:
- **Geometry before content.** The backbone is fine-tuned on the pretraining suite (libero_90) without memory. A router warm-up then trains only keys and query projections (values pinned at zero) on routing losses: inter-task separation on the joint slot distribution, a small contrastive term, an expert router anchored to the pooled instruction hidden state, and a pooled VLM key for the state region. The routers then freeze and an A-phase fills the values on libero_90.
- **Stationary addressing.** Routing reads memory-free backbone features (a frozen pre-pass), so writing values never moves where a task reads.
- **Protected writes.** Each step writes only the top-t slots by TF-IDF score (t = 3072 per site); earlier tasks' core slots score zero (protection β = 4, rank mode, corefrac).

**Paper cell** (pi05: PaliGemma-2B VLM + 300M action expert). Expert sites [4,6,8,10,14,16], VLM sites [5,7,9,11,13,15]; adjacent pairs share a table except the two deepest expert sites: 12 sites, 7 tables of 65,536 slots, ~2.7B value parameters. Sequential: 5,000 steps per task, value LR 2e-3 → 2e-4. Instrument: 25 episodes × 4 paired seeds per task, a retention triangle and a paired-noise loss matrix.

**pi05 headline results** (ten libero_10 tasks in sequence, 4-seed means; drift = own-task loss from just-trained to final):

| Row | Success | Drift |
|---|---|---|
| Full fine-tune on all ten tasks jointly, from the libero_90 base (ceiling) | 79.7 | — |
| Multitask LoRA r512 (one adapter, all tasks jointly) | 77.1 | — |
| Per-task LoRA specialists, task identity given (r512 / r128 / r64 / r32) | 74.9 / 69.5 / 68.8 / 63.7 | — |
| **Ours, sequential** | **65.1** | **+28.5%** |
| Naive sequential LoRA r512 / parameter-matched r1216 | 9.7 / 8.6 | +740 to +1567% (r256, 5 tasks) |
| RETAIN (weight merging, α 0.5) | 8.2 | +477% |
| Naive sequential full fine-tune | 7.7 | — |
| O-LoRA r64 | 6.5 | +72% (stops learning after task 2) |

- **Retention.** Earlier tasks average 62.2 after all ten blocks (naive: 0.0); tasks 1–5 score 64.6, tasks 6–10 65.6.
- **Order and length.** Reversed order: 62.4, inside the ±3-point replicate band. Five tasks: 65.2, drift +3.9%.
- **Ablations** (five tasks, control 66.0):
  - No stationary addressing: 64.4, but retention change Δ_final −4.0 against the control's +0.4, all on the oldest task.
  - No protection: 59.8, oldest task 32 against 53.
  - Routers and values trained jointly at matched LR: 59.0, lost acquiring the newest task (37 against 51).
  - No separation loss: fails the background-overlap gate at all six expert sites. No contrastive loss: little change.
- **Real robot** (WidowX AI, five tasks, function space only): drift +2.7% against +2356% for naive LoRA, and lower own-task loss than every r64 specialist. No robot rollouts exist.
- **Limits.** The basket family (envs 7, 0, 1: one scene, near-identical tasks) accounts for the whole deficit against the specialists; env 7 is the worst drift cell. Overall the method matches roughly an r45 specialist and sits 14.6 below the joint fine-tune.

**SmolVLA reproduction** (Entries 1–4). The same staged recipe and instrument on SmolVLA (~450M parameters), from an all-weights libero_90 base.
- Layout: 11 sites, 7 tables, ~1.5B value parameters. Warm-up at separation 2, the only separation weight that keeps every expert site above the core50 floor; otherwise pi05's cell settings.
- Chunks execute 10 actions (pi05: 50), so absolute numbers do not compare with pi05.
- LoRA rows adapt attention and MLP in both towers with α = r. LR is 1e-3, except where 1e-3 was unstable: r256 specialists and naive sequential at 3e-4, multitask at 1e-4.

| Row (ten libero_10 tasks, 4-seed means) | Success |
|---|---|
| Per-task LoRA specialists, task identity given (r32 / r64 / r128 / r256) | 85.9 / 85.8 / 84.5 / 84.1 |
| **Ours, sequential** | **81.7** |
| Full fine-tune on all ten tasks jointly | 79.6 |
| Multitask LoRA r128 (one adapter, all tasks jointly) | 72.2 |
| Zero-shot (libero_90 base) | 7.2 |

- **Retention.** Each task scores 86.5 on average just after training, matching the specialists. After all ten blocks the earlier tasks average 79.8, 5.3 below their just-trained level. The loss is uneven (e4 −20, e7 −16, the rest within ±7) and does not grow from five tasks (−5.0) to ten.
- **Naive sequential LoRA r128** (five tasks): every earlier task is at 0 from block 2; final row 10.6, against 81.0 for the memory's five-task run.
- **Against pi05's shape.** As on pi05, sequential memory sits far above naive training and close to the specialists. Unlike pi05, the specialists do not rise with rank, and the two joint-training rows sit below both the specialists and the memory (on pi05 they were the top two rows).
- **Replicate.** The ten-task run repeats the five-task run's first five blocks to within ~3 points per cell.

The SmolVLA prototype (E0–E17) used the earlier joint protocol on four tasks (best 57.5%, not comparable). History: `research_log.md`, summarised in `summary_research_log.md`.

**Plan: Paper 1, capacity, growth and priors for sparse memory** (the thesis plan, `phd_deliverables/upgrade_viva/conversion/chapters/05_future_work.tex`). Five items. Items 1 → 2 → 3 run in sequence, each building on the one before; items 4 and 5 are parallel threads.

1. **Scale the task sequence to 30–60 tasks.** The tasks are new environments built in the LIBERO simulator, with demonstrations generated for them. They are not drawn from libero_90 (the thesis plan's source), which is the pretraining corpus.
   - Protocol as now, in at least two task orders: related tasks grouped, and interleaved.
   - Retention matrix in full at every fifth task, the diagonal at every task.
   - Deliverable: acquisition, retention and loss increase as functions of the number of tasks K, with the protected-slot statistics at each boundary.
   - Why this range: on pi05 each task's core is ~1.7% of an expert table, and unrelated cores overlap at ~0.024 per pair, close to random placement. The share of a new task's reads outside every earlier core then falls as ~(1 − 0.024)^(K−1): 0.80 at K = 10 (measured 0.72–0.86), 0.30 at K = 50. The point where the write rule stops finding enough unprotected slots has never been measured.
2. **Measure capacity and calibrate a saturation detector,** on item 1's runs. Set marginal plasticity (task K's success at a fixed adaptation budget, against K) against the protection statistics: the protected fraction, and ω_K, the share of a new task's read mass outside every earlier core. Both statistics come from a forward pass before any values are updated. The detector is the ω_K (or protected fraction) at which plasticity starts to fall.
3. **Add capacity on detected saturation.** Grow only when the detector fires, and never change an existing lookup (stationary addressing). Candidates:
   - a writable value stacked on a contended slot (equivalent to widening the slot);
   - reserved tables, warmed up but held at zero values with gates closed, opened when the detector fires;
   - a floor on the protection score, so stale cores can be revised (consolidation; no new parameters).

   Compared on item 1's saturated sequences: acquisition of tasks after the trigger, retention of tasks before it.
4. **Meta-learned prior over memory contents.** Replace multitask value fill with first-order meta-learning (Reptile or first-order MAML) of the values' initialisation and the components every task shares (gates, output projections, shared tables). Deliverable: demonstrations or adaptation steps to reach a target success on a held-out task, for zero initialisation, multitask fill and the meta-learned prior.
5. **Selective engagement.** The memory is always in the loop, and the pretraining tasks have never been evaluated after sequential adaptation.
   - First step: libero_90 success before and after the ten-task sequence. Both SmolVLA checkpoints exist (the libero_90 base and the ten-task run's final).
   - If retention there is acceptable, the rest is a refinement. If not, add a per-input bypass to the memory-free pre-pass prediction, decided by one of: retrieval confidence, the existing gate trained to close on pretraining inputs, a null slot the routers learn to send pretraining inputs to, or the spread of sampled action chunks.

---
## Entry 1 - 6 Nov 26 (SmolVLA reproduction of the paper recipe, and the baselines it needs)

**Why SmolVLA.** Compute is now a shared cluster. SmolVLA (~450M parameters) costs about a tenth of pi05 per step, which buys iteration and training-seed replicates (pi05 had one replicate pair, hence the ±3 band). The aim is to reproduce the *shape* of the pi05 result under the same protocol (libero_90 → libero_10, same task order, same instrument): sequential memory at least matching the per-task specialists, far above naive sequential training, with small drift. Absolute numbers are not expected to transfer, so comparisons stay within SmolVLA.

**Base (stage 1).** All weights end to end, mirroring pi05's stage 1: smolvla_base on libero_90, no memory, 50k steps at batch 32, warmup 4k, cosine decay over 50k.
- Optimizer: SmolVLA's default (AdamW 1e-4 → 2.5e-6), as pi05 used its own (2.5e-5). With bf16 weights most 2.5e-5 updates would fall below bf16 resolution.
- Dtypes as HF's SmolVLA recipe (bf16 VLM and expert, no autocast). Two cameras, HF's LIBERO convention (pi05 filled a third slot with an empty image).
- All weights because the memory routes on and adapts VLM features, which on pi05 were LIBERO-adapted; the libero_90 substrate was worth +10.6 even to a full fine-tune (E60 add-9).

Both datasets were rebuilt from public sources and checked against the simulator by open-loop replay.

**Evaluation.** Same instrument (25 episodes × 4 paired seeds, retention triangle, paired-noise loss matrix), executing 10 actions per predicted chunk (`n_action_steps=10`). The SmolVLA paper's LIBERO ablation (its Table 13) gives 80.3 / 82.8 / 70.8 / 51.8 for 1 / 10 / 30 / 50 actions per chunk: 50 would push every row toward the floor, and replanning every step buys nothing over 10. The pi05 rows used 50 (pi05's default), one more reason the absolute numbers do not compare.

**Baselines.** From the SmolVLA libero_90 base, same instrument: zero-shot floor (pi05: 10.6); joint all-weights fine-tune on all ten tasks (ceiling); ten per-task LoRA specialists; multitask LoRA (one adapter, all ten tasks, 50k steps); naive sequential LoRA. They set the bars, the per-task difficulty and the seed noise that later arms are scored against.

Open: LoRA targets and rank. HF's default SmolVLA LoRA adapts only the expert's q/v and the action/state projections; the pi05 rows adapted attention and MLP in both towers. pi05's rank rule (smallest power of two above the per-site bottleneck, 2 × heads × knn = 288) gives r512, close to full rank on SmolVLA's 720- and 960-wide matrices.

**Memory port.** Every memory option carries over. Differences that matter:
- SmolVLA's VLM is SmolVLM2 cut to 16 layers (width 960), with a 16-layer expert (width 720); attention alternates between joint self-attention and expert-to-VLM cross-attention.
- The robot state is one continuous token in the prefix (pi05: dozens of state-as-text tokens), so the VLM memory covers the instruction tokens plus one state token.
- Nothing reads the last VLM layer's output, so it can carry no memory.

First layout, pending the probes below: expert [4,6,8,10,12,14] with (4,6) and (8,10) shared; VLM [5,7,9,11,13] with (5,7) and (9,11) shared. pi05's relative depths without the dead layer: 11 sites, 7 tables.

**What might not transfer.**
1. With one state token the VLM sites see far fewer distinct reads per batch. That risks write-mask saturation (E65 add-14: protection is bypassed once top-t reaches the read-set size) and leaves less VLM capacity per sample, where on pi05 the VLM side was the largest fit lever (E44).
2. Routing-loss weights, gate bands, top-t, value LR and bank size were calibrated on pi05; the real-robot port needed four warm-up arms (E65).
3. Seven tables are ~1.5B value parameters on a 450M model (pi05: ~0.7× the model), which sharpens the "added parameters" objection; bank size may need revisiting.

**Order.**
1. Base, zero-shot floor and baselines.
2. Alongside: an E44-style compass (LoRA on the expert, the VLM or both, on e4, e9, e7) and a per-layer geometry probe of the frozen base, to choose the layout.
3. The paper cell: warm-up and held-out audit, then A-phase and five tasks, with triangle and loss matrix.
4. Recalibrate whatever fails, one change per arm, cheapest instrument first.
5. Confirm on ten tasks, in reversed order, with replicate training seeds.

---
## Entry 2 - 7 Nov 26 (router warm-up sweeps on the SmolVLA libero_90 base: separation and expert anchor)

**Why.** The warm-up's geometry depends on how crowded the frozen backbone's features are, which changes with the backbone. On pi05 two knobs decided it:
- **Separation.** Removing it failed the background-overlap gate at all six expert sites (E68 C1).
- **Expert anchor weight.** Narrow window: 0.5 over-compacted, 0.35 failed, 0.40 passed (E53–E54). It mixes the normalised pooled instruction hidden state with the token's own normalised features, so its best value depends on how crowded the expert features are relative to the instruction features.

Contrastive was not swept: E68 C2 found it redundant with pooled VLM routing (revisited in Entry 3).

**Design.**
- Start: the SmolVLA libero_90 base (Entry 1).
- Reduced layout: three solo sites per tower, expert [4, 8, 14] and VLM [5, 9, 13], no sharing. With frozen-base routing each site's router trains on its own losses from memory-free features, so per-site certificates should carry over to fuller layouts.
- Warm-up as the paper cell (10k steps, batch 32, router LR 1e-4, values pinned at zero, contrastive 0.05, VLM pool (1, 0.5), pre-pass). Audit as pi05's (all ten libero_10 tasks, 400 steps × batch 8, value LR 1e-12).
- Arms: a reference at separation 8 and anchor 0.4 (the paper values), plus single changes: separation 2 and 32, anchor 0.2 and 0.6.
- Reads per site: famIoU, bgIoU, core50, min-task effnum. pi05's bands (expert bgIoU ≤ 0.10, mean core50 ≥ 400, min effnum ≥ 300; VLM min effnum ≥ 150) are a guide, not a gate.

**Pre-registered reads.**
- Separation: overlap falls as separation rises, with capacity the likely price at 32. Take the lowest overlap that keeps capacity above the floors.
- Anchor (expert sites only): pi05 showed a U in famIoU, with capacity falling as the anchor rises. Where the U's bottom sits says how crowded SmolVLA's expert features are relative to its instruction features.
- Plan: reference passes → the best value on each axis goes to the full layout; capacity short in every arm → next sweep follows the real robot (contrastive up, separation down); overlap high in every arm → separation higher.

**Correction (11 Nov 26): the VLM half of this sweep is void.** Two bugs on the SmolVLA span path:
- Truncated span: every sample was cut to the batch's shortest valid span. The state token ends each span, so the losses saw ~47% of span positions and the state token in ~6% of samples, mostly the instructions' shared leading words.
- Wiped queue: the routing queue was re-created empty whenever the batch's longest span changed, several times per 30 steps.

The VLM routers were trained on almost no task signal. The expert sites were unaffected (no token mask, fixed 50 rows per sample). Fixed in 5f59e3ef; the separation arms were rerun (addendum).

**Results** (8 Nov 26). Each cell: famIoU / bgIoU / mean core50 / min-task effnum, held-out audit on all ten libero_10 tasks. Expert sites:

| Arm | E4 | E8 | E14 |
|---|---|---|---|
| separation 2 | 0.106 / 0.026 / 554 / 563 | 0.094 / 0.025 / 524 / 685 | 0.138 / 0.055 / 1130 / 895 |
| **reference** (separation 8, anchor 0.4) | 0.088 / 0.018 / 400 / 423 | 0.075 / 0.018 / 393 / 529 | 0.122 / 0.041 / 817 / 787 |
| separation 32 | 0.076 / 0.012 / 299 / 332 | 0.069 / 0.012 / 299 / 407 | 0.105 / 0.029 / 579 / 716 |
| anchor 0.2 | 0.155 / 0.058 / 1246 / 805 | 0.106 / 0.042 / 886 / 868 | 0.145 / 0.073 / 1538 / 1027 |
| anchor 0.6 | 0.071 / 0.005 / 136 / 282 | 0.067 / 0.007 / 171 / 272 | 0.105 / 0.023 / 453 / 684 |

VLM sites: void (see the correction); first-sweep and rerun numbers are in the addendum.

Against the pi05 expert bands:
- Separation 2 and anchor 0.2 pass every clause.
- The reference misses narrowly: E8 mean core50 393 against 400; E4 is exactly 400.
- Separation 32 fails core50 at E4 and E8 (299).
- Anchor 0.6 fails both capacity clauses at E4 and E8 (core50 136 / 171, min effnum 282 / 272).

**Reading.**
- **Capacity, not overlap, is the constraint on SmolVLA.** Every expert site is well inside the overlap band in every arm (worst bgIoU 0.073). The arms differ in whether the cores stay large enough, and the paper settings sit on the capacity floor.
- **Separation behaved as pre-registered.** Overlap falls monotonically with separation (E4 bgIoU 0.026 → 0.018 → 0.012) and the cores shrink with it (E4 core50 554 → 400 → 299).
- **Anchor: no U.** Over 0.2–0.6, overlap, family overlap and core size fall together at every expert site (E4 famIoU 0.155 → 0.088 → 0.071; core50 1246 → 400 → 136). Capacity collapses at 0.6 before any famIoU minimum appears, as 0.5 over-compacted on pi05.
- **Outcome.** Neither fallback applies. Separation 2 (anchor 0.4) and anchor 0.2 (separation 8) pass; both trade overlap for capacity, which SmolVLA can afford. Ranked background-overlap first (the pi05 rule, E56), separation 2 leads (E4/E8/E14 bgIoU 0.026/0.025/0.055 against 0.058/0.042/0.073); ranked on capacity, anchor 0.2 leads (cores 1.6–2.3× larger).

### Entry 2 addendum (12 Nov 26): separation arms rerun with the VLM loss fix

**Expert sites replicate.** The fix does not touch expert routing. famIoU and bgIoU match the first sweep to three decimals, mean core50 to within 1% (reference E14 822 against 817) and min effnum to within 2%. The expert readings stand.

**VLM sites, first sweep (truncated span) and rerun (every valid span row).** famIoU / bgIoU / mean core50 / min-task effnum:

| Separation | V5 | V9 | V13 |
|---|---|---|---|
| 2, first | 0.212 / 0.050 / 777 / 1320 | 0.194 / 0.061 / 1113 / 1815 | 0.215 / 0.091 / 1770 / 2693 |
| 2, rerun | 0.204 / 0.043 / 827 / 843 | 0.189 / 0.053 / 1084 / 1041 | 0.202 / 0.080 / 1832 / 2168 |
| 8, first | 0.213 / 0.049 / 745 / 1263 | 0.191 / 0.060 / 1101 / 1693 | 0.217 / 0.088 / 1720 / 2593 |
| 8, rerun | 0.206 / 0.042 / 781 / 546 | 0.186 / 0.048 / 933 / 606 | 0.200 / 0.070 / 1508 / 1441 |
| 32, first | 0.212 / 0.048 / 733 / 1238 | 0.192 / 0.057 / 1045 / 1366 | 0.216 / 0.085 / 1632 / 2243 |
| 32, rerun | 0.191 / 0.039 / 705 / 452 | 0.188 / 0.042 / 776 / 409 | 0.203 / 0.061 / 1177 / 913 |

**Reading.**
- **The VLM routers now respond to separation.** Overlap and footprint fall together as separation rises, at every site (V13 bgIoU 0.080 → 0.070 → 0.061; min effnum 2168 → 1441 → 913).
- **The fix alone tightens and separates the VLM routing.** At matched separation, min effnum drops 1.5–3× and bgIoU 10–30%. At V5, overlap falls while the mean core grows (reference: bgIoU 0.049 → 0.042, core50 745 → 781). Per-task core ranges widen (reference V5 424–1204 → 209–1559), so footprints now differ by task.
- **Within the sweep, VLM separation slides along one overlap-versus-footprint curve,** as at the expert sites: bgIoU ÷ core50 is roughly constant across weights (V5 5.2, 5.4, 5.5 ×10⁻⁵).
- **Family overlap does not move:** famIoU 0.19–0.21 in every arm, against 0.08–0.145 on pi05's cell. Untested explanation: instruction tokens are routed one at a time and dominate the VLM reads, and the basket tasks' instructions share most of their words.
- **Against pi05's VLM cell:**
  - bgIoU is inside its range at separation 8 and 32 (0.039–0.070 against 0.023–0.074); separation 2's V13 is 0.080.
  - Min effnum is inside at V5/V9 for separation 8 and 32 (409–606 against 250–750) and above at V13; every arm clears the floor of 150.
  - Mean core50 is still broader at V9/V13 (776–1832 against 188–842).

---
## Entry 3 - 12 Nov 26 (separation 2 carried forward; contrastive sweep; first paper-cell chain on SmolVLA; zero-shot floor)

**Decision: separation 2, anchor 0.4.** One separation weight serves both towers.
- Expert: the only separation weight that clears the core50 floor at every site (reference E8: 393 and 394 in the two sweeps; separation 32: 299). Its overlap (bgIoU 0.026 / 0.025 / 0.055) is below pi05's cell at the same depths (0.038 / 0.037 / 0.058) and below anchor 0.2's (0.058 / 0.042 / 0.073).
- VLM: the loosest of the three (min effnum 843–2168; V13 bgIoU 0.080), but all three clear the VLM floor. No per-tower weight unless the VLM side causes trouble.

**Contrastive sweep.** On pi05 the contrastive weight c moved capacity, but its direction flipped between regimes:
- Joint training (E24–E25): c compacted footprints (core50 2,643 / 1,752 / 1,162 at c 0.01 / 0.05 / 0.1; c = 0 sprawled).
- Frozen base (E68 C2): removing c shrank the cores (expert core50 713–2,019 → 432–1,446; VLM min effnum 250–750 → 158–276) and lowered overlap with them.
- Real robot (E65): the arm that cleared capacity raised c from 0.3 to 1.0, together with three other changes.

That is one clean frozen-base measurement. Arms: c = 0, 0.2 and 1.0 at separation 2, warm-up and audit on the reduced layout (`wu2_sep2` is c = 0.05).

**Pre-registered reads.**
- **Primary read: overlap per unit footprint (bgIoU ÷ core50).** On SmolVLA, separation and anchor both move along one curve: the ratio is 4–5 ×10⁻⁵ at every expert site in every arm. Footprints placed at random also overlap in proportion to their size, so these knobs resize footprints without placing them better.
  - Flat across c: c is a third copy of the same dial; keep 0.05 and close the axis.
  - Falling (larger cores without proportionally more overlap, or less overlap without proportionally smaller cores): c moves the curve. Worth a full chain, probably with a higher separation.
- pi05's hint is mixed: in E68 C2, removing c lowered the ratio by 15–36% at the expert sites and raised it by 10–47% at the VLM sites.
- famIoU should not move: contrastive never separated the basket family on pi05 (E54 P2, E68 C2).
- At c = 1.0, watch for query collapse: cores shrinking to a handful of slots (E21).

**First paper-cell chain on SmolVLA.**
- Layout: Entry 1's (11 sites, 7 tables, ~1.5B value parameters).
- Warm-up at separation 2; otherwise the pi05 cell: contrastive 0.05, expert anchor 0.4, VLM pool (1, 0.5), pre-pass; A-phase 10k on libero_90; 5,000 steps per task; value LR 2e-3 → 2e-4; top-t 3072; protection β4, rank mode, corefrac.
- Five tasks (front-5: envs 4, 6, 9, 2, 7), as pi05's development runs.
- Instrument: retention triangle at every task boundary, 25 episodes × 4 paired seeds per cell, 10 actions per chunk. The audit covers all ten tasks; its solo sites (E14, V13) test whether the reduced-layout certificates carry over.

The question is whether the staged recipe reproduces pi05's shape on SmolVLA before any recalibration. On pi05, earlier tasks held their level through the sequence (prior-task mean 69.8 at block 5 against a diagonal of 65.6), where naive sequential training falls to zero. The main risk (Entry 1) is write-mask saturation at the VLM sites, which see one state token per sample. Absolute success needs SmolVLA's own bars, still to come.

**Zero-shot floor.** The libero_90 base on all ten libero_10 envs, same instrument (pi05: 10.6, single seed, 50-action chunks). It shows how much of any later score is adaptation.

Result (12 Nov 26): **7.2** mean over 4 seeds. Per env in train order: e4 0, e6 0, e9 0, e2 2, e7 0, e0 4, e8 0, e1 0, e3 0, e5 66. Every task but e5 (book into the caddy) is at or near zero; e5 is largely solved by the libero_90 base alone, so its later scores say little about adaptation.

### Entry 3 addendum (13 Nov 26): results

**Contrastive sweep** (separation 2; famIoU / bgIoU / mean core50 / min-task effnum, bgIoU ÷ core50 ×10⁻⁵ in brackets):

| c | E4 | E8 | E14 | V5 | V9 | V13 |
|---|---|---|---|---|---|---|
| 0 | 0.052 / 0.011 / 364 / 263 (3.1) | 0.079 / 0.011 / 285 / 336 (4.0) | 0.100 / 0.021 / 439 / 637 (4.7) | 0.250 / 0.058 / 215 / 179 (27.2) | 0.236 / 0.043 / 242 / 215 (17.6) | 0.202 / 0.054 / 938 / 277 (5.8) |
| 0.05 | 0.106 / 0.026 / 551 / 554 (4.8) | 0.094 / 0.025 / 522 / 690 (4.9) | 0.138 / 0.055 / 1129 / 894 (4.9) | 0.204 / 0.043 / 827 / 843 (5.2) | 0.189 / 0.053 / 1084 / 1041 (4.9) | 0.202 / 0.080 / 1832 / 2168 (4.4) |
| 0.2 | 0.110 / 0.028 / 581 / 593 (4.8) | 0.097 / 0.027 / 563 / 725 (4.9) | 0.141 / 0.058 / 1178 / 928 (4.9) | 0.204 / 0.044 / 827 / 880 (5.3) | 0.187 / 0.053 / 1101 / 1092 (4.9) | 0.201 / 0.081 / 1831 / 2201 (4.4) |
| 1.0 | 0.110 / 0.029 / 592 / 601 (4.9) | 0.100 / 0.028 / 578 / 740 (4.9) | 0.142 / 0.058 / 1197 / 930 (4.9) | 0.205 / 0.044 / 831 / 885 (5.3) | 0.189 / 0.054 / 1110 / 1096 (4.8) | 0.202 / 0.081 / 1841 / 2212 (4.4) |

- **No dose response above 0.05.** From 0.05 to 1.0 expert cores grow 5–7% with a flat ratio, and the VLM sites move by about 1%. No collapse at 1.0. The term acts as on/off, not as a dial.
- **c = 0 splits the towers, as pi05's E68 C2 did.**
  - Expert: the ratio falls (E4 −35%, E8 −18%, E14 −4%) and E4's famIoU halves, contrary to the pre-registered read that famIoU would not move. But every core falls below the 400 floor, and E4's min effnum drops to 263.
  - VLM: footprints collapse (V5/V9 core50 ~200), the ratio worsens 3.6–5× and famIoU rises. VLM min effnum (179–277) nearly matches pi05's C2 (158–276).
- **Decision:** keep c = 0.05; the axis is closed.

**Full-layout certificate** (`cell_s2_seq5`, warm-up at separation 2, audit on all ten tasks; * = shared table):

| Site | E4* | E6* | E8* | E10* | E12 | E14 | V5* | V7* | V9* | V11* | V13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| famIoU | 0.134 | 0.103 | 0.115 | 0.116 | 0.121 | 0.140 | 0.211 | 0.205 | 0.188 | 0.193 | 0.207 |
| bgIoU | 0.035 | 0.030 | 0.035 | 0.036 | 0.045 | 0.060 | 0.045 | 0.049 | 0.053 | 0.066 | 0.089 |
| core50 | 752 | 615 | 678 | 737 | 942 | 1239 | 839 | 954 | 1116 | 1428 | 2006 |
| min effnum | 741 | 624 | 627 | 791 | 752 | 1062 | 960 | 945 | 1306 | 1583 | 2323 |

- **Passes every pi05 clause.** Expert bgIoU 0.030–0.060, core50 615–1,239, min effnum 624–1,062; VLM min effnum 945–2,323.
- **The reduced-layout certificates carry over.**
  - The solo sites E14 and V13 come out ~10% broader than in the reduced layout, with ~10% more overlap and the same ratio (4.9, 4.4). Take ~10% as the run-to-run spread of a certificate.
  - The shared expert sites (E4, E8) have cores 30–36% larger than the same sites solo, with the same ratio.

**Retention triangle** (five tasks; 25 episodes × 4 paired seeds per cell; bold = just trained):

| After block | e4 | e6 | e9 | e2 | e7 | Row mean |
|---|---|---|---|---|---|---|
| b1 | **82** | | | | | 82.0 |
| b2 | 85 | **83** | | | | 84.0 |
| b3 | 87 | 72 | **84** | | | 81.0 |
| b4 | 75 | 76 | 82 | **85** | | 79.5 |
| b5 | 73 | 72 | 83 | 86 | **91** | 81.0 |
| zero-shot | 0 | 0 | 0 | 2 | 0 | 0.4 |

- **Fit.** Every task goes from a 0–2 floor to 82–91 when trained (diagonal mean 85.0). Final row 81.0.
- **Retention.** The four earlier tasks average 78.5 at the end against 83.5 just after training (−5.0). Paired by start state, just-trained → final:

  | Task | Change | Lost / gained | McNemar p |
  |---|---|---|---|
  | e4 | −9 | 15 / 6 | 0.08 |
  | e6 | −11 | 19 / 8 | 0.05 |
  | e9 | −1 | 10 / 9 | 1.0 |
  | e2 | +1 | 10 / 11 | 1.0 |
  | pooled | — | 54 / 34 | 0.04 |

  e6 dropped when e9 was trained (83 → 72), and e4 when e2 was trained (87 → 75).
- **Against pi05.** pi05's five-task run held its earlier tasks level: prior-task mean 69.8 at block 5, against a diagonal of 65.6. SmolVLA fits each task much better but leaks on its two oldest tasks. Absolute levels are not comparable: 10 actions executed per chunk here, 50 on pi05; the SmolVLA paper gives 82.8 vs 51.8 for those settings.
- **Not yet interpretable without SmolVLA baselines.** Naive sequential training shows how much forgetting the memory prevents; the joint fine-tune and the specialists show how far 81 sits from the ceiling.

---
## Entry 4 - 13 Nov 26 (SmolVLA baselines; the ten-task memory run)

**Why.** The five-task chain (Entry 3) cannot be read without SmolVLA's own bars: how much forgetting the memory prevents (naive sequential) and how far it sits from the ceiling (joint fine-tune, specialists).

**Common to every row.** From the SmolVLA libero_90 base; the headline instrument (25 episodes × 4 paired seeds, 10 actions per chunk); all ten libero_10 tasks, except naive sequential, which runs the five-task order so its triangle sits beside the chain's.

**LoRA settings.**
- **Placement as pi05's baselines:** attention q/k/v/o and MLP gate/up/down in all 16 layers of both towers, plus the state/action/time projections. That is 229 modules, 5 of them inert: VLM layer 15's q/o/MLP, whose outputs SmolVLA never reads. HF's default (the expert's q/v only) would be a weak baseline.
- **α = r** (gain 1), replacing pi05's untuned α = r/4.
- **Per task as pi05's rows:** 5,000 steps, batch 32, AdamW, LR linear to LR/10, optimizer rebuilt per task. Specialists are one-task runs of the same trainer as naive sequential, so the two share one recipe.
- **LR by calibration.** LR ∈ {3e-5, 1e-4, 3e-4, 1e-3} at ranks 32 and 256 (the ends of the sweep), as specialists on e4 and e1 (two of the hardest tasks on pi05). Evaluated on seeds 5000/6000, outside the instrument's 1000–4000. Rule: the best mean over the two tasks. If both ranks peak at the same LR, or within ~5 points of it, that LR applies to every LoRA row; otherwise the choice comes back for discussion.

**Rows.**

| Row | Training | Evaluation |
|---|---|---|
| Specialists r32 / r64 / r128 / r256 | one adapter per task, on that task's demos | own task |
| Naive sequential r128 | one adapter through the five tasks in order | retention triangle |
| Multitask LoRA r128 | one adapter on all ten tasks jointly, 50k steps (pi05's multitask length) | all ten |
| Joint fine-tune | all weights on all ten tasks, 50k steps, batch 32, warmup 4k (the stage-1 recipe) | all ten |

r256 extends the sweep upward because the memory's just-trained mean is 85: if r128 specialists land below it, the bracket would not close (pi05 needed r512 above r128 for the same reason, E64).

**Pre-registered reads.**
- **Naive sequential:** earlier tasks fall to near zero within a block or two (pi05: 0 from block 2). Substantial retention here would weaken the memory's retention claim on SmolVLA.
- **Specialists:** success rises with rank. The memory's equivalent specialist rank is where its per-task mean lands on the ladder (pi05: ~r45).
- **Joint fine-tune:** the highest row (pi05: 79.7), with multitask LoRA just below it (pi05: 77.1).

**Ten-task memory run.** The sequential stage and triangle from `cell_s2_seq5`'s A-phase, all ten tasks.
- Its first five blocks repeat the five-task chain up to GPU nondeterminism, which gives a measured noise level for the triangle.
- The later blocks show whether the leak on the oldest tasks grows with more tasks behind them.

### Entry 4 addendum (16 Nov 26): results

**LR calibration** (specialists on e4 and e1, seeds 5000/6000; mean success over the two tasks). Both ranks peak at 1e-3, so the rule chose 1e-3:

| Rank | 3e-5 | 1e-4 | 3e-4 | 1e-3 |
|---|---|---|---|---|
| r32 | 70.0 | 78.0 | 71.0 | 79.0 |
| r256 | 72.0 | 75.0 | 78.0 | 83.0 |

**LR 1e-3 was unstable for three rows**, which were rerun at lower LRs:
- **r256 specialists:** training diverged on 4 of 10 tasks (envs 9, 7, 0, 3). Their loss rose mid-run and ended at 0.30–0.80, against 0.03–0.08 on stable tasks. Those tasks scored 1, 18, 0 and 14.
- **Multitask LoRA:** loss stalled at ~0.4 with gradient spikes, and ended at 0.38 (joint fine-tune: 0.04).
- **Naive sequential:** each later task trained worse (end loss 0.05 on task 1 to 0.26 on task 5).

The calibration (two tasks, 5k steps, top edge of the grid) did not expose this. Reruns: multitask at 1e-4 (pi05's multitask LR), r256 at 3e-4 on all ten tasks, naive sequential at 3e-4. Specialists r32–r128 trained stably at 1e-3 and stand.

All three reruns trained stably:
- **r256 specialists:** every task ended at loss 0.028–0.065, gradient norm 0.13–0.21.
- **Naive sequential:** every task ended at loss 0.030–0.065.
- **Multitask LoRA:** loss fell smoothly from 0.27 at 5k steps to 0.062 at 50k, with gradient norm ≤ 0.39 throughout.

The tables below use the reruns. The 1e-3 rows they replace scored 53.2 (r256), 46.6 (multitask) and 4.4 (naive sequential's final row).

**Ten-task results** (all ten envs; 25 episodes × 4 paired seeds; 10 actions per chunk):

| Row | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Specialists r32 (task identity given) | 82 | 83 | 84 | 97 | 69 | 82 | 78 | 85 | 100 | 99 | 85.9 |
| Specialists r64 | 90 | 74 | 86 | 94 | 69 | 87 | 84 | 86 | 96 | 92 | 85.8 |
| Specialists r128 | 91 | 72 | 88 | 98 | 69 | 75 | 76 | 86 | 96 | 94 | 84.5 |
| Specialists r256 (LR 3e-4) | 77 | 76 | 88 | 96 | 71 | 78 | 80 | 81 | 99 | 95 | 84.1 |
| **Memory, sequential (after all ten tasks)** | 70 | 76 | 79 | 77 | 74 | 81 | 77 | 86 | 98 | 99 | **81.7** |
| Joint fine-tune (all weights) | 73 | 59 | 86 | 91 | 74 | 76 | 61 | 85 | 97 | 94 | 79.6 |
| Multitask LoRA r128 (LR 1e-4) | 67 | 73 | 80 | 90 | 67 | 52 | 49 | 78 | 92 | 74 | 72.2 |
| Zero-shot | 0 | 0 | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 66 | 7.2 |

**Five-task comparison** (envs 4, 6, 9, 2, 7):

| Row | e4 | e6 | e9 | e2 | e7 | Mean |
|---|---|---|---|---|---|---|
| Memory, five-task run (final row) | 73 | 72 | 83 | 86 | 91 | 81.0 |
| Memory, ten-task run (row 5) | 72 | 76 | 82 | 83 | 90 | 80.6 |
| Specialists r32 | 82 | 83 | 84 | 97 | 69 | 83.0 |
| Specialists r128 | 91 | 72 | 88 | 98 | 69 | 83.6 |
| Joint fine-tune | 73 | 59 | 86 | 91 | 74 | 76.6 |
| Naive sequential r128 (LR 3e-4, final row) | 0 | 0 | 0 | 0 | 53 | 10.6 |
| Zero-shot | 0 | 0 | 0 | 2 | 0 | 0.4 |

**Ten-task memory triangle** (bold = just trained; last column = mean over earlier tasks):

| After task | e4 | e6 | e9 | e2 | e7 | e0 | e8 | e1 | e3 | e5 | Earlier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **90** | | | | | | | | | | — |
| 2 | 84 | **79** | | | | | | | | | 84.0 |
| 3 | 89 | 79 | **83** | | | | | | | | 84.0 |
| 4 | 80 | 78 | 89 | **83** | | | | | | | 82.3 |
| 5 | 72 | 76 | 82 | 83 | **90** | | | | | | 78.2 |
| 6 | 72 | 72 | 79 | 84 | 86 | **77** | | | | | 78.6 |
| 7 | 76 | 81 | 81 | 79 | 88 | 85 | **73** | | | | 81.7 |
| 8 | 73 | 73 | 76 | 73 | 68 | 80 | 69 | **93** | | | 73.1 |
| 9 | 75 | 66 | 80 | 70 | 72 | 80 | 78 | 87 | **98** | | 76.0 |
| 10 | 70 | 76 | 79 | 77 | 74 | 81 | 77 | 86 | 98 | **99** | 79.8 |

**Naive sequential triangle** (r128, LR 3e-4; bold = just trained; last column = mean over earlier tasks):

| After task | e4 | e6 | e9 | e2 | e7 | Earlier |
|---|---|---|---|---|---|---|
| 1 | **77** | | | | | — |
| 2 | 0 | **76** | | | | 0.0 |
| 3 | 0 | 0 | **86** | | | 0.0 |
| 4 | 0 | 0 | 0 | **94** | | 0.0 |
| 5 | 0 | 0 | 0 | 0 | **53** | 0.0 |

Just-trained mean 77.2. On the same five tasks: memory 85.0 (both runs), specialists r128 83.6. At LR 1e-3 the same cells were 0 and the just-trained mean was 62.2.

**Replicate.** The ten-task run's first five rows repeat the five-task run (same A-phase, seed and settings) to within 3.3 points per cell on average (largest difference 8); row 5 is 80.6 against 81.0. Take ~3 points per cell as the triangle's run-to-run noise.

**Reading against the pre-registered reads.**
- **Naive sequential: confirmed.** Every earlier task scores 0 from block 2 on, as on pi05. The final row is 10.6, against the memory's 81.0.
  - Every task was learned (just-trained mean 77.2, up from 62.2 at 1e-3), so the forgetting is not an optimisation failure.
  - The just-trained scores sit below the memory's on average, most on e7, the last task: 53, against 91 / 90 (memory) and 69 (specialists r128).
- **Specialists do not rise with rank:** r32 85.9, r64 85.8, r128 84.5, r256 84.1 (r256 at LR 3e-4, the others at 1e-3). With a flat ladder the memory has no equivalent specialist rank. Its scores just after training (mean 86.5) match the specialists; its final row (81.7) is 2–4 points below them.
- **Not confirmed: the joint-training rows are the lowest of the trained rows.**
  - The joint fine-tune (79.6) is below the specialists and below the memory's final row, with weak cells on e6 (59) and e8 (61). Its recipe is stage 1's, carried over from pi05's joint row (50k steps at batch 32, SmolVLA's default optimizer: AdamW 1e-4 → 2.5e-6), and may not be optimal for this fine-tune.
  - Multitask LoRA (72.2) is 7.4 points below the joint fine-tune, not just below it (pi05: 2.6 below). Its weakest cells are e0 (52) and e8 (49), and its training loss ended at 0.062, against the joint fine-tune's 0.04.
  - Order on SmolVLA: specialists 84–86, memory's final row 81.7, joint fine-tune 79.6, multitask LoRA 72.2.
- **The leak on the oldest tasks does not grow with more tasks.** Earlier tasks lose −5.0 points on average at five tasks and −5.3 at ten. It is uneven: e4 −20 and e7 −16 at ten tasks, the rest within ±7. e4's drop replicates across the two runs (−9 and −18 at row 5); e6's five-task drop does not (−11 and −3).

---
## Entry 5 - 19 Nov 26 (sixty new tasks in a random order: the first scaled sequence)

**Why.** Paper 1 item 1: scale the sequence from 10 to 30–60 tasks and measure acquisition, retention and the protected-slot statistics as functions of the number of tasks K. On pi05, unrelated cores overlap at ~0.024 per pair, so the share of a new task's reads outside every earlier core should fall as ~(1 − 0.024)^(K−1): 0.80 at K = 10, 0.30 at K = 50. The point where the write rule stops finding unprotected slots has never been measured.

**Tasks.** libero_ext: 60 new tasks in 27 scenes built in the LIBERO simulator, 30 single-step and 30 two-step, 50 scripted-expert demos each (`dataset_gen.md` §2–3b). The libero_90 base scores 1.0% on them zero-shot (only "turn on the faucet" 2/5 and "lift the alphabet soup" 1/5 above zero), so nearly all later success is adaptation.

**Run `cell_s2_ext60_rand1`.** The ten-task run's settings (`cell_s2_seq10`) except:
- **Same starting point:** cell_s2_seq5's A-phase, i.e. the separation-2 router warm-up with its values filled on libero_90.
- **Data:** libero_ext; 5,000 steps per task, 300,000 in all.
- **Order:** a random permutation of the 60 tasks, `random.Random(1).shuffle(list(range(60)))`, recorded with the task names in `cluster/jobs/smolvla/configs/ext60_order_rand1.json`. It happens to hold 15 two-step tasks in each half.
- **In-run loss matrix:** after every task, the paired-noise flow-matching MSE on every task seen so far (16 batches of 16 per task, frames and noise seeded per task). This is the pi05 instrument, computed on the live policy rather than reloaded checkpoints.
- **Logging:** every 100 steps; per-task slot reads and updates (`memory_by_task/`); and the protection store and IDF at every boundary (`sequential_state.pt`). Every boundary checkpoint is kept, for rollout evaluation afterwards.

Rollout evaluation follows training, from the boundary checkpoints, per the Paper 1 plan: the just-trained task at every boundary, and the full retention row every fifth task (25 episodes × 4 paired seeds, 10 actions per chunk).

**Pre-registered reads.**
- **Acquisition.** If the memory has room for 60 tasks, the just-trained task's success stays near the ten-task diagonal (86.5) throughout; its own-task loss right after training stays flat in K. A fall in either with K, at the fixed 5,000-step budget, is the saturation signal item 2 calibrates.
- **Retention.** On libero_10, earlier tasks lost ~5 points whether 5 or 10 tasks followed. If that holds, the final row's mean over earlier tasks sits within ~5 points of their just-trained mean. In loss terms, drift (own-task loss from just-trained to final) grows with the number of tasks trained after (pi05 ten-task mean +28.5%).
- **Protection statistics** (computed afterwards from `memory_by_task/` and `sequential_state.pt`):
  - the protected fraction rises with K;
  - ω_K, the share of task K's read mass outside every earlier core, falls roughly as (1 − overlap)^(K−1).
  The K at which acquisition starts to fall, set against ω_K and the protected fraction there, is the first estimate of the detector threshold.
- **Task order.** A single random order cannot separate the effect of K from that of particular tasks. The scenes share fixtures, so related tasks are scattered through the sequence. A grouped order is the planned second run.
