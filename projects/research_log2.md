# Research Log 2 - VLA Memory on SmolVLA

## Executive summary (state at 6 Nov 26)

**Problem.** Learn new tasks one after another without forgetting earlier ones: frozen backbone, no task identity at inference, no replay, no per-task parameters.

**Method.** Sparse product-key memory modules sit beside selected MLPs (MLP(x) + Memory(x)) in the action expert and the VLM language model. Each slot value is a rank-2 LoRA transform of the hidden state; sequential adaptation updates only slot values. Three mechanisms keep tasks apart:
- **Geometry before content.** The backbone is fine-tuned on the pretraining suite (libero_90) without memory. A router warm-up then trains only keys and query projections (values pinned at zero) on routing losses: inter-task separation on the joint slot distribution, a small contrastive term, an expert router anchored to the pooled instruction hidden state, and a pooled VLM key for the state region. The routers then freeze and an A-phase fills the values on libero_90.
- **Stationary addressing.** Routing reads memory-free backbone features (a frozen pre-pass), so writing values never moves where a task reads.
- **Protected writes.** Each step writes only the top-t slots by TF-IDF score (t = 3072 per site); earlier tasks' core slots score zero (protection β = 4, rank mode, corefrac).

**Paper cell** (pi05: PaliGemma-2B VLM + 300M action expert). Expert sites [4,6,8,10,14,16], VLM sites [5,7,9,11,13,15]; adjacent pairs share a table except the two deepest expert sites: 12 sites, 7 tables of 65,536 slots, ~2.7B value parameters. Sequential: 5,000 steps per task, value LR 2e-3 → 2e-4. Instrument: 25 episodes × 4 paired seeds per task, a retention triangle and a paired-noise loss matrix.

**Headline results** (ten libero_10 tasks in sequence, 4-seed means; drift = own-task loss from just-trained to final):

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

The SmolVLA prototype (E0–E17) used the earlier joint protocol on four tasks (best 57.5%, not comparable); the staged protocol has never run on SmolVLA. History: `research_log.md`, summarised in `summary_research_log.md`.

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
- Layout: Entry 1's (11 sites, 7 tables, ~1.5B value parameters), chosen without the compass and geometry probes, which have not run.
- Warm-up at separation 2; otherwise the pi05 cell: contrastive 0.05, expert anchor 0.4, VLM pool (1, 0.5), pre-pass; A-phase 10k on libero_90; 5,000 steps per task; value LR 2e-3 → 2e-4; top-t 3072; protection β4, rank mode, corefrac.
- Five tasks (front-5: envs 4, 6, 9, 2, 7), as pi05's development runs.
- Instrument: retention triangle at every task boundary, 25 episodes × 4 paired seeds per cell, 10 actions per chunk. The audit covers all ten tasks; its solo sites (E14, V13) test whether the reduced-layout certificates carry over.

The question is whether the staged recipe reproduces pi05's shape on SmolVLA before any recalibration. On pi05, earlier tasks held their level through the sequence (prior-task mean 69.8 at block 5 against a diagonal of 65.6), where naive sequential training falls to zero. The main risk (Entry 1) is write-mask saturation at the VLM sites, which see one state token per sample. Absolute success needs SmolVLA's own bars, still to come.

**Zero-shot floor.** The libero_90 base on all ten libero_10 envs, same instrument (pi05: 10.6, single seed, 50-action chunks). It shows how much of any later score is adaptation.
