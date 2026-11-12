# Research Log 2 - VLA Memory on SmolVLA

## Executive summary (state at 6 Nov 26)

**Problem.** Teach a vision-language-action policy new tasks one after another without forgetting the earlier ones: frozen backbone, no task identity at inference, no replay, no per-task parameters.

**Method.** Sparse product-key memory modules sit beside selected MLPs (MLP(x) + Memory(x)) in both the action expert and the VLM language model. Each slot value is a rank-2 LoRA transform of the hidden state, and only slot values are updated during sequential adaptation. Three mechanisms keep tasks apart:
- **Geometry before content.** The backbone is first fine-tuned on the pretraining suite (libero_90) without memory. A router warm-up then trains only keys and query projections (values pinned at zero) on routing losses: inter-task separation on the joint slot distribution, a small contrastive term, an expert router anchored to the pooled instruction hidden state, and a pooled VLM key for the state region. The routers then freeze and an A-phase fills the values on libero_90.
- **Stationary addressing.** Routing reads memory-free backbone features (a frozen pre-pass), so writing values never moves where a task reads.
- **Protected writes.** Each step writes only the top-t slots by TF-IDF score (t = 3072 per site), and earlier tasks' core slots score zero (protection β = 4, rank mode, corefrac).

**Paper cell** (pi05: PaliGemma-2B VLM + 300M action expert). Expert sites [4,6,8,10,14,16] and VLM sites [5,7,9,11,13,15]; adjacent pairs share one table except the two deepest expert sites. That gives 12 sites, 7 tables of 65,536 slots, ~2.7B value parameters. Sequential stage: 5,000 steps per task, value LR 2e-3 → 2e-4. Instrument: 25 episodes × 4 paired seeds per task, plus a retention triangle and a paired-noise loss matrix.

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

- **Retention.** Earlier tasks still average 62.2 after all ten blocks (naive: 0.0). Tasks 1–5 score 64.6 and tasks 6–10 score 65.6.
- **Order and length.** The reversed task order gives 62.4, inside the ±3-point replicate band. On five tasks the result is 65.2 with drift +3.9%.
- **Ablations** (five tasks, control 66.0):
  - Without stationary addressing: 64.4, but retention change Δ_final −4.0 against the control's +0.4, all of it on the oldest task.
  - Without protection: 59.8, with the oldest task at 32 against 53.
  - Routers and values trained jointly at matched LR: 59.0, lost in acquiring the newest task (37 against 51).
  - Removing the separation loss fails the background-overlap gate at all six expert sites; removing the contrastive loss changes little.
- **Real robot** (WidowX AI, five tasks, function space only): drift +2.7% against +2356% for naive LoRA, and lower own-task loss than every r64 specialist. No robot rollouts exist.
- **Limits.** The basket family (envs 7, 0, 1: one scene, near-identical tasks) accounts for the whole deficit against the specialists, and env 7 is the worst drift cell. Overall the method matches roughly an r45 specialist and sits 14.6 below the joint fine-tune.

The prototype ran on SmolVLA (E0–E17) under the earlier joint protocol, on four tasks (best 57.5%, not comparable). The staged protocol above was developed on pi05 and has never been run on SmolVLA. Full history: `research_log.md`, summarised in `summary_research_log.md`.

---
## Entry 1 - 6 Nov 26 (SmolVLA reproduction of the paper recipe, and the baselines it needs)

**Why SmolVLA.** Compute is now limited to a shared cluster. SmolVLA (~450M parameters) costs roughly a tenth of pi05 per step, which buys iteration and training-seed replicates (pi05 had a single replicate pair, hence the ±3 band). The aim is to reproduce the *shape* of the pi05 result under the same protocol: libero_90 → libero_10, same task order, same instrument. That means sequential memory at least matching the per-task specialists, far above naive sequential training, with small drift. Absolute numbers are not expected to transfer, so every comparison stays within the SmolVLA substrate.

**Base (stage 1).** All weights, end to end, mirroring the pi05 stage 1:
- smolvla_base on libero_90, no memory.
- 50k steps at batch 32, warmup 4k with cosine decay over 50k.
- VLM and vision encoder trained as well as the expert.
- Optimizer: SmolVLA's own default (AdamW 1e-4 → 2.5e-6), just as the pi05 run used pi05's (2.5e-5). With bf16 weights, most 2.5e-5 updates would fall below bf16 resolution.
- Dtypes follow HF's SmolVLA recipe (bf16 VLM and expert, no autocast).
- Two cameras, HF's LIBERO convention. pi05 filled a third camera slot with an empty image.

Why all weights: the memory routes on, and the VLM memory adapts, VLM features, and on pi05 those features were LIBERO-adapted. The libero_90 substrate was also worth +10.6 even to a full fine-tune (E60 add-9).

Both datasets were rebuilt from public sources after the pi05-era copies were lost, and checked against the simulator by open-loop replay.

**Evaluation.** Same instrument as before (25 episodes × 4 paired seeds, retention triangle, paired-noise loss matrix). SmolVLA executes 10 actions per predicted chunk (`n_action_steps=10`).
- Evidence: the SmolVLA paper's own LIBERO ablation (its Table 13) gave 80.3 / 82.8 / 70.8 / 51.8 for executing 1 / 10 / 30 / 50 actions before re-observing.
- Full 50-action chunks would push every row toward the floor. Replanning every step, the paper's stated simulation protocol, buys nothing over 10.
- The pi05 rows used full 50-action chunks (pi05's default). That is one more reason the two models' absolute numbers are not comparable.

**Baselines.** All start from the SmolVLA libero_90 base and use the same instrument:
- Zero-shot floor on libero_10 (pi05: 10.6).
- Joint all-weights fine-tune on all ten tasks (the ceiling).
- Ten per-task LoRA specialists.
- Multitask LoRA: one adapter on all ten tasks, 50k steps.
- Naive sequential LoRA.

They run first because they set the bars, SmolVLA's per-task difficulty, and the seed noise that later arms are scored against.

Still open: the LoRA target set and rank.
- HF's default LoRA for SmolVLA adapts only the expert's q/v projections and the action/state projections; the pi05 rows adapted attention and MLP in both towers.
- The pi05 rank rule picked r512, the smallest power of two above the per-site bottleneck (2 × heads × knn = 288). On SmolVLA's 720- and 960-wide matrices that is close to full rank.

**Memory port.** Every memory option carries over. The architectural differences that matter:
- SmolVLA's VLM is SmolVLM2 cut to 16 layers (width 960), paired with a 16-layer expert (width 720). Attention alternates between joint self-attention and expert-to-VLM cross-attention.
- The robot state is a single continuous token in the prefix, where pi05 had dozens of state-as-text tokens. The VLM memory therefore covers the instruction tokens plus one state token.
- Nothing reads the last VLM layer's output, so it can carry no memory.

First layout, pending the probes below: expert [4,6,8,10,12,14] with (4,6) and (8,10) shared, and VLM [5,7,9,11,13] with (5,7) and (9,11) shared. That is pi05's relative depths without the dead layer: 11 sites, 7 tables.

**What might not transfer.**
1. With one state token, the VLM sites see far fewer distinct reads per batch. That risks write-mask saturation (E65 add-14: protection is bypassed once top-t reaches the read-set size). It also leaves less VLM capacity per sample, yet on pi05 the VLM side was the largest fit lever (E44).
2. The routing-loss weights, gate bands, top-t, value LR and bank size were all calibrated on pi05. The real-robot port needed four warm-up arms (E65).
3. A 7-table bank is ~1.5B value parameters on a 450M model, against ~0.7× the model on pi05. The "you just added parameters" objection gets louder, and bank size may need revisiting.

**Order.**
1. Base, zero-shot floor and baselines.
2. Alongside the baselines: an E44-style compass (LoRA restricted to the expert, to the VLM, or to both, on e4, e9 and e7) and a per-layer geometry probe of the frozen base. These choose the layout.
3. The paper cell on SmolVLA: warm-up and held-out audit first, then A-phase and the five-task sequence, with triangle and loss matrix.
4. Recalibrate whatever fails, one change per arm, cheapest instrument first.
5. Confirm on ten tasks, in the reversed order, with replicate training seeds.

---
## Entry 2 - 7 Nov 26 (router warm-up sweeps on the SmolVLA libero_90 base: separation and expert anchor)

**Why.** The warm-up recipe was calibrated on pi05, and the router's geometry depends on how crowded the frozen backbone's features are, which changes with the backbone. On pi05 two knobs decided that geometry:
- **Separation.** Removing the inter-task separation loss failed the background-overlap gate at all six expert sites (E68 C1).
- **Expert anchor weight.** It had a narrow window: 0.5 over-compacted, 0.35 failed, 0.40 passed (E53–E54). It is a pure mix ratio between the pooled instruction hidden state and the token's own features (both normalised), so its best value depends on how crowded the expert features are relative to the instruction features.

The contrastive term is not swept: E68 C2 showed it is redundant with pooled VLM routing.

**Design.**
- **Starting checkpoint:** the SmolVLA libero_90 all-weights base (Entry 1).
- **Reduced layout:** three solo sites per tower, expert [4, 8, 14] and VLM [5, 9, 13], with no table sharing; VLM layer 15 is excluded. With frozen-base routing each site's router trains on its own losses from memory-free features, so per-site certificates should carry over to fuller layouts.
- **Warm-up:** as the paper cell (10k steps, batch 32, router LR 1e-4, values pinned at zero, contrastive 0.05, VLM pool (1, 0.5), pre-pass).
- **Audit:** as the pi05 audit (all ten libero_10 tasks, 400 steps × batch 8, value LR 1e-12).
- **Arms:** a reference at separation 8 and anchor 0.4 (the paper values), plus four single changes from it: separation 2 and 32, anchor 0.2 and 0.6.
- **Reads per site:** famIoU, bgIoU, core50, min-task effnum. The pi05 bands (expert bgIoU ≤ 0.10, mean core50 ≥ 400, min effnum ≥ 300; VLM min effnum ≥ 150) are a guide, not a gate.

**Pre-registered reads.**
- **Separation:** overlap (bgIoU, famIoU) falls as separation rises, with capacity (core50, effnum) the likely price at 32. The working value is the lowest overlap that keeps capacity above the floors.
- **Anchor** (expert sites only; the VLM settings are the same in all three arms): pi05 showed a U in famIoU, with capacity falling as the anchor rises. Where the bottom of the U sits on SmolVLA says how crowded its expert features are relative to its instruction features.
- **What each outcome means for the plan:**
  - If the reference passes, the best value on each axis goes to the full layout.
  - If capacity is short in every arm, the next sweep follows the real-robot direction: contrastive up, separation down.
  - If overlap is high in every arm, separation goes higher.

**Correction (11 Nov 26): the VLM half of these results is void.** On the SmolVLA span path, the warm-up's routing losses and queue had two bugs.
- **Truncated span.** Every sample was cut to the batch's shortest valid span. SmolVLA's state token ends each variable-length span, so the losses saw only ~47% of span positions and the state token in ~6% of samples. What remained was mostly the shared leading words of the instructions.
- **Wiped queue.** The routing queue was re-created empty whenever the batch's longest span changed, several times per 30 steps.

The VLM routers were therefore trained on almost no task signal. That explains their insensitivity to separation below. The expert sites were unaffected (no token mask, fixed 50 rows per sample), so the expert results stand. Fixed in commit 5f59e3ef; the separation arms are being rerun.

**Results** (8 Nov 26). Each cell is famIoU / bgIoU / mean core50 / min-task effnum, from the held-out audit on all ten libero_10 tasks.

Expert sites:

| Arm | E4 | E8 | E14 |
|---|---|---|---|
| separation 2 | 0.106 / 0.026 / 554 / 563 | 0.094 / 0.025 / 524 / 685 | 0.138 / 0.055 / 1130 / 895 |
| **reference** (separation 8, anchor 0.4) | 0.088 / 0.018 / 400 / 423 | 0.075 / 0.018 / 393 / 529 | 0.122 / 0.041 / 817 / 787 |
| separation 32 | 0.076 / 0.012 / 299 / 332 | 0.069 / 0.012 / 299 / 407 | 0.105 / 0.029 / 579 / 716 |
| anchor 0.2 | 0.155 / 0.058 / 1246 / 805 | 0.106 / 0.042 / 886 / 868 | 0.145 / 0.073 / 1538 / 1027 |
| anchor 0.6 | 0.071 / 0.005 / 136 / 282 | 0.067 / 0.007 / 171 / 272 | 0.105 / 0.023 / 453 / 684 |

VLM sites (the anchor arms match the reference to within 0.003 and about 30 slots, as their VLM settings are the same):

| Arm | V5 | V9 | V13 |
|---|---|---|---|
| separation 2 | 0.212 / 0.050 / 777 / 1320 | 0.194 / 0.061 / 1113 / 1815 | 0.215 / 0.091 / 1770 / 2693 |
| reference | 0.213 / 0.049 / 745 / 1263 | 0.191 / 0.060 / 1101 / 1693 | 0.217 / 0.088 / 1720 / 2593 |
| separation 32 | 0.212 / 0.048 / 733 / 1238 | 0.192 / 0.057 / 1045 / 1366 | 0.216 / 0.085 / 1632 / 2243 |

Against the pi05 bands:
- **Separation 2 and anchor 0.2:** pass every clause.
- **Reference:** misses by a hair. E8's mean core50 is 393 against the floor of 400, and E4 sits exactly on it.
- **Separation 32:** fails the core50 floor at E4 and E8 (299).
- **Anchor 0.6:** fails both capacity clauses at E4 and E8 (core50 136 / 171, min effnum 282 / 272).
- **VLM sites:** pass everywhere.

**Reading.**
- **Overlap is not the constraint on SmolVLA; capacity is.** Every expert site sits well inside the overlap band in every arm (worst bgIoU 0.073). The arms differ only in whether the cores stay large enough. The paper's settings sit exactly on the capacity floor.
- **Separation behaved as pre-registered.** Overlap falls monotonically as separation rises (E4 bgIoU 0.026 → 0.018 → 0.012), and the cores shrink with it (E4 core50 554 → 400 → 299).
- **Anchor: no U, contrary to the pre-registered read.** Over 0.2–0.6, overlap, family overlap and core size all fall together at every expert site (E4 famIoU 0.155 → 0.088 → 0.071; core50 1246 → 400 → 136). Capacity collapses at 0.6 before any famIoU minimum appears, as 0.5 over-compacted on pi05.
- **The VLM routers barely respond to separation** between 2 and 32. This is unlike pi05, where removing separation doubled VLM family overlap (E68 C1). The VLM tower has more overlap than pi05's paper cell (famIoU 0.19–0.22 against 0.08–0.15) and much broader footprints (min effnum 1,238–2,693 against 250–750), yet it is still within the bands. Why it is insensitive is open.
- **Outcome against the pre-registered plan:**
  - The reference misses narrowly, but capacity is not short everywhere and overlap is never high, so neither fallback applies.
  - Two settings pass: separation 2 (with anchor 0.4) and anchor 0.2 (with separation 8). Both buy capacity with overlap, which SmolVLA has margin to spare.
  - Ranked background-overlap first (the pi05 rule, E56), separation 2 leads: E4/E8/E14 bgIoU 0.026/0.025/0.055 against 0.058/0.042/0.073. Ranked on capacity, anchor 0.2 leads, with cores 1.6–2.3× larger.

### Entry 2 addendum (12 Nov 26): separation arms rerun with the VLM loss fix

**Expert sites replicate the first sweep.** The fix does not touch expert routing, and all three arms reproduce their first-sweep expert certificates:
- famIoU and bgIoU are identical to three decimals.
- Mean core50 is within 1%, e.g. the reference E14 is 822 against 817.
- Min effnum is within 2%.

That is GPU nondeterminism only, and every expert reading above stands.

**VLM sites, first sweep (losses on the truncated span) → rerun (every valid span row).** Each cell is famIoU / bgIoU / mean core50 / min-task effnum.

| Separation | V5 | V9 | V13 |
|---|---|---|---|
| 2, first | 0.212 / 0.050 / 777 / 1320 | 0.194 / 0.061 / 1113 / 1815 | 0.215 / 0.091 / 1770 / 2693 |
| 2, rerun | 0.204 / 0.043 / 827 / 843 | 0.189 / 0.053 / 1084 / 1041 | 0.202 / 0.080 / 1832 / 2168 |
| 8, first | 0.213 / 0.049 / 745 / 1263 | 0.191 / 0.060 / 1101 / 1693 | 0.217 / 0.088 / 1720 / 2593 |
| 8, rerun | 0.206 / 0.042 / 781 / 546 | 0.186 / 0.048 / 933 / 606 | 0.200 / 0.070 / 1508 / 1441 |
| 32, first | 0.212 / 0.048 / 733 / 1238 | 0.192 / 0.057 / 1045 / 1366 | 0.216 / 0.085 / 1632 / 2243 |
| 32, rerun | 0.191 / 0.039 / 705 / 452 | 0.188 / 0.042 / 776 / 409 | 0.203 / 0.061 / 1177 / 913 |

**Reading.**
- **The VLM routers now respond to separation.** Overlap and footprint size fall together as separation rises, at every site (V13 bgIoU 0.080 → 0.070 → 0.061; min effnum 2168 → 1441 → 913).
- **The fix alone tightens and separates the VLM routing.** At matched separation, min effnum drops 1.5–3× and bgIoU falls 10–30%.
  - At V5 overlap falls while the mean core grows (reference: bgIoU 0.049 → 0.042, core50 745 → 781). That is a genuine separation gain, not just smaller footprints.
  - The per-task core ranges widen (reference V5 424–1204 → 209–1559), so footprints now differ by task.
- **Within the sweep, VLM separation behaves like the expert sites.** bgIoU ÷ core50 stays roughly constant across separation weights (V5 5.2, 5.4, 5.5 ×10⁻⁵), so the weight mostly slides along an overlap-versus-footprint curve.
- **Family overlap does not move.** famIoU stays at 0.19–0.21 in every arm, against 0.08–0.145 on the pi05 paper cell. Separation from 2 to 32 does not reduce it. The likely cause is that instruction tokens are routed one at a time and dominate the VLM reads, while the basket tasks' instructions share most of their words. That last point is untested.
- **Against the pi05 VLM bands:**
  - **bgIoU:** now inside them at separation 8 and 32 (0.039–0.070 against 0.023–0.074). Separation 2's V13 is just above (0.080).
  - **Min effnum:** inside them at V5/V9 for separation 8 and 32 (409–606 against 250–750), and above at V13. Every arm clears the VLM floor of 150.
  - **Mean core50:** still broader than pi05's at V9/V13 (776–1832 against 188–842).

---
## Entry 3 - 12 Nov 26 (separation 2 carried forward; a contrastive sweep; the first paper-cell chain on SmolVLA; the zero-shot floor)

**Decision: separation 2, with anchor 0.4.** One separation weight serves both towers.
- **Expert:** S=2 is the only separation weight that clears the core50 floor at every site. The reference's E8 sits at 393 and 394 in the two sweeps; S=32 drops to 299. Its overlap (bgIoU 0.026 / 0.025 / 0.055) is still below pi05's paper cell at the same depths (0.038 / 0.037 / 0.058). Ranked background-overlap first, it also beats the other passing arm, anchor 0.2, which has about twice the overlap.
- **VLM:** S=2 is the loosest of the three settings (min effnum 843–2168; V13 bgIoU 0.080), but every setting clears the VLM floor. A per-tower separation weight is not worth adding unless the VLM side causes trouble.

**Contrastive sweep.** On pi05 the contrastive weight moved capacity, but its direction flipped between regimes:
- **Joint training (E24–E25):** contrastive compacted footprints. Core50 was 2,643 / 1,752 / 1,162 at c 0.01 / 0.05 / 0.1, and c=0 sprawled.
- **Frozen base (E68 C2):** removing it shrank the cores (expert core50 713–2,019 → 432–1,446; VLM min effnum 250–750 → 158–276) and lowered overlap with them.
- **Real robot (E65):** the arm that cleared capacity had raised c from 0.3 to 1.0, together with three other changes.

That leaves one clean frozen-base measurement. Arms: c = 0, 0.2 and 1.0 at S=2, warm-up and audit on the reduced solo layout (`wu2_sep2` is the c=0.05 point).

Pre-registered reads:
- **The read that matters is overlap per unit of footprint.** On SmolVLA, separation and anchor both move along one curve: bgIoU grows in proportion to core50 (4–5 ×10⁻⁵ at every expert site in every arm). Footprints placed at random also overlap in proportion to their size, so these knobs resize footprints without placing them better.
  - If that ratio stays flat across c, c is a third copy of the same dial: keep 0.05 and close the axis.
  - If c lowers the ratio (bigger cores without proportionally more overlap, or less overlap without proportionally smaller cores), it moves the curve. That is worth a full chain, probably paired with a higher separation.
- **pi05's hint is mixed.** In E68 C2, removing c lowered the ratio by 15–36% at the expert sites and raised it by 10–47% at the VLM sites.
- **famIoU is not expected to move:** contrastive never separated the basket family on pi05 (E54 P2, E68 C2).
- **At c=1.0, watch for query collapse:** cores shrinking towards a handful of slots (E21).

**First paper-cell chain on SmolVLA.** The full staged chain:
- **Layout:** Entry 1's layout (expert [4,6,8,10,12,14] with (4,6) and (8,10) shared; VLM [5,7,9,11,13] with (5,7) and (9,11) shared). That is 11 sites and 7 tables, ~1.5B value parameters.
- **Warm-up:** at S=2.
- **Everything else as the pi05 cell:** contrastive 0.05, expert anchor 0.4, VLM pool (1, 0.5), pre-pass; A-phase 10k on libero_90; 5,000 steps per task; value LR 2e-3 → 2e-4; top-t 3072; protection β4, rank mode, corefrac.
- **Tasks:** five (front-5: envs 4, 6, 9, 2, 7), as pi05's development runs.
- **Instrument:** the retention triangle at every task boundary, 25 episodes × 4 paired seeds per cell, 10 actions executed per chunk.
- **Audit:** all ten tasks. Its solo sites (E14, V13) also test whether the reduced-layout certificates carry over to the full layout.

The question is whether the staged recipe reproduces pi05's shape on SmolVLA before any recalibration. On pi05, earlier tasks held their level through the sequence: prior-task mean 69.8 at block 5 against a diagonal of 65.6, where naive sequential training falls to zero. The main risk Entry 1 flagged is write-mask saturation at the VLM sites, which see one state token per sample. Absolute success needs SmolVLA's own bars, which are still to come.

**Zero-shot floor.** The libero_90 base on all ten libero_10 envs, same instrument (pi05: 10.6, single seed, 50-action chunks). It shows how much of any later score is adaptation.
