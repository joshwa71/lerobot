#!/bin/bash
# E54 GRADUATION — CELL B: spread substrate + anchored/FiLM-free sep8 router + corefrac
# =====================================================================================
# B is one of the two routers that certified in the E54 six-probe batch (P3 the other).
# Certificate (audit_heldout_jointwarm_layermax_A_anchor040_sep8_nofilm_e2468_v10121416_10k):
#   expert famIoU 0.160/0.170/0.160/0.192 (L2/4/6/8; one grace at L8 <= 0.20)
#   expert bg     0.025-0.037   <- the headline number (see attribution below)
#   expert core50 585-732       <- absmax band; below the OLD 800 floor, above the
#                                  relaxed >=400 tripwire (floor falsified in E54 by the
#                                  53.6 frontier, whose cores are 425-648 at ALL layers)
#   VLM famIoU 0.132-0.154 PASS, min-eff 381-914
# Router: expert text-anchor (pooled LM instruction hidden) at w=0.40, FiLM OFF
# (lang_to_query=false) + sep 8.0 + c 0.05. No mpnet, no film_mlp — the anchor is the
# ONLY language conditioning, which E54-P4 proved is load-bearing at warm-up time
# (zero language => famIoU 0.50-0.73 sprawl).
#
# WHAT THIS RUN IS FOR (the attribution question)
# -----------------------------------------------
# The 53.6 frontier (absmax: 13 banks / 5.37B / anchor w=0.5 / corefrac) confounds TWO
# things: bank count and shoulder cleanliness (expert bg 0.026 vs compact's 0.080).
# B carries the frontier's bg profile (0.03) on an 8-bank / 3.2B spread substrate whose
# corefrac number is already measured with the PLAIN router (47.6) — so this is a clean
# single-delta router cell.
#   >= ~52  -> shoulder cleanliness was the active ingredient; headline config becomes
#              spread at 3.2B and absmax demotes to capacity-scaling evidence.
#   ~48-50  -> the 13 banks did it; absmax's size problem is real and unavoidable.
# Mechanism behind the bet: corefrac already zeroes the CORE damage channel (E53: 0
# events into prior cores at all 40 pair x module cells, 0% core drift), and E53 located
# corefrac's residual -5.6 give-back in SHOULDER relocation — evicted core write pressure
# landing on shoulders. B's bg is a 3x shoulder cleanup over the plain spread router
# (0.094-0.119 -> 0.025-0.037), i.e. aimed at exactly the one channel corefrac leaves open.
#
# Comparators (50-ep finals): spread+corefrac (plain router, THE single-delta twin) 47.6;
# spread+peak 41.2; compact+corefrac 51.6; absmax+anchor05+corefrac 53.6 (frontier).
#
# PRE-REGISTERED READS
# --------------------
#   - core events = 0 at all 8 modules; MSE matrix flat (<= ~+5% diag drift); function
#     give-back <= ~5%/task (the corefrac signature — E53: +3.6/+2.0/+4.4/+2.8/0.0%).
#   - BEAT 47.6 = the router delta pays on spread at all;
#     BEAT 51.6 = spread takes the frontier from compact;
#     BEAT 53.6 = frontier at 3.2B, i.e. the headline config for the paper.
#   - e7 IS THE DECISION CELL and corefrac cannot help it (last task, zero exposure =>
#     pure fit/conversion). Spread-peak rolled 20 at the best e7 function ever measured;
#     compact-corefrac 36; frontier 28; multitask-LoRA 48. e7 ~20 again => spread has a
#     real conversion deficit that caps it regardless of retention, and that reads on the
#     SUBSTRATE, not the router or the protection.
#   - CAPACITY TRIPWIRE (E54 relaxed gate, since core50 585-732 sits in the band the old
#     >=800 floor condemned): watch q_intra <= ~0.93, per-batch effnum, and footprint
#     dispersion <= 2x median for the E21/E22c constant-palette pathology. The 53.6 run
#     says this band is fine; it is not yet says-so at 8 banks.
#   - Shoulder watch: E2 was the leakiest module of the spread family (bleeds to 34% at
#     peak-norm). B's bg cleanup should show up there first if it shows up anywhere.
#
# Config = the E53 arm-2 spread+corefrac recipe VERBATIM (lr 2e-3 -> 2e-4, top_t 3072,
# beta4 rank-mode + corefrac, 5x5000 steps, bs16 x accum2, 20-ep intermediates + 50-ep
# final). The ONLY delta vs the 47.6 run is the warm-up checkpoint = B's router.
# Stage A runs here (no A-phase exists for this router yet), ~3.5h, then ~12-14h seq.
#
# NOTE (preemptible VM): stage-level skip guards make a relaunch of this wrapper
# idempotent — a completed A-phase is skipped, a completed sequential is skipped. There
# is no WITHIN-stage resume, so a preemption costs the current stage's progress (<= ~3.5h
# in A, <= one 5k block in the sequential). The partial-A guard below keeps a relaunch
# from silently downgrading the batch-size rung.
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s surface as bogus "vocabulary corrupted" tokenizer errors; all assets local

export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export GRAD_TAG=layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k
# Rung 1 == the byte-identical arm-2 config (bs16 x acc2, no grad-ckpt, PROVEN on this
# substrate by the 47.6 run); rungs 2/3 are preemption/VRAM insurance only.
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

# Partial-A guard: if a previous attempt died mid-A-phase (save_freq=10000 => the only
# save is at the end, so no checkpoint exists), wipe the stub. Without this, the common
# body's rung-1 relaunch hits a non-empty output dir, fails, and the ladder silently
# demotes to bs16 x accum2.
A_OUT_GUARD=/home/josh/lerobot/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
