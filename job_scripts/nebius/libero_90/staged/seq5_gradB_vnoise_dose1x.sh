#!/bin/bash
# E57 ARM 1 — value-input noise, FULL calibrated dose, on the B config (53.2 baseline)
# =====================================================================================
# The off-trail campaign (E57) located the specialist gap in the VALUE content's
# competence radius: retrieval healthy off-manifold (self-written mass 0.77-0.83 at all
# distances), but B's function diverges from the (rollout-validated) specialist's
# exactly on the far excursion states its failures visit. This arm trains the values to
# express the demo action from a NEIGHBORHOOD of hidden states: Bernoulli-masked
# Gaussian noise on the x consumed by the LoRA slot transforms, training-only,
# value-path-only (router/gate read the frozen branch; swilu + plain MLP keep clean x).
#
# DOSE (calibrated, probe_value_input_calib.py on the e56 harvest bank — obs-driven
# first-denoise-step displacement, mid excursion band, variance-matched at p=0.25):
#   p=0.25, per-row amp ~ U[0.5, 1.5] (the near-to-far band spread)
#   expert L2/4/6/8   sigma_rel = 0.1 / 0.3 / 0.75 / 1.05   (measured ratios x2)
#   vlm   L10/12/14/16 sigma_rel = 1.15 / 1.4 / 1.65 / 1.75
# Known approximation, eyes open: expert displacement is STRUCTURED (top-10 SVD dirs
# carry ~70-80% of energy); per-dim independent noise is isotropic. v1 bets on generic
# flatness; the structured (subspace-projected) variant is the measured fallback.
#
# Config otherwise = the B graduation VERBATIM (spread substrate, anchor040+sep8/nofilm
# router, corefrac, lr 2e-3->2e-4, top_t 3072, 5x5000, bs16xacc2, 50-ep final), reusing
# B's existing A-checkpoint (stage-A skip guard). Single delta = the noise flags.
#
# PRE-REGISTERED READS
#   - Harvest-bank pre-screen (run_e57_vnoise_queue.sh rescore stage, ~10 min): READ-1
#     D-vs-distance against spec_e7 on the SAME e56 states. PASS = far-region (Q4) D
#     shrinks vs B's 0.38/0.41, ESPECIALLY on spec-success states (where good behavior
#     is validated); anchor row must stay ~0.032-0.036 (on-demo fit preserved).
#   - Block-min mse_loss <= ~1.10x B's per task (the fit-cost guardrail).
#   - MSE forgetting matrix flat (corefrac signature <= ~+5%) — noise must not
#     destabilize the stationary-era retention.
#   - Only a pre-screen PASS earns weight on the 50-ep final (which runs anyway as part
#     of the chain): e7 > 20 = the conversion bet pays; e4/e9 watched.
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export GRAD_TAG=layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise1x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"
# E57 noise flags (no spaces — expanded unquoted in the common body's seq stage)
export SEQ_EXTRA_ARGS="--policy.memory_layer.value_input_noise_p=0.25 --policy.memory_layer.value_input_noise_sigma=[0.1,0.3,0.75,1.05] --policy.memory_layer.vlm_value_input_noise_sigma=[1.15,1.4,1.65,1.75] --policy.memory_layer.value_input_noise_amp=[0.5,1.5]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
