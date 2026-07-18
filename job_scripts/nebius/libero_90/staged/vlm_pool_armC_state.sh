#!/bin/bash
# E45 arm C: state-only pool (no instruction anchor) — Josh's option-2 variant and the
# probe's predicted-weak point (state-pool between-task cos 0.931/0.970, family
# 0.981/0.991 at init). Runs as the bracket end + probe-calibration cell: if it audits
# healthy despite the raw-feature crowding, the trained-proj caveat is real and the
# querystats instrument undersells trainability.
ARM_TAG=statepool_c0.05_sep5.0
POOL_MODE=state
POOL_W='[1.0,1.0]'
C_WEIGHT=0.05
SEP_WEIGHT=5.0
source "$(dirname "$0")/vlm_pool_chain_common.sh"
