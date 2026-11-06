#!/bin/bash
# E45 arm A (base box): anchored pool, (a,b)=(1.0,0.0) — pure instruction-anchor key for
# the state region. The querystats probe's dominant point: best init separation
# (inter-cos 0.800/0.866, family 0.882/0.919) with within-task conditionality already
# carried by contextual bleed into instruction tokens (intra-cos 0.864/0.890 — NOT the
# E21 0.99 constant-palette regime).
ARM_TAG=anchor10_c0.05_sep5.0
POOL_MODE=anchored
POOL_W='[1.0,0.0]'
C_WEIGHT=0.05
SEP_WEIGHT=5.0
source "$(dirname "$0")/vlm_pool_chain_common.sh"
