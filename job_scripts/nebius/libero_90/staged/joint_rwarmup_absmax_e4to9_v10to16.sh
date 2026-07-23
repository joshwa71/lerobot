#!/bin/bash
# E51 Part-9 ABSOLUTE LAYER-MAX warm-up (Josh's option 1): expert [4-9] (6 contiguous)
# + VLM text-field [10-16] (7 contiguous) = 13 modules, 5.4B values, n256/r2.
# CERTIFY-FIRST: this wrapper runs warm-up -> audit -> analyses -> STOP for manual
# review (the automated gate's 3-of-4 expert rule doesn't transfer to a 6-layer
# stack; read the per-layer table against the bands below). The full chain launches
# only on a pass, with its layout possibly revised by the spread-A verdict
# (A >= compact -> consider spreading; A < compact -> contiguous stands).
#
# Evidence base: layers is the only axis that has paid four times running (VLM build
# +6, layermax +4.8, levers-on-substrate +6). Depth bands: expert L4 is the known
# marginal layer (attempt-A re-audit: famIoU ~0.195 at L4, 0.212 at L2; L8+ healthy
# 0.14-0.163; E36 probe L4 separability 89.5 vs the 98 plateau) - if ONLY L4 fails
# its band, the trim option is expert [5-9] (rewarm, ~3h). VLM [10-16] is fully
# inside certified territory (attempt-A VLM at 10/12/14/16: 0.132-0.152, parity with
# compact, better anchor geometry per E49).
# Review bands (manual): expert famIoU <= ~0.20 on >= 5/6 layers, core50 healthy
# (~1300+ at the n256 law); VLM famIoU <= ~0.25 all 7, per-task effnum >= ~500,
# no ~2-draw palette collapse (min-task effnum >= 150).
#
# VRAM: warm-up is CHEAP regardless of module count (router-only, values pinned,
# router_only_fast skips the value path - imgspan broadcast precedent: 0.87s/step
# @ 32GiB); bs32 native. The AUDIT runs the full value path + backward + both
# frozen-route forks (expert 4->9, VLM 10->16) at 5.4B values - bs8 x 400 steps
# (matched audited-sample coverage vs bs16x200 / bs32x100; mass-normalized stats
# invariant, E48 precedent).
set -eo pipefail
export ARM_TAG=absmax_e4to9_v10to16
export EXP_LAYERS='[4,5,6,7,8,9]'
export VLM_LAYERS='[10,11,12,13,14,15,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
export AUDIT_BS=8 AUDIT_STEPS=400
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_rwarmup_common.sh"
