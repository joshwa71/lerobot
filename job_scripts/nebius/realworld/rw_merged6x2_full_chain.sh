#!/bin/bash
# E65 REAL-WORLD (WidowX AI) merged-6x2 chain — realworld DUPLICATE of
# libero_90/staged/joint_merged6x2_e468101416_v579111315_prepass_full_chain.sh (E62, the paper
# cell: 12 sites / 7 shared tables / 2.8B values; expert [4,6,8,10,14,16] share (4,6)+(8,10),
# solo 14/16; VLM [5,7,9,11,13,15] all pairs shared; B's router recipe: anchored w0.40, sep8,
# c0.05, FiLM-free, broadcast losses, frozen_prepass; sequential C-config: corefrac beta4,
# top_t 3072, lr 2e-3 -> 2e-4, 5k steps/task). SINGLE DELTA vs the sim chain = the data.
#
# Stages (each skip-guarded on its output; preemption-safe: stage-1 --resume, sequential resume):
#   0  stage-1 base finetune   pi05_base -> RW pretrain split, no memory       rw_stage1_base.sh
#   1  router warm-up          both towers, values pinned, aux losses only     rw_rwarmup_common.sh
#   2  held-out routing audit  RW seq split, inert sweep + analyses            rw_audit_heldout_routing.sh
#   3  bg-first gate           E59 standing rule (bg <= 0.10, mean core50 >= 400, min-eff >= 300;
#                              VLM min-eff >= 150); famIoU INFORMATIONAL and None-safe (the LIBERO
#                              gate hard-crashed on a None famIoU when the (4,5,7) family is absent)
#   4  A-phase                 values both towers, routers frozen, RW pretrain rw_aphase_seq_common.sh
#   5  sequential              RW seq split, --eval.type=loss, per-task ckpts  rw_aphase_seq_common.sh
#
# Usage:      RW_TAG=v5 RW_FAMILY=1-3 bash rw_merged6x2_full_chain.sh
# Smoke test: SMOKE=1 RW_TAG=v1smoke RW_PRETRAIN_ROOT=$ROOT/outputs/realworld_pretrain \
#             RW_SEQ_ROOT=$ROOT/outputs/realworld_seq RW_SEQ_TASK_IDS='[0,1]' bash rw_merged6x2_full_chain.sh
# Launch under systemd (CLAUDE.md 9.5): sudo systemd-run --unit=rw-chain --property=User=josh \
#   --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot \
#   --setenv=RW_TAG=v5 --setenv=RW_FAMILY=1-3 /bin/bash /home/josh/lerobot/job_scripts/nebius/realworld/rw_merged6x2_full_chain.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/rw_env.sh"

# ---- stage 0: stage-1 base finetune (no memory) ----
source "$SCRIPT_DIR/rw_stage1_base.sh"

# ---- stage 1+2: warm-up + audit + analyses ----
export ARM_TAG=${ARM_TAG:-merged6x2_e468101416_v579111315_anchor040_sep8_prepass}
export EXP_LAYERS='[4,6,8,10,14,16]'
export VLM_LAYERS='[5,7,9,11,13,15]'
export SHARE_GROUPS='[[4,6],[8,10]]'
export VLM_SHARE_GROUPS='[[5,7],[9,11],[13,15]]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
export LANG_TO_QUERY=false
export EXPERT_ANCHOR=text
export EXPERT_ANCHOR_W=${EXPERT_ANCHOR_W:-0.40}
export VLM_POOL_W=${VLM_POOL_W:-[1.0,0.5]}
export SEP_W=${SEP_W:-8.0}
export CONTRASTIVE_W=${CONTRASTIVE_W:-0.05}
export PREPASS=true
source "$SCRIPT_DIR/rw_rwarmup_common.sh"

# ---- stage 3: automated gate (BG-FIRST bands, per SITE — E59 standing rule; None-safe) ----
AUDIT_DIR="$ROOT_DIR/outputs/train/$AUDIT_RUN"
if python - "$AUDIT_DIR" <<'EOF'
import json, sys
base = sys.argv[1] + "/"
exp = json.load(open(base + "expert_audit_summary.json"))
vlm = json.load(open(base + "vlm_audit_summary.json"))
fails = []
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("_bgIoU")}, key=lambda x: int(x[1:]))
def tasks(d, L):
    return sorted(int(k.split("_t")[1]) for k in d if k.startswith(f"{L}_t"))
def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"
for L in layers(exp):
    f, bg = exp.get(f"{L}_famIoU"), exp.get(f"{L}_bgIoU")
    ts = tasks(exp, L)
    cores = [exp[f"{L}_t{t}"]["core50"] for t in ts]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in ts]
    print(f"[gate] expert {L}: bg {fmt(bg)} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f} (famIoU {fmt(f)} informational; {len(ts)} tasks)")
    if bg is not None and bg > 0.10: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.10")
    if sum(cores) / len(cores) < 400: fails.append(f"expert {L} mean core50 < 400")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
for L in layers(vlm):
    f = vlm.get(f"{L}_famIoU")
    ts = tasks(vlm, L)
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in ts]
    print(f"[gate] vlm {L}: min-eff {min(effs):.0f} bg {fmt(vlm.get(f'{L}_bgIoU'))} (famIoU {fmt(f)} informational)")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150 (palette-collapse tripwire)")
    if f is not None and f >= 0.45: fails.append(f"vlm {L} famIoU {f:.3f} >= 0.45 backstop")
if fails:
    print("GATE: HARD FAIL"); [print("  -", x) for x in fails]; sys.exit(1)
print("GATE: PASS (bg-first bands)")
EOF
then
  echo "RW merged6x2 chain: gate PASSED - graduating to A-phase + sequential."
elif [ "$SMOKE" = "1" ]; then
  echo "RW merged6x2 chain: gate result INFORMATIONAL under SMOKE=1 - continuing."
else
  echo "RW merged6x2 chain: GATE FAILED - stopping after certificate (override: rerun with SKIP_GATE=1)."
  [ "${SKIP_GATE:-0}" = "1" ] || exit 1
fi

# ---- stage 4+5: A-phase + sequential (C-config levers verbatim from E62) ----
export WARM_RUN=$RUN
export GRAD_TAG=${ARM_TAG}
# E65 add-14/15 (Josh, 30 Aug): env-overridable so the top_t rerun is a SINGLE DELTA on the same
# A-phase checkpoint. Default unchanged (3072) => byte-identical when unset. The RW rerun uses
# SEQ_TOP_T=1536: task 1 reads too few distinct slots per batch for k=min(top_t, n_read) to leave
# corefrac's zero-score core slots out of the mask (mask saturation, E65 add-14).
export SEQ_TOP_T=${SEQ_TOP_T:-3072}
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=${SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_${GRAD_TAG}_beta4corefrac_topt3072_lr2x_steps5k}
if [ "$SMOKE" = "1" ]; then
  export SEQ_LADDER=${SEQ_LADDER:-"16:2:false"}
  export A_LADDER=${A_LADDER:-"16:2:false"}
else
  export SEQ_LADDER=${SEQ_LADDER:-"32:1:false,16:2:false,8:4:false,16:2:true"}
fi
A_OUT_GUARD=$ROOT_DIR/outputs/train/${RUN_PREFIX}realworld_${RW_TAG}_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi
source "$SCRIPT_DIR/rw_aphase_seq_common.sh"
echo "RW-CHAIN-DONE tag=$RW_TAG smoke=$SMOKE at $(date)"
