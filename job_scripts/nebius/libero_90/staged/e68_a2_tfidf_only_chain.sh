#!/bin/bash
# E68 ablation A2 — TF-IDF-ONLY WRITES (prior-usefulness protection OFF) on the paper cell.
# Forks the paper cell's 5-task control at its TASK-1 BOUNDARY: with an empty protection store the
# write score is tf*idf (trainer docstring: "With an empty store (task 0) score == tfidf"), so task 1
# is identical by construction. Tasks 2-5 run with protect_prior_slots=false; everything else
# C-config verbatim (top_t 3072, online IDF restored from sequential_state.pt, lr2x, optimizer reset,
# union masks on shared tables, frozen-base routing + pre-pass). Replaces the missing "no protection"
# row = the Lin et al. sparse-memory-finetuning recipe at our budget. Pre-registration: E68.
# The paper cell's A-phase checkpoint must be present (stage A is then skipped by the common body).
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "E68-A2 TF-IDF-only chain started on $(hostname) at $(date -u)"

CONTROL_RUN=libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export GRAD_TAG=merged6x2_e468101416_v579111315_anchor040_sep8_prepass   # => A_RUN = the paper A-phase (present => skipped)
export SEQ_RUN=libero_10_seq5_jw_e68a2_tfidfonly_merged6x2_e468101416_v579111315_prepass_noprotect_topt3072_lr2x_steps5k
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
SRC="$ROOT_DIR/outputs/train/$CONTROL_RUN/checkpoints/005000"
A_CKPT_REQ="$ROOT_DIR/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}/checkpoints/010000/pretrained_model"
[ -d "$A_CKPT_REQ" ] || { echo "ERROR: paper A-phase checkpoint missing: $A_CKPT_REQ"; exit 1; }

# ---- stage 0: seed the run dir with the control's task-1 boundary (copy + byte-count witness) ----
if [ ! -f "$SEQ_OUT/checkpoints/005000/sequential_state.pt" ]; then
  [ -f "$SRC/sequential_state.pt" ] && [ -d "$SRC/pretrained_model" ] || { echo "ERROR: control task-1 boundary missing at $SRC"; exit 1; }
  mkdir -p "$SEQ_OUT/checkpoints" "$SEQ_OUT/memory_by_task"
  rm -rf "$SEQ_OUT/checkpoints/005000.tmp"
  cp -a "$SRC" "$SEQ_OUT/checkpoints/005000.tmp"
  a=$(du -sb "$SRC" | cut -f1); b=$(du -sb "$SEQ_OUT/checkpoints/005000.tmp" | cut -f1)
  [ "$a" = "$b" ] || { echo "ERROR: seed copy size mismatch ($a vs $b)"; exit 1; }
  mv "$SEQ_OUT/checkpoints/005000.tmp" "$SEQ_OUT/checkpoints/005000"
  ln -sfn 005000 "$SEQ_OUT/checkpoints/last"
  cp -a "$ROOT_DIR/outputs/train/$CONTROL_RUN/memory_by_task/memory_usage_task_0.json" "$SEQ_OUT/memory_by_task/" 2>/dev/null \
    || echo "[a2] control task-0 usage json not found (non-fatal; diagnostics only)"
  echo "[a2] seeded $SEQ_OUT/checkpoints/005000 from the control ($a bytes); last -> 005000"
fi

# ---- stage B: tasks 2-5 with protection OFF (resume path of the common body) ----
export SEQ_PROTECT=false
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_UNORM=corefrac
# Rung selection lives HERE (not the common body's SEQ_LADDER): the body's ladder treats a failure
# with checkpoints/005000 present as "not VRAM" and would never step down, and its wipe would delete
# the seed. A rung that fails before 010000 exists is a VRAM failure -> clean partial artifacts
# (keeping the seed) and try the next rung; a failure after 010000 is NOT VRAM -> abort loudly.
A2_LADDER="${A2_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"
if [ -d "$SEQ_OUT/checkpoints/025000" ]; then
  echo "[a2] final checkpoint exists - nothing to do."
else
  ok=0
  for rung in ${A2_LADDER//,/ }; do
    IFS=: read -r rb ra rc <<< "$rung"
    echo "[a2] rung: bs=$rb accum=$ra grad_ckpt=$rc"
    if SEQ_BS=$rb SEQ_ACCUM=$ra SEQ_GRAD_CKPT=$rc SEQ_LADDER= bash "$SCRIPT_DIR/joint_aphase_seq5_common.sh"; then ok=1; break; fi
    if [ -d "$SEQ_OUT/checkpoints/010000" ]; then
      echo "[a2] rung failed AFTER the task-2 checkpoint - not a VRAM failure; aborting (relaunch resumes)."; exit 1
    fi
    echo "[a2] rung failed before 010000 (treating as VRAM) - cleaning partial artifacts, keeping the seed"
    find "$SEQ_OUT/checkpoints" -mindepth 1 -maxdepth 1 -type d ! -name 005000 -exec rm -rf {} +
    ln -sfn 005000 "$SEQ_OUT/checkpoints/last"
  done
  [ "$ok" = 1 ] || { echo "ERROR: all A2 rungs failed"; exit 1; }
fi
[ -d "$SEQ_OUT/checkpoints/025000" ] || { echo "ERROR: A2 finished but 025000 missing"; exit 1; }
echo "E68-A2 TF-IDF-only chain COMPLETE at $(date -u)"
