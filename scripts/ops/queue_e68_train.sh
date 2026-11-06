#!/bin/bash
# E68 TRAIN queue (VM-side on nebius2 = josh-vm-spot-2, H100 80GB; unit e68-train). Modes:
#   smoke : (1) live-routing smoke suite L/G/X (scripts/vla_analysis/run_smoke_live_routing.sh);
#           (2) stage-A periodic-save/resume smoke on the real warm-up checkpoint with the A1 live
#           flags (run_smoke_e68_aphase_resume.sh) = the H100 profiling gate.
#           Writes outputs/e68/smoke_live_ok / smoke_resume_ok on success.
#   full  : requires both markers; runs $STAGES in order (default "a1 a2 c1 c2"; add a3 to run the
#           joint-prep arm here). Every stage is skip-guarded and resumable, so relaunching this unit
#           after a preemption continues where it stopped (heartbeat_e68.sh does that).
# Markers on stdout: E68-SMOKE-LIVE-FAIL, E68-SMOKE-RESUME-FAIL, E68-SMOKE-DONE, E68-<STAGE>-DONE,
#   E68-<STAGE>-FAIL, E68-TRAIN-DONE.
set -uo pipefail
MODE=${1:-full}
STAGES=${STAGES:-"a1 a2 c1 c2"}
ROOT=/home/josh/lerobot
J=$ROOT/job_scripts/nebius/libero_90/staged
cd $ROOT; mkdir -p $ROOT/outputs/e68
say(){ echo "[e68-train] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E68 TRAIN QUEUE START mode=$MODE stages=$STAGES $(date -u) ==="
if [ "$MODE" = "smoke" ]; then
  say "live-routing smoke (L/G/X)"
  if bash scripts/vla_analysis/run_smoke_live_routing.sh > outputs/e68/smoke_live.log 2>&1; then
    touch outputs/e68/smoke_live_ok; say "live smoke OK"
  else echo "E68-SMOKE-LIVE-FAIL"; say "live smoke FAILED (outputs/e68/smoke_live.log)"; fi
  say "stage-A resume/profiling smoke"
  if bash scripts/vla_analysis/run_smoke_e68_aphase_resume.sh > outputs/e68/smoke_resume.log 2>&1; then
    touch outputs/e68/smoke_resume_ok; say "resume smoke OK"; grep -E "\[PROFILE\]|updt_s" outputs/e68/smoke_resume.log | tail -4
  else echo "E68-SMOKE-RESUME-FAIL"; say "resume smoke FAILED (outputs/e68/smoke_resume.log)"; fi
  echo "E68-SMOKE-DONE"; exit 0
fi
[ -f outputs/e68/smoke_live_ok ] && [ -f outputs/e68/smoke_resume_ok ] || { say "smoke markers missing - run 'queue_e68_train.sh smoke' first"; echo "E68-TRAIN-FAIL"; exit 1; }
A1_SEQ=outputs/train/libero_10_seq5_jw_e68a1_live_merged6x2_e468101416_v579111315_beta4corefrac_topt3072_lr2x_steps5k
A2_SEQ=outputs/train/libero_10_seq5_jw_e68a2_tfidfonly_merged6x2_e468101416_v579111315_prepass_noprotect_topt3072_lr2x_steps5k
A3_SEQ=outputs/train/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
C1_AUD=outputs/train/audit_heldout_jointwarm_e68c1_merged6x2_e468101416_v579111315_anchor040_sep0_c005_prepass_10k
C2_AUD=outputs/train/audit_heldout_jointwarm_e68c2_merged6x2_e468101416_v579111315_anchor040_sep8_c0_prepass_10k
for st in $STAGES; do
  case "$st" in
    a1) script=e68_a1_live_routing_chain.sh; final=$A1_SEQ/checkpoints/025000/pretrained_model ;;
    a2) script=e68_a2_tfidf_only_chain.sh;   final=$A2_SEQ/checkpoints/025000/pretrained_model ;;
    a3) script=e68_a3_jointprep_chain.sh;    final=$A3_SEQ/checkpoints/025000/pretrained_model ;;
    c1) script=e68_c1_cert_sep0.sh;          final=$C1_AUD/expert_audit_summary.json ;;
    c2) script=e68_c2_cert_c0.sh;            final=$C2_AUD/expert_audit_summary.json ;;
    *) say "unknown stage '$st'"; echo "E68-TRAIN-FAIL"; exit 1 ;;
  esac
  U=$(echo "$st" | tr a-z A-Z)
  if [ -e "$final" ]; then say "stage $st already complete"; echo "E68-$U-DONE"; continue; fi
  say "stage $st: $script"
  if ! bash "$J/$script"; then say "stage $st wrapper exited non-zero (resumable; relaunch continues)"; echo "E68-$U-FAIL"; exit 1; fi
  if [ -e "$final" ]; then echo "E68-$U-DONE"; else say "stage $st ended without its final artifact ($final)"; echo "E68-$U-FAIL"; exit 1; fi
done
echo "E68-TRAIN-DONE"
