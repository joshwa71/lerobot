#!/usr/bin/env bash
# E65 heartbeat — watcher for the REAL-WORLD merged-6x2 chain (unit rw-chain on nebius-spot,
# log outputs/rw_chain_v5.log) and, with WEEKEND=1, the weekend baseline queue behind it
# (unit rw-weekend: battery -> smokes -> r64 specialists -> r64 naive sequential -> matrices;
# log outputs/rw_weekend_v5.log). Successor to heartbeat_e64c.sh; same rules (E63 add-5 / E64 add-3):
#   - reports ARTIFACT STATE (checkpoint dirs, JSON counts, gate line, unit state, disk), never
#     "the last line that matched";
#   - one multiplexed SSH connection (ControlMaster) — no login-session churn (CLAUDE.md 9.5.1);
#   - VM unreachable is cross-checked against github:22 before concluding preemption (1 Aug
#     precedent), then recovered via the local nebius CLI and the unit(s) relaunched — every chain /
#     queue stage is skip-guarded / resumable, so a relaunch is always safe and never re-measures.
# Emits on discrete state change + a forced beat every HEARTBEAT_EVERY polls; always emits on:
# new error lines, gate verdict, unit down with the chain/queue incomplete, disk tripwire, unreachable.
# ONESHOT=1 prints one state line and exits (for the cron self-prompt).
# SKIP_GATE=1 mirrors a gate-overridden launch: passed through on relaunch, and a HARD_FAIL gate line
# is then NOT read as 'stopped at gate' (a dead unit is reported as UNIT DOWN instead).
# WEEKEND=1: also watch/relaunch rw-weekend; the watcher exits only when BOTH are done. A queue that
# stopped on RW-WEEKEND-SMOKE-FAIL / -BOOTSTRAP-FAIL is NOT relaunched (needs a fix); a queue that dies
# 3x without artifact progress is left alone (manual).
set -uo pipefail

VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00hks7a4fq3atcpsm
UNIT=rw-chain
WUNIT=rw-weekend
RW_TAG=${RW_TAG:-v5}
RW_FAMILY=${RW_FAMILY:-0-4,3-4}
ARM=${ARM:-merged6x2_e468101416_v579111315_anchor040_sep8_prepass}
SEP_W=${SEP_W:-8.0}
CONTRASTIVE_W=${CONTRASTIVE_W:-0.05}
EXPERT_ANCHOR_W=${EXPERT_ANCHOR_W:-0.40}
VLM_POOL_W=${VLM_POOL_W:-[1.0,0.5]}
SKIP_GATE=${SKIP_GATE:-0}   # 1 = chain launched with the gate overridden (Josh's call, E65 add-12); relaunches MUST carry it
WEEKEND=${WEEKEND:-0}
LORA_R=${LORA_R:-64}
CHAIN=/home/josh/lerobot/job_scripts/nebius/realworld/rw_merged6x2_full_chain.sh
QUEUE=/home/josh/lerobot/scripts/vla_analysis/realworld/run_rw_weekend_queue.sh
LOG=/home/josh/lerobot/outputs/rw_chain_${RW_TAG}.log
WLOG=/home/josh/lerobot/outputs/rw_weekend_${RW_TAG}.log
BLOG=/home/josh/lerobot/outputs/rw_battery_${RW_TAG}.log
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-36}     # 6 h forced emit at POLL=600
DISK_TRIPWIRE=${DISK_TRIPWIRE:-88}
ONESHOT=${ONESHOT:-0}

ts() { date -u +%H:%MZ; }
emit() { echo "[$(ts)] $*"; }

remote_state() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" "RW_TAG=$RW_TAG LOG=$LOG WLOG=$WLOG BLOG=$BLOG ARM=$ARM WEEKEND=$WEEKEND LORA_R=$LORA_R bash -s" <<'REMOTE'
R=/home/josh/lerobot/outputs/train
SEQR=realworld_${RW_TAG}_seq5_jw_${ARM}_beta4corefrac_topt3072_lr2x_steps5k
u=$(systemctl is-active rw-chain 2>/dev/null); true
s1=$(ls -d $R/realworld_${RW_TAG}_pi05_base_nomem_50k/checkpoints/[0-9]* 2>/dev/null | wc -l)
s1f=$([ -d $R/realworld_${RW_TAG}_pi05_base_nomem_50k/checkpoints/050000 ] && echo 1 || echo 0)
warm=$([ -d $R/realworld_${RW_TAG}_pi05_jointwarm10k_${ARM}/checkpoints/last/pretrained_model ] && echo 1 || echo 0)
aud=$(ls $R/audit_heldout_rw_${RW_TAG}_jointwarm_${ARM}_10k/memory_by_task/*.json 2>/dev/null | wc -l)
gate=$(grep -oE "^GATE: (PASS|HARD FAIL)" $LOG 2>/dev/null | tail -1 | cut -d' ' -f2- | tr ' ' '_')
aph=$([ -d $R/realworld_${RW_TAG}_pi05_jointA10k_${ARM}/checkpoints/last/pretrained_model ] && echo 1 || echo 0)
seq=$(ls -d $R/$SEQR/checkpoints/[0-9]* 2>/dev/null | wc -l)
lrows=$(wc -l < $R/$SEQR/eval/loss_results.jsonl 2>/dev/null); lrows=${lrows:-0}
step=$(grep -oE "step:[0-9]+K?" $LOG 2>/dev/null | tail -1 | cut -d: -f2)
stage=$(grep -oE "^\[(stage1|warmup|audit|A-phase|seq)\]|^RW joint router warm-up|^Audit .* started|^RW graduation chain|^RW-CHAIN-DONE" $LOG 2>/dev/null | tail -1 | tr ' ' '_' | cut -c1-24)
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
er=$(grep -cE "Traceback|OutOfMemoryError|CUDA out of memory|^ERROR:|ERROR: all" $LOG 2>/dev/null); er=${er:-0}
fin=$(grep -c "RW-CHAIN-DONE" $LOG 2>/dev/null); fin=${fin:-0}
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
line="unit=$u s1=$s1/5ck(final=$s1f) warm=$warm audit=$aud/5 gate=${gate:-none} A=$aph seq=$seq/5 lossrows=$lrows step=${step:-0} last=${stage:-none} disk=${dk}% err=$er fin=$fin gpu=${gpu:-NA}"
if [ "$WEEKEND" = "1" ]; then
  wk=$(systemctl is-active rw-weekend 2>/dev/null); true
  bat=$(grep -c "RW-BATTERY-DONE" $BLOG 2>/dev/null); bat=${bat:-0}
  spec=$(ls -d $R/rw_${RW_TAG}_loraft_baseline_r${LORA_R}/task*/checkpoints/005000 2>/dev/null | wc -l)
  nv=$(ls -d $R/realworld_${RW_TAG}_seq5_naive_lora_r${LORA_R}_a*_steps5k/checkpoints/[0-9]* 2>/dev/null | wc -l)
  wst=$(grep -oE "^\[weekend\] [a-z-]+" $WLOG 2>/dev/null | tail -1 | cut -d' ' -f2)
  wkerr=$(grep -cE "Traceback|OutOfMemoryError|FAILED|SMOKE-FAIL|BOOTSTRAP-FAIL|INCOMPLETE" $WLOG 2>/dev/null); wkerr=${wkerr:-0}
  wkstop=$(grep -cE "RW-WEEKEND-(SMOKE|BOOTSTRAP)-FAIL" $WLOG 2>/dev/null); wkstop=${wkstop:-0}
  wkfin=$(grep -c "RW-WEEKEND-DONE" $WLOG 2>/dev/null); wkfin=${wkfin:-0}
  line="$line | wk=${wk:-none} bat=$bat spec=$spec/5 naive=$nv/5 wkstage=${wst:-none} wkerr=$wkerr wkstop=$wkstop wkfin=$wkfin"
fi
echo "$line"
REMOTE
}

key_of() { sed -E 's/ step=[0-9]*K?//; s/ gpu=[^ ]*//' <<<"$1"; }
field() { sed -nE "s/.*(^| )$1=([^ |]+).*/\2/p" <<<"$2"; }

relaunch_unit() {
  ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "sudo systemctl reset-failed $UNIT 2>/dev/null; sudo systemd-run --unit=$UNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot --setenv=RW_TAG=$RW_TAG '--setenv=RW_FAMILY=$RW_FAMILY' --setenv=ARM_TAG=$ARM --setenv=SEP_W=$SEP_W --setenv=CONTRASTIVE_W=$CONTRASTIVE_W --setenv=EXPERT_ANCHOR_W=$EXPERT_ANCHOR_W '--setenv=VLM_POOL_W=$VLM_POOL_W' --setenv=SKIP_GATE=$SKIP_GATE /bin/bash -c 'bash $CHAIN >> $LOG 2>&1'" >/dev/null 2>&1
}

# The weekend BOOTSTRAP: waits for the chain to finish (RW-CHAIN-DONE + unit stopped), then git-pulls
# (unit stopped => allowed, CLAUDE.md 9.8) and runs the queue. Idempotent: every queue stage is guarded.
relaunch_weekend() {
  ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "sudo systemctl reset-failed $WUNIT 2>/dev/null; sudo systemd-run --unit=$WUNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot --setenv=RW_TAG=$RW_TAG --setenv=ARM_TAG=$ARM --setenv=LORA_R=$LORA_R /bin/bash -c 'until grep -q RW-CHAIN-DONE $LOG 2>/dev/null && ! systemctl is-active --quiet $UNIT; do sleep 60; done; echo \"[bootstrap] chain done; pulling \$(date -u)\" >> $WLOG; git pull -q --ff-only >> $WLOG 2>&1 || echo \"[bootstrap] git pull FAILED\" >> $WLOG; [ -f $QUEUE ] || { echo RW-WEEKEND-BOOTSTRAP-FAIL >> $WLOG; exit 1; }; bash $QUEUE >> $WLOG 2>&1'" >/dev/null 2>&1
}

recover() {
  emit "RECOVERY: probing VM state via nebius API"
  local st ip cfg_ip
  st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
  emit "RECOVERY: API reports state=${st:-UNKNOWN}"
  if [ "$st" != "RUNNING" ]; then
    for a in 1 2 3 4 5; do
      "$NEBIUS" compute instance start --id "$VM_ID" >/dev/null 2>&1 && { emit "RECOVERY: start issued (attempt $a)"; break; }
      emit "RECOVERY: start failed (attempt $a) — backing off $((a*30))s"; sleep $((a*30))
    done
    for _ in $(seq 1 60); do
      st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
      [ "$st" = "RUNNING" ] && break; sleep 10
    done
    [ "$st" = "RUNNING" ] || { emit "RECOVERY FAILED: VM did not reach RUNNING — manual intervention needed"; return 1; }
  fi
  ip=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null \
       | jq -r '.status.network_interfaces[0].public_ip_address.address' 2>/dev/null | cut -d/ -f1)
  cfg_ip=$(awk '/Host nebius-spot/{f=1} f&&/HostName/{print $2; exit}' ~/.ssh/config)
  [ -n "$ip" ] && [ "$ip" != "$cfg_ip" ] && emit "RECOVERY WARNING: public IP $ip != ssh config $cfg_ip — update ~/.ssh/config"
  for _ in $(seq 1 30); do
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null && break; sleep 10
  done
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null || { emit "RECOVERY FAILED: RUNNING but sshd never came up"; return 1; }
  emit "RECOVERY: SSH back up"
  local st2 u fin
  st2=$(remote_state 2>/dev/null); u=$(field unit "$st2"); fin=$(field fin "$st2")
  if [ "${fin:-0}" -ge 1 ] 2>/dev/null; then
    emit "RECOVERY: chain already complete (fin=$fin) — not relaunching $UNIT"
  elif [ "$u" != "active" ] && [ "$u" != "activating" ]; then
    relaunch_unit && emit "RECOVERY: $UNIT relaunched (stage-1 resumes from its last save_freq checkpoint; sequential resumes from the last task boundary; finished stages skip)" \
                  || emit "RECOVERY FAILED: could not relaunch $UNIT"
  else
    emit "RECOVERY: $UNIT already $u — nothing to relaunch"
  fi
  if [ "$WEEKEND" = "1" ]; then
    local wk wkfin wkstop
    wk=$(field wk "$st2"); wkfin=$(field wkfin "$st2"); wkstop=$(field wkstop "$st2")
    if [ "${wkfin:-0}" -ge 1 ] 2>/dev/null; then emit "RECOVERY: weekend queue already complete"
    elif [ "${wkstop:-0}" -ge 1 ] 2>/dev/null; then emit "RECOVERY: weekend queue had stopped on a smoke/bootstrap failure — NOT relaunched"
    elif [ "$wk" != "active" ] && [ "$wk" != "activating" ]; then
      relaunch_weekend && emit "RECOVERY: $WUNIT bootstrap relaunched (waits for the chain; queue stages skip-guarded; naive self-resumes)" \
                       || emit "RECOVERY FAILED: could not relaunch $WUNIT"
    else emit "RECOVERY: $WUNIT already $wk"; fi
  fi
}

if [ "${LAUNCH_WEEKEND:-0}" = "1" ]; then   # one-off: launch the rw-weekend bootstrap with the exact relaunch command
  relaunch_weekend && emit "rw-weekend bootstrap launched (waits for RW-CHAIN-DONE, pulls, runs the queue)" || emit "rw-weekend launch FAILED"
  exit 0
fi
if [ "$ONESHOT" = "1" ]; then
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then emit "$state"; else emit "VM UNREACHABLE"; fi
  exit 0
fi

prev_key=""; prev_err=0; prev_wkerr=0; prev_gate="none"; i=0; unreachable_run=0
chain_done_said=0; wk_relaunches=0; wk_prev_progress=""; wk_stop_said=0; wk_giveup_said=0
emit "heartbeat-RW armed: unit $UNIT tag $RW_TAG arm $ARM skip_gate=$SKIP_GATE weekend=$WEEKEND; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/3600))h"
while true; do
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then
    if [ "$unreachable_run" -gt 0 ]; then
      emit "VM reachable again after $unreachable_run failed poll(s) | $state"
      unreachable_run=0; prev_key=""
    fi
    key="$(key_of "$state")"
    err=$(field err "$state"); disk=$(field disk "$state" | tr -dc '0-9'); unit=$(field unit "$state")
    fin=$(field fin "$state"); gate=$(field gate "$state")

    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $state"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $state"
    if [ "$gate" != "$prev_gate" ] && [ "$gate" != "none" ]; then
      emit "GATE VERDICT: $gate | $state"; prev_gate="$gate"
    fi
    if [ "${fin:-0}" -ge 1 ] 2>/dev/null; then
      [ "$chain_done_said" = 0 ] && { emit "RW CHAIN COMPLETE | $state"; chain_done_said=1; }
      [ "$WEEKEND" != "1" ] && exit 0
    elif [ "$unit" != "active" ] && [ "$unit" != "activating" ]; then
      if [ "$gate" = "HARD_FAIL" ] && [ "$SKIP_GATE" != "1" ]; then
        emit "CHAIN STOPPED AT GATE (by design; SKIP_GATE=1 to override) | $state"
      else
        emit "UNIT DOWN with chain incomplete | $state"
      fi
    fi
    if [ "$WEEKEND" = "1" ]; then
      wk=$(field wk "$state"); wkerr=$(field wkerr "$state"); wkstop=$(field wkstop "$state"); wkfin=$(field wkfin "$state")
      progress="$(field bat "$state")/$(field spec "$state")/$(field naive "$state")"
      [ "${wkerr:-0}" -gt "${prev_wkerr:-0}" ] 2>/dev/null && emit "WEEKEND NEW ERROR LINES (wkerr ${prev_wkerr}->${wkerr}) | $state"
      prev_wkerr=${wkerr:-0}
      if [ "${wkfin:-0}" -ge 1 ] 2>/dev/null; then
        emit "RW WEEKEND QUEUE COMPLETE | $state"; exit 0
      elif [ "$wk" != "active" ] && [ "$wk" != "activating" ]; then
        if [ "${wkstop:-0}" -ge 1 ] 2>/dev/null; then
          [ "$wk_stop_said" = 0 ] && { emit "WEEKEND QUEUE STOPPED on smoke/bootstrap failure — needs a fix, NOT relaunching | $state"; wk_stop_said=1; }
        elif [ "$wk_relaunches" -ge 3 ] && [ "$progress" = "$wk_prev_progress" ]; then
          [ "$wk_giveup_said" = 0 ] && { emit "WEEKEND QUEUE down 3x without artifact progress — giving up on auto-relaunch (manual) | $state"; wk_giveup_said=1; }
        else
          [ "$progress" != "$wk_prev_progress" ] && wk_relaunches=0
          wk_relaunches=$((wk_relaunches+1)); wk_prev_progress="$progress"
          relaunch_weekend && emit "WEEKEND UNIT DOWN with queue incomplete — bootstrap relaunched (#$wk_relaunches) | $state" \
                           || emit "WEEKEND UNIT DOWN — relaunch FAILED | $state"
        fi
      fi
    fi
    if [ "$key" != "$prev_key" ] || [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; then
      emit "$state"; prev_key="$key"
    fi
    prev_err=${err:-0}
  else
    unreachable_run=$((unreachable_run+1))
    if nc -z -w5 github.com 22 2>/dev/null; then
      emit "VM UNREACHABLE (poll $unreachable_run) but github:22 reachable => VM-side"
      [ "$unreachable_run" -ge 2 ] && recover
    else
      emit "VM unreachable (poll $unreachable_run) AND github:22 unreachable => LOCAL network outage; no action (1 Aug precedent)"
    fi
  fi
  i=$((i+1)); sleep "$POLL"
done
