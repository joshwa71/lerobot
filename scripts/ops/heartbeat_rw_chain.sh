#!/usr/bin/env bash
# E65 heartbeat — watcher for the REAL-WORLD merged-6x2 chain (unit rw-chain on nebius-spot,
# log outputs/rw_chain_v5.log). Successor to heartbeat_e64c.sh; same rules (E63 add-5 / E64 add-3):
#   - reports ARTIFACT STATE (checkpoint dirs, JSON counts, gate line, unit state, disk), never
#     "the last line that matched";
#   - one multiplexed SSH connection (ControlMaster) — no login-session churn (CLAUDE.md 9.5.1);
#   - VM unreachable is cross-checked against github:22 before concluding preemption (1 Aug
#     precedent), then recovered via the local nebius CLI and the unit relaunched — every chain
#     stage is skip-guarded / resumable (stage-1 --resume, sequential resume), so a relaunch is
#     always safe and never re-measures.
# Emits on discrete state change + a forced beat every HEARTBEAT_EVERY polls; always emits on:
# new error lines, gate verdict, unit down with the chain incomplete, disk tripwire, unreachable.
# ONESHOT=1 prints one state line and exits (for the cron self-prompt).
set -uo pipefail

VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00hks7a4fq3atcpsm
UNIT=rw-chain
RW_TAG=${RW_TAG:-v5}
RW_FAMILY=${RW_FAMILY:-0-4,3-4}
ARM=${ARM:-merged6x2_e468101416_v579111315_anchor040_sep8_prepass}
SEP_W=${SEP_W:-8.0}
CONTRASTIVE_W=${CONTRASTIVE_W:-0.05}
EXPERT_ANCHOR_W=${EXPERT_ANCHOR_W:-0.40}
VLM_POOL_W=${VLM_POOL_W:-[1.0,0.5]}
CHAIN=/home/josh/lerobot/job_scripts/nebius/realworld/rw_merged6x2_full_chain.sh
LOG=/home/josh/lerobot/outputs/rw_chain_${RW_TAG}.log
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-36}     # 6 h forced emit at POLL=600
DISK_TRIPWIRE=${DISK_TRIPWIRE:-88}
ONESHOT=${ONESHOT:-0}

ts() { date -u +%H:%MZ; }
emit() { echo "[$(ts)] $*"; }

remote_state() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" "RW_TAG=$RW_TAG LOG=$LOG ARM=$ARM bash -s" <<'REMOTE'
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
er=$(grep -cE "Traceback|OutOfMemoryError|CUDA out of memory|^ERROR:|ERROR: all" $LOG 2>/dev/null)
fin=$(grep -c "RW-CHAIN-DONE" $LOG 2>/dev/null)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "unit=$u s1=$s1/5ck(final=$s1f) warm=$warm audit=$aud/5 gate=${gate:-none} A=$aph seq=$seq/5 lossrows=$lrows step=${step:-0} last=${stage:-none} disk=${dk}% err=$er fin=$fin gpu=${gpu:-NA}"
REMOTE
}

key_of() { sed -E 's/ step=[0-9]*K?//; s/ gpu=[^ ]*//' <<<"$1"; }

relaunch_unit() {
  ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "sudo systemctl reset-failed $UNIT 2>/dev/null; sudo systemd-run --unit=$UNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot --setenv=RW_TAG=$RW_TAG '--setenv=RW_FAMILY=$RW_FAMILY' --setenv=ARM_TAG=$ARM --setenv=SEP_W=$SEP_W --setenv=CONTRASTIVE_W=$CONTRASTIVE_W --setenv=EXPERT_ANCHOR_W=$EXPERT_ANCHOR_W '--setenv=VLM_POOL_W=$VLM_POOL_W' /bin/bash -c 'bash $CHAIN >> $LOG 2>&1'" >/dev/null 2>&1
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
  local u; u=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "systemctl is-active $UNIT 2>/dev/null; true")
  if [ "$u" != "active" ] && [ "$u" != "activating" ]; then
    relaunch_unit && emit "RECOVERY: $UNIT relaunched (stage-1 resumes from its last save_freq checkpoint; sequential resumes from the last task boundary; finished stages skip)" \
                  || emit "RECOVERY FAILED: could not relaunch $UNIT"
  else
    emit "RECOVERY: $UNIT already $u — nothing to relaunch"
  fi
}

if [ "$ONESHOT" = "1" ]; then
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then emit "$state"; else emit "VM UNREACHABLE"; fi
  exit 0
fi

prev_key=""; prev_err=0; prev_gate="none"; i=0; unreachable_run=0
emit "heartbeat-RW armed: unit $UNIT tag $RW_TAG; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/3600))h"
while true; do
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then
    if [ "$unreachable_run" -gt 0 ]; then
      emit "VM reachable again after $unreachable_run failed poll(s) | $state"
      unreachable_run=0; prev_key=""
    fi
    key="$(key_of "$state")"
    err=$(sed -E 's/.* err=([0-9]+) .*/\1/' <<<"$state")
    disk=$(sed -E 's/.* disk=([0-9]+)%.*/\1/' <<<"$state")
    unit=$(sed -E 's/^unit=([a-z]+) .*/\1/' <<<"$state")
    fin=$(sed -E 's/.* fin=([0-9]+) .*/\1/' <<<"$state")
    gate=$(sed -E 's/.* gate=([A-Za-z_]+) .*/\1/' <<<"$state")

    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $state"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $state"
    if [ "$gate" != "$prev_gate" ] && [ "$gate" != "none" ]; then
      emit "GATE VERDICT: $gate | $state"; prev_gate="$gate"
    fi
    if [ "$unit" != "active" ] && [ "$unit" != "activating" ]; then
      if [ "${fin:-0}" -ge 1 ] 2>/dev/null; then
        emit "RW CHAIN COMPLETE | $state"; exit 0
      elif [ "$gate" = "HARD_FAIL" ]; then
        emit "CHAIN STOPPED AT GATE (by design; SKIP_GATE=1 to override) | $state"
      else
        emit "UNIT DOWN with chain incomplete | $state"
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
