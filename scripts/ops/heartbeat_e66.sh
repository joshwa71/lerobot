#!/usr/bin/env bash
# E66 heartbeat — watcher for the e66 chain on nebius-spot:
#   stage 0  E63 corrected MSE matrix over all 10 tasks (~80 min GPU)
#   stage 1  E66 smoke (2 tasks x 20 steps) on the VRAM ladder — fatal on failure
#   stage 2  E66 full run: parameter-matched naive seq-LoRA r1216/a304, 10 tasks x 5k steps (~14h)
# Same rules as heartbeat_rw_chain.sh: ARTIFACT STATE (not "last matching line"), one multiplexed
# SSH connection (CLAUDE.md 9.5.1), preemption cross-checked against github:22 before recovery.
# Always emits on: new error lines, stage change, unit down with the chain incomplete, disk
# tripwire, smoke failure, ladder demotion, unreachable. ONESHOT=1 prints one line and exits.
set -uo pipefail
VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00hks7a4fq3atcpsm
UNIT=e66
LOG=/home/josh/lerobot/outputs/e66_paramatched.log
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-6}     # forced emit every ~1h at POLL=600
DISK_TRIPWIRE=${DISK_TRIPWIRE:-88}
ONESHOT=${ONESHOT:-0}
ts(){ date -u +%H:%MZ; }
emit(){ echo "[$(ts)] $*"; }

remote_state(){
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" 'bash -s' <<'REMOTE'
LFULL=/home/josh/lerobot/outputs/e66_paramatched.log
# slice from the LAST queue launch so a previous invocation's failures are not re-counted
L=/tmp/e66_slice.log
awk '/^=== E66 QUEUE START/{buf=""} {buf=buf $0 ORS} END{printf "%s", buf}' "$LFULL" > "$L" 2>/dev/null || cp "$LFULL" "$L
R=/home/josh/lerobot/outputs/train/libero_10_seq10_naive_lora_r1216_a304_paramatched_steps5k
M=/home/josh/lerobot/outputs/analysis/e65_rematrix/mse_matrix_e63_seq10_FIXED.jsonl
u=$(systemctl is-active e66 2>/dev/null); true
e63rows=$(grep -c '^{' $M 2>/dev/null); e63rows=${e63rows:-0}
e63tasks=$(python3 -c "import json;print(len(json.loads(open('$M').readline())['per_task']))" 2>/dev/null); e63tasks=${e63tasks:-0}
smoke=$(grep -c "smoke OK" $L 2>/dev/null); smoke=${smoke:-0}
smokefail=$(grep -c "E66-SMOKE-FAIL" $L 2>/dev/null); smokefail=${smokefail:-0}
rung=$(grep -oE "bs[0-9]+ x acc[0-9]+" $L 2>/dev/null | tail -1 | tr -d ' ')
demote=$(grep -c "treating as VRAM" $L 2>/dev/null); demote=${demote:-0}
ck=$(ls -d $R/checkpoints/[0-9]* 2>/dev/null | wc -l)
step=$(grep -oE "step:[0-9]+K?" $L 2>/dev/null | tail -1 | cut -d: -f2)
stage=$(grep -oE "^\[e66\] [a-zA-Z0-9 -]+" $L 2>/dev/null | tail -1 | sed 's/^\[e66\] //' | cut -c1-22 | tr ' ' '_')
err=$(grep -cE "Traceback|OutOfMemoryError|^ERROR|E66-RUN-FAIL" $L 2>/dev/null); err=${err:-0}
fin=$(grep -c "E66-DONE" $L 2>/dev/null); fin=${fin:-0}
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "unit=$u e63=${e63rows}rows/${e63tasks}tasks smoke=$smoke(fail=$smokefail) rung=${rung:-none} demote=$demote seq=$ck/10 step=${step:-0} last=${stage:-none} disk=${dk}% err=$err fin=$fin gpu=${gpu:-NA}"
REMOTE
}
key_of(){ sed -E 's/ step=[0-9]*K?//; s/ gpu=[^ ]*//' <<<"$1"; }
field(){ sed -nE "s/.*(^| )$1=([^ ]+).*/\2/p" <<<"$2"; }

recover(){
  emit "RECOVERY: probing VM via nebius API"
  local st; st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
  emit "RECOVERY: state=${st:-UNKNOWN}"
  if [ "$st" != "RUNNING" ]; then
    for a in 1 2 3 4 5; do
      "$NEBIUS" compute instance start --id "$VM_ID" >/dev/null 2>&1 && { emit "RECOVERY: start issued (attempt $a)"; break; }
      emit "RECOVERY: start failed (attempt $a)"; sleep $((a*30))
    done
    for _ in $(seq 1 60); do
      st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
      [ "$st" = "RUNNING" ] && break; sleep 10
    done
  fi
  for _ in $(seq 1 30); do ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null && break; sleep 10; done
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null || { emit "RECOVERY FAILED: sshd never came up"; return 1; }
  emit "RECOVERY: SSH back up"
  local u; u=$(ssh -o BatchMode=yes "$VM" "systemctl is-active $UNIT 2>/dev/null; true")
  if [ "$u" != "active" ] && [ "$u" != "activating" ]; then
    ssh -o BatchMode=yes "$VM" "sudo systemctl reset-failed $UNIT 2>/dev/null; sudo systemd-run --unit=$UNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot /bin/bash -c 'bash scripts/ops/queue_e66_paramatched.sh >> $LOG 2>&1'" >/dev/null 2>&1 \
      && emit "RECOVERY: $UNIT relaunched (E63 matrix skip-guarded; the sequential self-resumes from its last task boundary)" \
      || emit "RECOVERY FAILED: could not relaunch $UNIT"
  else emit "RECOVERY: $UNIT already $u"; fi
}

if [ "$ONESHOT" = "1" ]; then
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then emit "$s"; else emit "VM UNREACHABLE"; fi; exit 0
fi

prev_key=""; prev_err=0; prev_demote=0; i=0; unreach=0
emit "heartbeat-E66 armed: unit $UNIT; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/60))min"
while true; do
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then
    [ "$unreach" -gt 0 ] && { emit "VM reachable again after $unreach failed poll(s) | $s"; unreach=0; prev_key=""; }
    key="$(key_of "$s")"
    err=$(field err "$s"); disk=$(field disk "$s" | tr -dc '0-9'); unit=$(field unit "$s")
    fin=$(field fin "$s"); sf=$(field smoke "$s"); dem=$(field demote "$s")
    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $s"
    [ "${dem:-0}" -gt "${prev_demote:-0}" ] 2>/dev/null && emit "LADDER DEMOTION (VRAM) — microbatch dropped, effective batch still 32 | $s"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $s"
    case "$sf" in *"fail=1"*|*"fail=2"*) emit "E66 SMOKE FAILED — full run not launched | $s";; esac
    if [ "${fin:-0}" -ge 1 ] 2>/dev/null; then emit "E66 CHAIN COMPLETE | $s"; exit 0
    elif [ "$unit" != "active" ] && [ "$unit" != "activating" ]; then emit "UNIT DOWN with the chain incomplete | $s"; fi
    { [ "$key" != "$prev_key" ] || [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; } && { emit "$s"; prev_key="$key"; }
    prev_err=${err:-0}; prev_demote=${dem:-0}
  else
    unreach=$((unreach+1))
    if nc -z -w5 github.com 22 2>/dev/null; then
      emit "VM UNREACHABLE (poll $unreach) but github:22 reachable => VM-side"
      [ "$unreach" -ge 2 ] && recover
    else emit "VM unreachable (poll $unreach) AND github:22 unreachable => LOCAL outage; no action"; fi
  fi
  i=$((i+1)); sleep "$POLL"
done
