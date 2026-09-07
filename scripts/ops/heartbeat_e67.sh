#!/usr/bin/env bash
# E67 heartbeat — local watcher for the two E67 units on nebius-spot:
#   e67-train : RETAIN chain (10 x 5k full FT + merge) then O-LoRA chain (10 x 5k)
#   e67-eval  : retention-triangle rows as boundaries land (VRAM-gated), then the drift matrices
# Same rules as heartbeat_e66.sh: artifact state, one multiplexed SSH connection (CLAUDE.md 9.5.1),
# preemption cross-checked against github:22 before recovery, relaunch BOTH units (each resumes
# from its own on-disk state). ONESHOT=1 prints one line and exits.
set -uo pipefail
VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00hks7a4fq3atcpsm
TLOG=/home/josh/lerobot/outputs/e67/train.log
ELOG=/home/josh/lerobot/outputs/e67/eval.log
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-6}
DISK_TRIPWIRE=${DISK_TRIPWIRE:-90}
ONESHOT=${ONESHOT:-0}
ts(){ date -u +%H:%MZ; }
emit(){ echo "[$(ts)] $*"; }
remote_state(){
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" 'bash -s' <<'REMOTE'
R=/home/josh/lerobot
TLOG=$R/outputs/e67/train.log; ELOG=$R/outputs/e67/eval.log
L=/tmp/e67_train_slice.log
awk '/^=== E67 TRAIN QUEUE START/{buf=""} {buf=buf $0 ORS} END{printf "%s", buf}' "$TLOG" > "$L" 2>/dev/null || cp "$TLOG" "$L" 2>/dev/null
ut=$(systemctl is-active e67-train 2>/dev/null); ue=$(systemctl is-active e67-eval 2>/dev/null); true
RET=$R/outputs/train/libero_10_seq10_retain_a05_fullft_steps5k; OLO=$R/outputs/train/libero_10_seq10_olora_r64_a16_lam05_steps5k
rt=$(python3 -c "import json;print(json.load(open('$RET/retain_state/progress.json'))['completed_tasks'])" 2>/dev/null); rt=${rt:-0}
ot=$(python3 -c "import json;print(json.load(open('$OLO/olora_state/progress.json'))['completed_tasks'])" 2>/dev/null); ot=${ot:-0}
step=$(grep -oE "task [0-9]+ step [0-9]+/[0-9]+" $L 2>/dev/null | tail -1 | sed 's/task //; s/ step /:/')
sps=$(grep -oE "[0-9.]+s/step" $L 2>/dev/null | tail -1)
orth=$(grep -oE "orth [0-9.e+-]+" $L 2>/dev/null | tail -1 | cut -d' ' -f2)
ck=$(grep -oE "state written in [0-9.]+s" $L 2>/dev/null | tail -1 | grep -oE "[0-9.]+s")
smk=$(ls $R/outputs/e67/smoke_retain_ok $R/outputs/e67/smoke_olora_ok 2>/dev/null | wc -l)
tri_r=$(ls $R/outputs/analysis/e67/seeds_tri_retain10_a05_b*.json 2>/dev/null | wc -l)
tri_o=$(ls $R/outputs/analysis/e67/seeds_tri_olora10_r64_b*.json 2>/dev/null | wc -l)
m_r=$(grep -c '^{' $R/outputs/analysis/e67/mse_matrix_retain10_a05.jsonl 2>/dev/null); m_r=${m_r:-0}
m_o=$(grep -c '^{' $R/outputs/analysis/e67/mse_matrix_olora10_r64.jsonl 2>/dev/null); m_o=${m_o:-0}
err=$(grep -cE "Traceback|OutOfMemoryError|^ERROR|E67-[A-Z-]*FAIL|\[FAIL\]|exactness check failed|frozen adapter .* changed" $L $ELOG 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
fin=$(grep -c "E67-TRAIN-DONE" $L 2>/dev/null); fin=${fin:-0}
efin=$(grep -c "E67-EVAL-DONE" $ELOG 2>/dev/null); efin=${efin:-0}
paused=$(grep -c "E67-TRAIN-PAUSED" $L 2>/dev/null); paused=${paused:-0}
last=$(grep -oE "^\[e67-train\] [a-zA-Z0-9 ().-]+" $L 2>/dev/null | tail -1 | sed 's/^\[e67-train\] //' | cut -c1-28 | tr ' ' '_')
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "train=$ut eval=$ue smoke=$smk/2 retain=$rt/10 olora=$ot/10 at=${step:-0} ${sps:-} orth=${orth:-} ck=${ck:-} tri=r${tri_r}/o${tri_o} mat=r${m_r}/o${m_o} last=${last:-none} disk=${dk}% err=$err paused=$paused fin=$fin/$efin gpu=${gpu:-NA}"
REMOTE
}
key_of(){ sed -E 's/ at=[^ ]*//; s/ [0-9.]+s\/step//; s/ orth=[^ ]*//; s/ ck=[^ ]*//; s/ gpu=[^ ]*//' <<<"$1"; }
field(){ sed -nE "s/.*(^| )$1=([^ ]+).*/\2/p" <<<"$2"; }
launch_units(){
  ssh -o BatchMode=yes "$VM" "sudo systemctl reset-failed e67-train e67-eval 2>/dev/null; \
    systemctl is-active e67-train >/dev/null 2>&1 || sudo systemd-run --unit=e67-train --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot /bin/bash -c 'bash scripts/ops/queue_e67_train.sh full >> $TLOG 2>&1'; \
    systemctl is-active e67-eval  >/dev/null 2>&1 || sudo systemd-run --unit=e67-eval  --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot /bin/bash -c 'bash scripts/ops/queue_e67_eval.sh >> $ELOG 2>&1'" >/dev/null 2>&1
}
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
  launch_units && emit "RECOVERY: e67-train / e67-eval relaunched (both resume from on-disk state)" || emit "RECOVERY FAILED: could not relaunch"
}
if [ "$ONESHOT" = "1" ]; then
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then emit "$s"; else emit "VM UNREACHABLE"; fi; exit 0
fi
prev_key=""; prev_err=0; i=0; unreach=0
emit "heartbeat-E67 armed: units e67-train + e67-eval; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/60))min"
while true; do
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then
    [ "$unreach" -gt 0 ] && { emit "VM reachable again after $unreach failed poll(s) | $s"; unreach=0; prev_key=""; }
    key="$(key_of "$s")"
    err=$(field err "$s"); disk=$(field disk "$s" | tr -dc '0-9'); ut=$(field train "$s"); ue=$(field eval "$s")
    fin=$(field fin "$s"); paused=$(field paused "$s")
    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $s"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $s"
    case "$fin" in 1/1*) emit "E67 COMPLETE (train + eval) | $s"; exit 0;; esac
    if [ "$ut" != "active" ] && [ "$ut" != "activating" ] && [ "${fin%%/*}" = "0" ]; then
      if [ "${paused:-0}" -ge 1 ]; then emit "train unit exited after a preemption save - relaunching | $s"; launch_units
      else emit "TRAIN UNIT DOWN with the chain incomplete (not a clean pause) | $s"; fi
    fi
    if [ "$ue" != "active" ] && [ "$ue" != "activating" ] && [ "${fin##*/}" = "0" ]; then emit "EVAL UNIT DOWN with rows incomplete | $s"; fi
    { [ "$key" != "$prev_key" ] || [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; } && { emit "$s"; prev_key="$key"; }
    prev_err=${err:-0}
  else
    unreach=$((unreach+1))
    if nc -z -w5 github.com 22 2>/dev/null; then
      emit "VM UNREACHABLE (poll $unreach) but github:22 reachable => VM-side"
      [ "$unreach" -ge 2 ] && recover
    else emit "VM unreachable (poll $unreach) AND github:22 unreachable => LOCAL outage; no action"; fi
  fi
  i=$((i+1)); sleep "$POLL"
done
