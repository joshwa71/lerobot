#!/usr/bin/env bash
# E64 heartbeat — artifact-state watcher for the `e64-lora-r512` queue on the spot VM.
#
# Design follows the standing rules (E63 addendum 5):
#   - reports ARTIFACT STATE (checkpoint/JSON counts, unit state, disk, error count),
#     never "the last line that matched";
#   - every glob used as a counter was verified against a real path before arming
#     (6-char ckpt glob -> 10 on the seq10 run; specialist glob -> 10 on the r32 dirs;
#      seeds glob -> 10 on the r32 spec rows);
#   - one SSH connection is reused (ControlMaster in ~/.ssh/config) so polling does not
#     churn login sessions (CLAUDE.md 9.5.1, RemoveIPC).
#
# Emits one line when the discrete state key CHANGES, plus a forced heartbeat every
# HEARTBEAT_EVERY polls so silence never looks like success. Failure signatures
# (unit died, new Traceback/OOM, disk tripwire, unreachable host) always emit.
#
# Preemption handling (CLAUDE.md 9.2/9.3): SSH failure is cross-checked against an
# independent host before concluding anything (the 1 Aug lesson: a local network
# outage looks identical to a preemption). If the VM is genuinely down it is started
# via the local nebius CLI with backoff, then the queue unit is relaunched — the
# runner is relaunch-safe (stages 1-2 skip finished runs; stage 3 resumes).
set -uo pipefail

VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"          # NOT on PATH in non-login shells
VM_ID=computeinstance-e00hks7a4fq3atcpsm
RUNNER=/home/josh/lerobot/scripts/vla_analysis/run_e64_lora_r512_queue.sh
UNIT=e64-lora-r512
POLL=${POLL:-600}                          # 10 min
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-36}     # 36 x 10 min = 6 h forced emit
DISK_TRIPWIRE=${DISK_TRIPWIRE:-88}

ts() { date -u +%H:%MZ; }
emit() { echo "[$(ts)] $*"; }

remote_state() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" 'bash -s' <<'REMOTE'
R=/home/josh/lerobot
L=$R/outputs/e64_lora_r512.log
u=$(systemctl is-active e64-lora-r512 2>/dev/null); true
tri_u=$(systemctl is-active e64-triangles 2>/dev/null); true
mt=$(ls -d $R/outputs/train/loraft_multitask10_r512_50k/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | wc -l)
step=$(grep -oE "[0-9]+/50000" $L 2>/dev/null | tail -1 | cut -d/ -f1)
sp=$(ls -d $R/outputs/train/loraft_baseline_r512/task*/checkpoints/005000 2>/dev/null | wc -l)
nv=$(ls -d $R/outputs/train/libero_10_seq10_naive_lora_r512_a128_steps5k/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | wc -l)
sd=$(ls $R/outputs/analysis/e60/seeds_multitask10_r512.json \
        $R/outputs/analysis/e60/seeds_spec_r512_e*.json \
        $R/outputs/analysis/e60/seeds_naive10_r512_final.json 2>/dev/null | wc -l)
trin=$(ls $R/outputs/analysis/e60/seeds_tri_naive10_r512_b*.json 2>/dev/null | wc -l)
trim=$(ls $R/outputs/analysis/e60/seeds_tri_merged6x2_10task_b*.json 2>/dev/null | wc -l)
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
er=$(grep -cE "Traceback|OutOfMemoryError|CUDA out of memory|\[FAIL\]" $L 2>/dev/null)
done_marker=$(grep -c "QUEUE COMPLETE" $L 2>/dev/null)
tri_done=$(grep -c "TRIANGLES COMPLETE" $R/outputs/e64_triangles.log 2>/dev/null)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "unit=$u tri_unit=$tri_u mt=$mt/5 step=${step:-0} spec=$sp/10 naive=$nv/10 seeds=$sd/12 tri=$trin/10+$trim/10 disk=${dk}% err=$er fin=$done_marker tri_fin=$tri_done gpu=${gpu:-NA}"
REMOTE
}

# state key for change detection: everything except the fast-moving step/gpu fields
key_of() { sed -E 's/ step=[0-9]+//; s/ gpu=[^ ]*//' <<<"$1"; }

recover() {
  emit "RECOVERY: probing VM state via nebius API"
  local st ip
  st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
  emit "RECOVERY: API reports state=${st:-UNKNOWN}"
  [ "$st" = "RUNNING" ] && { emit "RECOVERY: already RUNNING — waiting for sshd"; }
  if [ "$st" != "RUNNING" ]; then
    for a in 1 2 3 4 5; do
      "$NEBIUS" compute instance start --id "$VM_ID" >/dev/null 2>&1 && { emit "RECOVERY: start issued (attempt $a)"; break; }
      emit "RECOVERY: start failed (attempt $a) — backing off $((a*30))s"
      sleep $((a*30))
    done
    for _ in $(seq 1 60); do
      st=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.state' 2>/dev/null)
      [ "$st" = "RUNNING" ] && break
      sleep 10
    done
    emit "RECOVERY: state=${st:-UNKNOWN} after start poll"
    [ "$st" = "RUNNING" ] || { emit "RECOVERY FAILED: VM did not reach RUNNING — manual intervention needed"; return 1; }
  fi
  ip=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null \
       | jq -r '.status.network_interfaces[0].public_ip_address.address' 2>/dev/null | cut -d/ -f1)
  local cfg_ip; cfg_ip=$(awk '/Host nebius-spot/{f=1} f&&/HostName/{print $2; exit}' ~/.ssh/config)
  [ -n "$ip" ] && [ "$ip" != "$cfg_ip" ] && emit "RECOVERY WARNING: public IP $ip != ssh config $cfg_ip — update ~/.ssh/config"
  for _ in $(seq 1 30); do
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null && break
    sleep 10
  done
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null || { emit "RECOVERY FAILED: RUNNING but sshd never came up"; return 1; }
  emit "RECOVERY: SSH back up"
  local u; u=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" 'systemctl is-active '"$UNIT"' 2>/dev/null; true')
  if [ "$u" != "active" ] && [ "$u" != "activating" ]; then
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "sudo systemctl reset-failed $UNIT 2>/dev/null; sudo systemd-run --unit=$UNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot /bin/bash $RUNNER" >/dev/null 2>&1 \
      && emit "RECOVERY: $UNIT relaunched (stages 1-2 restart partial runs from scratch; stage 3 resumes)" \
      || emit "RECOVERY FAILED: could not relaunch $UNIT"
  else
    emit "RECOVERY: $UNIT already $u — nothing to relaunch"
  fi
}

prev_key=""
prev_err=0
i=0
unreachable_run=0
emit "heartbeat armed: poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/3600))h, disk tripwire ${DISK_TRIPWIRE}%"
while true; do
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then
    if [ "$unreachable_run" -gt 0 ]; then
      emit "VM reachable again after $unreachable_run failed poll(s) | $state"
      unreachable_run=0; prev_key=""      # force a fresh state line
    fi
    key=$(key_of "$state")
    err=$(sed -E 's/.* err=([0-9]+) .*/\1/' <<<"$state")
    disk=$(sed -E 's/.* disk=([0-9]+)%.*/\1/' <<<"$state")
    unit=$(sed -E 's/^unit=([a-z]+) .*/\1/' <<<"$state")
    tri_unit=$(sed -E 's/.* tri_unit=([a-z]+) .*/\1/' <<<"$state")
    tri_fin=$(sed -E 's/.* tri_fin=([0-9]+) .*/\1/' <<<"$state")
    fin=$(sed -E 's/.* fin=([0-9]+) .*/\1/' <<<"$state")

    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $state"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% >= ${DISK_TRIPWIRE}% | $state"

    if [ "$unit" != "active" ] && [ "$unit" != "activating" ] \
       && [ "$tri_unit" != "active" ] && [ "$tri_unit" != "activating" ]; then
      if [ "${fin:-0}" -ge 1 ] 2>/dev/null && [ "${tri_fin:-0}" -ge 1 ] 2>/dev/null; then
        emit "QUEUE + TRIANGLES COMPLETE | $state"; exit 0
      fi
      emit "BOTH UNITS DOWN with work incomplete | $state"
      sleep "$POLL"; i=$((i+1)); prev_err=${err:-0}; prev_key="$key"; continue
    fi

    if [ "$key" != "$prev_key" ] || [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; then
      emit "$state"
      prev_key="$key"
    fi
    prev_err=${err:-0}
  else
    unreachable_run=$((unreachable_run+1))
    if nc -z -w5 github.com 22 2>/dev/null; then
      emit "VM UNREACHABLE (poll $unreachable_run) but github:22 IS reachable => VM-side, not local network"
      [ "$unreachable_run" -ge 2 ] && recover
    else
      emit "VM unreachable (poll $unreachable_run) AND github:22 unreachable => LOCAL network outage; no action (1 Aug precedent)"
    fi
  fi
  i=$((i+1))
  sleep "$POLL"
done
