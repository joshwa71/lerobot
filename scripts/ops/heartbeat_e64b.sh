#!/usr/bin/env bash
# E64b heartbeat — phase B watcher: the r128 specialist ladder on the VM AND the
# E64 r512 cold-storage ship on the desk PC. Successor to heartbeat_e64.sh, which
# exited on "QUEUE + TRIANGLES COMPLETE" (24 Aug 10:12 UTC).
#
# Same rules as its predecessor (E63 add-5 / E64 add-3):
#   - reports ARTIFACT STATE (dir/row counts, PASS lines, unit state, disk), never
#     "the last line that matched";
#   - every glob verified against a real path before arming;
#   - one SSH connection reused (ControlMaster) so polling does not churn login
#     sessions (CLAUDE.md 9.5.1, RemoveIPC);
#   - VM unreachable is cross-checked against an independent host before concluding
#     preemption (1 Aug precedent), then recovered via the local nebius CLI.
#
# Emits on discrete state change, plus a forced beat every HEARTBEAT_EVERY polls.
# Failure signatures always emit: unit died with work left, shipper process gone,
# FAIL-TRANSFER / FAIL-VERIFY in the archive status, disk tripwire, host unreachable.
set -uo pipefail

VM=nebius-spot
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00hks7a4fq3atcpsm
RUNNER=/home/josh/lerobot/scripts/vla_analysis/run_e64b_r128_after_triangles.sh
UNIT=e64b-r128
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e64_status.txt
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-36}     # 6 h forced emit
DISK_TRIPWIRE=${DISK_TRIPWIRE:-88}

ts() { date -u +%H:%MZ; }
emit() { echo "[$(ts)] $*"; }

remote_state() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" 'bash -s' <<'REMOTE'
R=/home/josh/lerobot
u=$(systemctl is-active e64b-r128 2>/dev/null); true
sp=$(ls -d $R/outputs/train/loraft_baseline_r128/task*/checkpoints/005000 2>/dev/null | wc -l)
sd=$(ls $R/outputs/analysis/e60/seeds_spec_r128_e*.json 2>/dev/null | wc -l)
step=$(grep -oE "step:[0-9]+K?" $R/outputs/e64b_r128.log 2>/dev/null | tail -1 | cut -d: -f2)
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
er=$(grep -cE "Traceback|OutOfMemoryError|CUDA out of memory|\[FAIL\]" $R/outputs/e64b_r128.log 2>/dev/null)
fin=$(grep -c "E64b r128 ladder COMPLETE" $R/outputs/e64b_r128.log 2>/dev/null)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "unit=$u spec128=$sp/10 rows128=$sd/10 step=${step:-0} disk=${dk}% err=$er fin=$fin gpu=${gpu:-NA}"
REMOTE
}

ship_state() {
  local alive pass fail
  pgrep -f "bash scripts/ops/ship_e64_batch_to_cold" >/dev/null && alive=yes || alive=NO
  pass=$(grep -c "^PASS " "$STATUS" 2>/dev/null); true
  fail=$(grep -cE "^FAIL" "$STATUS" 2>/dev/null); true
  pass=${pass:-0}; fail=${fail:-0}
  echo "ship=$alive pass=$pass/4 fail=$fail"
}

key_of() { sed -E 's/ step=[0-9]*K?//; s/ gpu=[^ ]*//' <<<"$1"; }

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
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$VM" "sudo systemctl reset-failed $UNIT 2>/dev/null; sudo systemd-run --unit=$UNIT --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot /bin/bash $RUNNER" >/dev/null 2>&1 \
      && emit "RECOVERY: $UNIT relaunched (in-flight r128 task restarts from scratch; finished tasks/rows skip-guarded)" \
      || emit "RECOVERY FAILED: could not relaunch $UNIT"
  else
    emit "RECOVERY: $UNIT already $u — nothing to relaunch"
  fi
}

prev_key=""; prev_err=0; prev_ship=""; prev_fail=0; i=0; unreachable_run=0
emit "heartbeat-B armed: r128 ladder + E64 cold ship; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/3600))h"
while true; do
  s_ship=$(ship_state)
  if state=$(remote_state 2>/dev/null) && [ -n "$state" ]; then
    if [ "$unreachable_run" -gt 0 ]; then
      emit "VM reachable again after $unreachable_run failed poll(s) | $state | $s_ship"
      unreachable_run=0; prev_key=""
    fi
    key="$(key_of "$state") $s_ship"
    err=$(sed -E 's/.* err=([0-9]+) .*/\1/' <<<"$state")
    disk=$(sed -E 's/.* disk=([0-9]+)%.*/\1/' <<<"$state")
    unit=$(sed -E 's/^unit=([a-z]+) .*/\1/' <<<"$state")
    fin=$(sed -E 's/.* fin=([0-9]+) .*/\1/' <<<"$state")
    rows=$(sed -E 's/.* rows128=([0-9]+)\/10 .*/\1/' <<<"$state")

    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $state"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $state"
    # edge-triggered: a standing FAIL line must not re-alarm every poll (24 Aug --
    # the spurious FAIL-VERIFY from the shipper's grep -c bug fired every 10 min).
    nfail=$(sed -E 's/.* fail=([0-9]+).*/\1/' <<<"$s_ship")
    [ "${nfail:-0}" -gt "${prev_fail:-0}" ] 2>/dev/null && emit "NEW ARCHIVE FAILURE (fail ${prev_fail:-0}->${nfail}) in $STATUS | $s_ship"
    prev_fail=${nfail:-0}
    [[ "$s_ship" == *"ship=NO"* ]] && [[ "$s_ship" != *"pass=4/4"* ]] && emit "SHIPPER PROCESS GONE with transfers incomplete | $s_ship"

    if [ "$unit" != "active" ] && [ "$unit" != "activating" ]; then
      if [ "${fin:-0}" -ge 1 ] 2>/dev/null || [ "${rows:-0}" -ge 10 ] 2>/dev/null; then
        emit "R128 LADDER COMPLETE | $state | $s_ship"
        [[ "$s_ship" == *"pass=4/4"* ]] && { emit "SHIP ALSO COMPLETE — exiting"; exit 0; }
      else
        emit "R128 UNIT DOWN with work incomplete | $state"
      fi
    fi

    if [ "$key" != "$prev_key" ] || [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; then
      emit "$state | $s_ship"; prev_key="$key"
    fi
    prev_err=${err:-0}
  else
    unreachable_run=$((unreachable_run+1))
    if nc -z -w5 github.com 22 2>/dev/null; then
      emit "VM UNREACHABLE (poll $unreachable_run) but github:22 reachable => VM-side | $s_ship"
      [ "$unreachable_run" -ge 2 ] && recover
    else
      emit "VM unreachable (poll $unreachable_run) AND github:22 unreachable => LOCAL network outage; no action (1 Aug precedent)"
    fi
  fi
  i=$((i+1)); sleep "$POLL"
done
