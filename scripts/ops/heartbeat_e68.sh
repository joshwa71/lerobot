#!/usr/bin/env bash
# E68 heartbeat — local watcher for unit e68-train on nebius2 (josh-vm-spot-2, H100, preemptible).
# Same rules as heartbeat_e67.sh: artifact state (not the last log line), one multiplexed SSH
# connection per poll (CLAUDE.md 9.5.1), preemption cross-checked against github:22 before recovery,
# relaunch the unit (every stage resumes from on-disk state). ONESHOT=1 prints one line and exits.
set -uo pipefail
VM=nebius2
NEBIUS="$HOME/.nebius/bin/nebius"
VM_ID=computeinstance-e00h8htkzavxm81d24
TLOG=/home/josh/lerobot/outputs/e68/train.log
POLL=${POLL:-600}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-6}
DISK_TRIPWIRE=${DISK_TRIPWIRE:-90}
ONESHOT=${ONESHOT:-0}
STAGES=${STAGES:-"a1 a2 c1 c2"}
ts(){ date -u +%H:%MZ; }
emit(){ echo "[$(ts)] $*"; }
remote_state(){
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM" 'bash -s' <<'REMOTE'
R=/home/josh/lerobot; T=$R/outputs/train; TLOG=$R/outputs/e68/train.log
L=/tmp/e68_train_slice.log
awk '/^=== E68 TRAIN QUEUE START/{buf=""} {buf=buf $0 ORS} END{printf "%s", buf}' "$TLOG" > "$L" 2>/dev/null || cp "$TLOG" "$L" 2>/dev/null
u=$(systemctl is-active e68-train 2>/dev/null); true
A1A=$T/libero_90_pi05_jointA10k_e68a1_live_merged6x2_e468101416_v579111315_anchor040_sep8
A1S=$T/libero_10_seq5_jw_e68a1_live_merged6x2_e468101416_v579111315_beta4corefrac_topt3072_lr2x_steps5k
A2S=$T/libero_10_seq5_jw_e68a2_tfidfonly_merged6x2_e468101416_v579111315_prepass_noprotect_topt3072_lr2x_steps5k
A3J=$T/libero_90_pi05_jointA10k_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
A3S=$T/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
C1=$T/audit_heldout_jointwarm_e68c1_merged6x2_e468101416_v579111315_anchor040_sep0_c005_prepass_10k
C2=$T/audit_heldout_jointwarm_e68c2_merged6x2_e468101416_v579111315_anchor040_sep8_c0_prepass_10k
nck(){ ls -d "$1"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | wc -l; }
a1="$(nck $A1A)/$(nck $A1S)"; a2="$(nck $A2S)"; a3="$(nck $A3J)/$(nck $A3S)"
c1=$(ls $C1/memory_by_task/*.json 2>/dev/null | wc -l); c2=$(ls $C2/memory_by_task/*.json 2>/dev/null | wc -l)
[ -f $C1/expert_audit_summary.json ] && c1="${c1}+cert"; [ -f $C2/expert_audit_summary.json ] && c2="${c2}+cert"
smk=$(ls $R/outputs/e68/smoke_live_ok $R/outputs/e68/smoke_resume_ok 2>/dev/null | wc -l)
stage=$(grep -oE "^\[e68-train\] stage [a-z0-9]+" $L 2>/dev/null | tail -1 | awk '{print $3}')
step=$(grep -oE "(step:[0-9]+|Online task [0-9]+/[0-9]+|step [0-9]+/[0-9]+)" $L 2>/dev/null | tail -1 | tr ' ' '_')
sps=$(grep -oE "updt_s:[0-9.]+" $L 2>/dev/null | tail -1)
loss=$(grep -oE "loss:[0-9.]+" $L 2>/dev/null | tail -1)
rung=$(grep -oE "rung: bs=[0-9]+ accum=[0-9]+ grad_ckpt=[a-z]+" $L 2>/dev/null | tail -1 | tr ' ' '_')
err=$(grep -cE "Traceback|OutOfMemoryError|^ERROR|E68-[A-Z0-9-]*FAIL|\[FAIL\]" $L 2>/dev/null); err=${err:-0}
done_m=$(grep -oE "E68-[A-Z0-9]+-DONE" $L 2>/dev/null | sort -u | tr '\n' ',' | sed 's/E68-//g; s/-DONE//g')
fin=$(grep -c "E68-TRAIN-DONE" $L 2>/dev/null); fin=${fin:-0}
dk=$(df --output=pcent /home/josh | tail -1 | tr -dc '0-9')
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
lg=$(loginctl show-user josh -p Linger 2>/dev/null | cut -d= -f2)
echo "unit=$u smoke=$smk/2 stage=${stage:-none} ${rung:-} at=${step:-0} ${sps:-} ${loss:-} a1=$a1 a2=$a2 a3=$a3 c1=$c1 c2=$c2 done=${done_m:-none} disk=${dk}% gpu=${gpu:-NA} linger=${lg:-?} err=$err fin=$fin"
REMOTE
}
key_of(){ sed -E 's/ at=[^ ]*//; s/ updt_s:[^ ]*//; s/ loss:[^ ]*//; s/ gpu=[^ ]*//' <<<"$1"; }
field(){ sed -nE "s/.*(^| )$1=([^ ]+).*/\2/p" <<<"$2"; }
launch_unit(){
  ssh -o BatchMode=yes "$VM" "sudo systemctl reset-failed e68-train 2>/dev/null; \
    systemctl is-active e68-train >/dev/null 2>&1 || sudo systemd-run --unit=e68-train --property=User=josh --property=KillSignal=SIGTERM --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot --setenv=PYTHONUNBUFFERED=1 --setenv=STAGES='$STAGES' /bin/bash -c 'bash scripts/ops/queue_e68_train.sh full >> $TLOG 2>&1'" >/dev/null 2>&1
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
    ip=$("$NEBIUS" compute instance get --id "$VM_ID" --format json 2>/dev/null | jq -r '.status.network_interfaces[0].public_ip_address.address' 2>/dev/null | cut -d/ -f1)
    cfg_ip=$(awk '/^Host nebius2$/{f=1} f&&/HostName/{print $2; exit}' ~/.ssh/config)
    [ -n "$ip" ] && [ "$ip" != "$cfg_ip" ] && emit "RECOVERY: PUBLIC IP CHANGED $cfg_ip -> $ip (update ~/.ssh/config Host nebius2)"
  fi
  for _ in $(seq 1 30); do ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null && break; sleep 10; done
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM" true 2>/dev/null || { emit "RECOVERY FAILED: sshd never came up"; return 1; }
  emit "RECOVERY: SSH back up"
  launch_unit && emit "RECOVERY: e68-train relaunched (resumes from on-disk state)" || emit "RECOVERY FAILED: could not relaunch"
}
if [ "$ONESHOT" = "1" ]; then
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then emit "$s"; else emit "VM UNREACHABLE"; fi; exit 0
fi
prev_key=""; prev_err=0; i=0; unreach=0
emit "heartbeat-E68 armed: unit e68-train on $VM; poll ${POLL}s, forced beat every $((POLL*HEARTBEAT_EVERY/60))min"
while true; do
  if s=$(remote_state 2>/dev/null) && [ -n "$s" ]; then
    [ "$unreach" -gt 0 ] && { emit "VM reachable again after $unreach failed poll(s) | $s"; unreach=0; prev_key=""; }
    key="$(key_of "$s")"
    err=$(field err "$s"); disk=$(field disk "$s" | tr -dc '0-9'); u=$(field unit "$s"); fin=$(field fin "$s")
    [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null && emit "NEW ERROR LINES (err ${prev_err}->${err}) | $s"
    [ "${disk:-0}" -ge "$DISK_TRIPWIRE" ] 2>/dev/null && emit "DISK TRIPWIRE ${disk}% | $s"
    [ "${fin:-0}" -ge 1 ] && { emit "E68 TRAIN QUEUE COMPLETE | $s"; exit 0; }
    if [ "$u" != "active" ] && [ "$u" != "activating" ]; then
      if [ "${err:-0}" -gt "${prev_err:-0}" ] 2>/dev/null || grep -q "FAIL" <<<"$s"; then emit "UNIT DOWN after a failure marker - NOT relaunching automatically | $s"
      else emit "unit down with the queue incomplete (clean exit / preemption) - relaunching | $s"; launch_unit; fi
    fi
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
