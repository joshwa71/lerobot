# Router warm-up sweep arm: inter-task separation weight 8 -> 32, everything else as wu_ref.
source "$REPO/cluster/jobs/smolvla/configs/wu_common.sh"
NAME=wu_sep32
WARMUP_ARGS+=(--policy.memory_layer.routing_inter_task_separation_weight=32.0)
