# Contrastive-weight sweep arm (research_log2 Entry 3): separation 2 (the carried-forward setting), contrastive 0.05 -> 1.0,
# everything else as wu2_sep2 (which is the c=0.05 point). Warm-up + held-out audit only.
source "$REPO/cluster/jobs/smolvla/configs/wu_common.sh"
NAME=wu2_c1
WARMUP_ARGS+=(--policy.memory_layer.routing_inter_task_separation_weight=2.0 --policy.memory_layer.contrastive_loss_weight=1.0)
