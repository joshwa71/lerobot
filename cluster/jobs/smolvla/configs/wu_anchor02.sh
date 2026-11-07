# Router warm-up sweep arm: expert anchor weight 0.4 -> 0.2, everything else as wu_ref.
source "$REPO/cluster/jobs/smolvla/configs/wu_common.sh"
NAME=wu_anchor02
MEMORY+=(--policy.memory_layer.expert_anchor_weight=0.2)
