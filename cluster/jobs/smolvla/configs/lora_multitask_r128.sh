# Multitask LoRA, rank 128 (alpha = r, LR 1e-3): one adapter on all ten tasks jointly, 50k steps (research_log2 Entry 4).
RANK=128 LR=1e-3
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lora_multitask_r128
EVAL_FROM=joint
