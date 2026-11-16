# Multitask LoRA, rank 128 (alpha = r), rerun at LR 1e-4 (pi05's multitask LR): at 1e-3 training stalled at loss ~0.4
# with gradient spikes (research_log2 Entry 4 addendum). One adapter on all ten tasks jointly, 50k steps.
RANK=128 LR=1e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lora_multitask_r128_lr1e-4
EVAL_FROM=joint
