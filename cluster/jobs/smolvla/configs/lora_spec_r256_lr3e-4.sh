# LoRA specialists, rank 256 (alpha = r), rerun at LR 3e-4 on all ten tasks: at 1e-3 training diverged on 4 of 10 tasks
# (research_log2 Entry 4 addendum). Submit with SPEC_GROUP=5.
RANK=256 LR=3e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lora_spec_r256_lr3e-4
