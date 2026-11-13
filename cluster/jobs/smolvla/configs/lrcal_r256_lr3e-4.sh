# LoRA learning-rate calibration (research_log2 Entry 4): rank 256, alpha = r, LR 3e-4; specialists on e4 and e1 (task_index
# 0 and 7), evaluated on seeds 5000/6000, outside the headline instrument's 1000-4000.
RANK=256 LR=3e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lrcal_r256_lr3e-4
SEQ_TASKS='[0,7]'
EVAL_SEEDS=(5000 6000)
