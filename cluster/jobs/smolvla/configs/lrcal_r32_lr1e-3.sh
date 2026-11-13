# LoRA learning-rate calibration (research_log2 Entry 4): rank 32, alpha = r, LR 1e-3; specialists on e4 and e1 (task_index
# 0 and 7), evaluated on seeds 5000/6000, outside the headline instrument's 1000-4000.
RANK=32 LR=1e-3
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lrcal_r32_lr1e-3
SEQ_TASKS='[0,7]'
EVAL_SEEDS=(5000 6000)
