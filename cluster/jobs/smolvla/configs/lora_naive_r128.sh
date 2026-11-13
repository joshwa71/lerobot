# Naive sequential LoRA, rank 128 (alpha = r, LR 1e-3): one adapter through the five-task order (research_log2 Entry 4).
RANK=128 LR=1e-3
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lora_naive_r128_seq5
SEQ_INIT=$STAGE1_CKPT
SEQ_TASKS='[0,1,2,3,4]'
