# Naive sequential LoRA, rank 128 (alpha = r), rerun at LR 3e-4: at 1e-3 each later task trained worse (end loss 0.05 ->
# 0.26; research_log2 Entry 4 addendum). One adapter through the five-task order.
RANK=128 LR=3e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=lora_naive_r128_seq5_lr3e-4
SEQ_INIT=$STAGE1_CKPT
SEQ_TASKS='[0,1,2,3,4]'
