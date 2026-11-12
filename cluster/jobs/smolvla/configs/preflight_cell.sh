# Preflight of cell_s2_seq5.sh at full size: the real layout, full banks and real batch sizes for a few steps
# of every stage, on the smallest GPUs the chain may get (48 GB), so the A-phase (bs32, values + Adam moments
# for 7 full tables) and the sequential stage are known to fit before the chain reaches them. Stages on
# libero_10 (~15 min to stage instead of libero_90's ~75); memory per step does not depend on the dataset.
#   rm -rf /SAN/vision/jo71_vla_wd/runs/preflight_cell && cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/preflight_cell.sh warmup audit aphase seq tri
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_seq5.sh"
NAME=preflight_cell
GPUS="gpu=true,gpu_type=(a40|a6000|l40s|rtx6000ada)"   # the 48 GB cards only
RES_WARMUP="tmem=64G,h_rt=4:00:00,tscratch=80G"
RES_AUDIT="tmem=48G,h_rt=4:00:00,tscratch=60G"
RES_APHASE="tmem=64G,h_rt=4:00:00,tscratch=100G"
RES_SEQ="tmem=64G,h_rt=4:00:00,tscratch=80G"
RES_TRI="tmem=48G,h_rt=4:00:00,tscratch=40G"
RUN_MARGIN_MIN=15

PRETRAIN_DATASET=libero_10
SEQ_TASKS='[0,1]'
AUDIT_TASKS='[0,1]'
WARMUP_STEPS=30
WARMUP_ARGS+=(--save_freq=15 --log_freq=5)
AUDIT_ARGS+=(--online_steps_per_task=5)
APHASE_STEPS=30
APHASE_ARGS+=(--save_freq=15 --log_freq=5)
SEQ_STEPS_PER_TASK=20
SEQ_ARGS+=(--log_freq=5)
EVAL_SEEDS=(1000)
EVAL_ARGS+=(--eval.n_episodes=2 --eval.batch_size=2)
