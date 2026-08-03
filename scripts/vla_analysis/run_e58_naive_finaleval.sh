#!/bin/bash
# E58: standalone final 50-ep eval of the naive sequential LoRA r256 arm (025000).
# Replaces the trainer's in-run final eval, killed by preemption #2 (~10:22 UTC 3 Aug,
# ~1.5h into the serial eval). Batched (bs=10) per the E56-addendum validated pattern;
# seed 1000 matches the historical 50-ep finals. --policy.use_peft passed EXPLICITLY
# (the factory needs it to load base + adapter; do not rely on the checkpoint config).
# Waits for the e58-msemat unit to free the GPU. Skip-guards allow relaunch after a
# further preemption.
set -o pipefail
ROOT=/home/josh/lerobot
OUT=$ROOT/outputs/analysis/e58/naive_final_eval
LOG=$ROOT/outputs/e58_naive_finaleval.log
mkdir -p "$OUT"
exec >> "$LOG" 2>&1
echo "=== E58 naive-r256 final eval started (waiting for e58-msemat) $(date -u) ==="
while systemctl is-active --quiet e58-msemat; do sleep 60; done
echo "=== GPU free -> evals $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

CKPT=$ROOT/outputs/train/libero_10_seq5_naive_lora_r256_a64_steps5k/checkpoints/025000/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
[ -d "$CKPT" ] || { echo "FATAL: $CKPT missing"; exit 1; }

for ENV in 4 6 9 2 7; do
  D=$OUT/e${ENV}
  if [ -f "$D/eval_info.json" ]; then echo "[skip] e$ENV (exists)"; continue; fi
  rm -rf "$D"
  echo "[run] e$ENV $(date -u)"
  lerobot-eval \
    --policy.path="$CKPT" \
    --policy.use_peft=true \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[$ENV]" \
    --rename_map="$RENAME" \
    --eval.batch_size=10 --eval.n_episodes=50 \
    --seed=1000 \
    --output_dir="$D" \
    && echo "[done] e$ENV $(date -u)" \
    || echo "[FAIL] e$ENV (skip guard allows relaunch)"
done

echo "=== summary $(date -u) ==="
python - <<'PYEOF'
import json, os
OUT = "/home/josh/lerobot/outputs/analysis/e58/naive_final_eval"
rows = {}
for env in [4, 6, 9, 2, 7]:
    p = os.path.join(OUT, f"e{env}", "eval_info.json")
    if os.path.exists(p):
        with open(p) as f:
            info = json.load(f)
        agg = info.get("aggregated", {})
        rows[f"e{env}"] = agg.get("pc_success")
print("NAIVE r256 FINAL (50 eps, seed 1000):", json.dumps(rows))
vals = [v for v in rows.values() if v is not None]
if len(vals) == 5:
    print(f"MEAN: {sum(vals)/5:.1f}")
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(rows, f)
PYEOF
echo "=== E58 naive final eval COMPLETE $(date -u) ==="
