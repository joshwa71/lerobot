#!/bin/bash
# E58 weekend eval-seeds campaign (discharges the E56-addendum TODO):
#   3 eval seeds (1000/2000/3000) x 100 eps/task x { B (spread frontier, 5 envs) +
#   the 5 LoRA specialists (own env each) } = 3,000 episodes, batched vec envs.
# GATED: waits for the e57-vnoise unit to finish before touching the GPU (local
# systemctl poll — no SSH sessions, no RemoveIPC churn).
#
# Design choices:
# - batch_size=10: exact divisor of 100 (no discarded episodes) and <=10 async env
#   workers + main stays inside the 16-vCPU budget. Init-state coverage at batch 10:
#   worker w sees init states {w, w+10, ...} -> 100 eps cover the 50-state set exactly
#   twice, uniformly (libero.py init_state_id stride = n_envs).
# - per-(model, seed, env) invocations: <=10 env workers alive at any time, each cell an
#   independent process with a skip guard -> one failure never cascades, relaunch is
#   idempotent (the spot-VM rule; if preempted, just relaunch this unit).
# - seed-major order: each pass over seeds is a COMPLETE replicate (B + all specialists),
#   so a partial weekend read has full replicates rather than a lopsided table.
# - seed=1000 subsumes the historical 50-ep finals' seed range; 2000/3000 are fresh.
# Stats note for the writeup: 100 eps/cell -> +/-4-5pp; 300 pooled/cell -> +/-2.6pp;
# 5-task mean se ~+/-1.2pp. LIBERO init states wrap mod 50, so extra episodes re-visit
# the fixed state set with fresh policy noise (the CI is over that distribution).
set -o pipefail
ROOT=/home/josh/lerobot
OUT=$ROOT/outputs/eval_seeds
SUMDIR=$ROOT/outputs/analysis/e58_evalseeds
LOG=$ROOT/outputs/e58_evalseeds.log
mkdir -p "$OUT" "$SUMDIR"
exec >> "$LOG" 2>&1
echo "=== E58 eval-seeds campaign started (waiting for e57-vnoise) $(date -u) ==="
while systemctl is-active --quiet e57-vnoise; do sleep 300; done
echo "=== e57-vnoise finished -> starting evals $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
BCKPT=$ROOT/outputs/train/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k/checkpoints/025000/pretrained_model

run_eval () {  # $1 tag  $2 ckpt  $3 env  $4 seed
  local D=$OUT/$1
  if [ -f "$D/eval_info.json" ]; then echo "[skip] $1 (exists)"; return 0; fi
  if [ ! -d "$2" ]; then echo "[FAIL] $1: checkpoint missing $2"; return 1; fi
  rm -rf "$D"
  echo "[run] $1 $(date -u)"
  lerobot-eval \
    --policy.path="$2" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[$3]" \
    --rename_map="$RENAME" \
    --eval.batch_size=10 --eval.n_episodes=100 \
    --seed=$4 \
    --output_dir="$D" \
    && echo "[done] $1 $(date -u)" \
    || echo "[FAIL] $1 (non-fatal, skip guard allows relaunch)"
}

declare -A SPEC=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 )
for SEED in 1000 2000 3000; do
  echo "=== replicate seed=$SEED $(date -u) ==="
  for ENV in 4 6 9 2 7; do
    run_eval "B_seed${SEED}_e${ENV}" "$BCKPT" $ENV $SEED
  done
  for T in 0 1 2 3 4; do
    ENV=${SPEC[$T]}
    run_eval "spec_t${T}_e${ENV}_seed${SEED}" \
      "$ROOT/outputs/train/loraft_baseline/task${T}_e${ENV}/checkpoints/005000/pretrained_model" \
      $ENV $SEED
  done
done

echo "=== summary $(date -u) ==="
python - <<'PYEOF'
import json, os, re, math
from collections import defaultdict

OUT = "/home/josh/lerobot/outputs/eval_seeds"
SUM = "/home/josh/lerobot/outputs/analysis/e58_evalseeds/summary.json"

def pc(d):
    """Find pc_success + n_episodes in an eval_info.json, defensively."""
    if isinstance(d, dict):
        if "pc_success" in d and "n_episodes" in d:
            return float(d["pc_success"]), int(d["n_episodes"])
        for v in d.values():
            r = pc(v)
            if r:
                return r
    return None

cells = defaultdict(dict)  # (model, env) -> {seed: pc}
for name in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, name, "eval_info.json")
    if not os.path.isfile(p):
        continue
    m = re.match(r"(B)_seed(\d+)_e(\d+)$", name) or \
        re.match(r"(spec)_t\d+_e(?:(\d+))_seed(\d+)$", name)
    if not m:
        continue
    if m.group(1) == "B":
        model, seed, env = "B", int(m.group(2)), int(m.group(3))
    else:
        model, env, seed = "spec", int(m.group(2)), int(m.group(3))
    r = pc(json.load(open(p)))
    if r:
        cells[(model, env)][seed] = r[0]

seeds = [1000, 2000, 3000]
envs = [4, 6, 9, 2, 7]
out = {"cells": {}, "means": {}}
print(f"{'model':<6}{'env':<5}" + "".join(f"s{s:<7}" for s in seeds) + "mean   sd")
for model in ("B", "spec"):
    for env in envs:
        d = cells.get((model, env), {})
        vals = [d[s] for s in seeds if s in d]
        row = "".join(f"{d.get(s, float('nan')):<8.1f}" for s in seeds)
        mu = sum(vals) / len(vals) if vals else float("nan")
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else float("nan")
        print(f"{model:<6}{env:<5}{row}{mu:<7.1f}{sd:.1f}" if vals else f"{model:<6}{env:<5}(no data)")
        out["cells"][f"{model}_e{env}"] = {"by_seed": d, "mean": mu, "sd": sd, "n_seeds": len(vals)}
    # per-seed 5-env averages -> replicate-level mean +/- sd
    reps = []
    for s in seeds:
        vs = [cells[(model, e)][s] for e in envs if s in cells.get((model, e), {})]
        if len(vs) == len(envs):
            reps.append(sum(vs) / len(vs))
    if reps:
        mu = sum(reps) / len(reps)
        sd = (sum((v - mu) ** 2 for v in reps) / max(1, len(reps) - 1)) ** 0.5 if len(reps) > 1 else float("nan")
        print(f"{model:<6}AVG  replicates {['%.1f' % r for r in reps]} -> {mu:.1f} +/- {sd:.1f}")
        out["means"][model] = {"replicates": reps, "mean": mu, "sd": sd}

json.dump(out, open(SUM, "w"), indent=1)
print(f"[summary] -> {SUM}")
PYEOF
echo "=== E58 eval-seeds campaign COMPLETE $(date -u) ==="
