# External continual-learning baselines (E67): RETAIN and O-LoRA

Self-contained baselines run under the paper's continual protocol (stage-1 LIBERO-90 base, ten
LIBERO-10 tasks in dataset order, 5,000 steps per task, effective batch 32, no replay, no task
identity at inference, optimizer re-initialised per task, the 25-episode x 4-seed instrument).
They **import** our code (dataset/policy factories, the sequential trainer's per-task dataloader
and paired-noise loss evaluator, lerobot-train's update structure) and modify none of it.

| | RETAIN (Yadav et al., ICLR 2026, 2512.08333) | O-LoRA (Wang et al., EMNLP-F 2023, 2310.14152) |
|---|---|---|
| per task | full fine-tune of the running merged model (pi0.5 full-FT preset compressed to 5k steps) | a new rank-64 adapter (specialist recipe), earlier adapters frozen but active |
| boundary | `theta <- 0.5 theta_prev + 0.5 theta_ft` (uniform, their continual setting) | export the sum of adapters as ONE rank-(64k) PEFT adapter, padded to 640 |
| extra loss | none | `0.5 * sum_{i<t} \|A_i A_t^T\|_1` (official-code form), skipped on the 32-input projections |
| evaluates via | dense `pretrained_model` dirs (campaign script unchanged) | `--policy.use_peft=true` (campaign + `mse_matrix_peft.py` unchanged) |
| drift instrument | `mse_matrix_dense.py` (strict full reload) | `scripts/vla_analysis/mse_matrix_peft.py` |

Files: `common.py` (setup mirror, preemption helpers, train step), `retain/retain_sequential_train.py`,
`retain/retain_merge.py` (lerobot-free interpolation + CLI), `olora/olora_sequential_train.py`,
`olora/olora_common.py` (penalty, rank concatenation, export), `mse_matrix_dense.py`,
`run_baseline_triangle.sh`, `tests/test_algebra.py` (local, torch+peft only).
VM wrappers: `job_scripts/nebius/baselines/{retain_a05,olora_r64}_10task.sh` (`SMOKE=1` = 2 tasks x 20
steps with a forced stop/resume and the drift instrument). Queues: `scripts/ops/queue_e67_{train,eval}.sh`;
watcher: `scripts/ops/heartbeat_e67.sh`. Research log: Entry 67.
