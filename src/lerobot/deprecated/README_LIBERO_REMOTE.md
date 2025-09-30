# LeRobot ↔ LIBERO Remote Environment (Client)

This document explains how to use a remote LIBERO environment from `lerobot-upstream` without mixing dependencies.

## Overview
- A small server runs inside the LIBERO repo and exposes a gym-like API over TCP.
- `lerobot-upstream` connects via `LiberoRemoteEnv` and treats it like any other environment.
- You can record `LeRobotDataset`, train, and evaluate in LeRobot while rolling out tasks in LIBERO.

## Prerequisites
- LIBERO installed in its own Python environment (with mujoco / robosuite).
- `lerobot-upstream` installed in a separate environment.

## Start LIBERO server
In a LIBERO venv:
```bash
cd /home/josh/phddev/LIBERO
python scripts/remote_env_server.py \
  --benchmark libero_10 \
  --task_id 0 \
  --host 127.0.0.1 \
  --port 5555 \
  --height 128 \
  --width 128
```
This opens a TCP server that loads the selected task and streams observations.

## Run LeRobot client
In a LeRobot venv:
```bash
cd /home/josh/phddev/lerobot-upstream
python -m lerobot.scripts.rl.gym_manipulator \
  --config_path src/lerobot/envs/sim_configs/libero_remote_example.json
```
- To record a dataset, set `mode` to `record` in the JSON and provide `repo_id` / `num_episodes`.
- To replay, set `mode` to `replay` with `episode`.

## Train and Evaluate
- Train using the recorded dataset with the usual LeRobot training command.
- Evaluate a pretrained policy:
```bash
python -m lerobot.scripts.rl.eval_policy --config_path src/lerobot/envs/sim_configs/libero_remote_example.json
```

## Notes
- The server currently maps a 4D action `[dx, dy, dz, gripper]` to robosuite OSC pose `[dx, dy, dz, 0, 0, 0, grip]` with a simple gripper mapping. Extend it as needed.
- Observations include `agentview` and `wrist` images plus a state vector concatenating joints / gripper / eef if available.
