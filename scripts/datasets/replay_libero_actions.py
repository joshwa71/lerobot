"""Sanity-check a LIBERO LeRobot dataset against the simulator by replaying its actions.

For K episodes of each task: reset LeRobot's LiberoEnv to benchmark init state k (the eval init
states, NOT the demo's own start, so open-loop success is expected to be partial), then check

  * image orientation -- dataset frame 0 vs the env's first image, raw and rotated 180 degrees
    (LiberoProcessorStep rotates env images at eval time, so a correct dataset matches "rot180");
  * state             -- dataset state[0] vs the env's [eef_pos, axis-angle, gripper_qpos];
  * actions           -- open-loop replay of the episode's actions. The sharp signal is TRACKING: the
                         env's end-effector path vs the dataset's recorded one over the first steps
                         (it starts ~1-3 cm apart and should stay close if action scale/axes/frame are
                         right), and gripper open/close agreement. Task success is reported too, but
                         replaying from a different init state rarely succeeds on long tasks (5% on
                         HuggingFaceVLA's libero_10), so it is a weak signal on its own.

A dataset whose conventions are wrong (flipped images, wrong gripper sign, wrong frame) shows a
non-"rot180" orientation, large state offsets, fast tracking divergence and gripper disagreement.
Compare against a known-good dataset (libero_10 from HuggingFaceVLA/libero) through the same script.

    MUJOCO_GL=egl python scripts/datasets/replay_libero_actions.py --root outputs/libero_10 \
        --suite libero_10 --episodes-per-task 3 --report outputs/_replay_libero_10.json
"""

import argparse
import io
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from libero.libero import benchmark
from PIL import Image

from lerobot.envs.libero import LiberoEnv
from lerobot.processor import LiberoProcessorStep

IMG, WRIST = "observation.images.image", "observation.images.image2"


def _img(cell) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).mean(-1).ravel()
    b = b.astype(np.float32).mean(-1).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _env_state(obs) -> np.ndarray:
    rs = obs["robot_state"]
    quat = torch.as_tensor(np.asarray(rs["eef"]["quat"], dtype=np.float32))[None]
    axis_angle = LiberoProcessorStep()._quat2axisangle(quat)[0].numpy()
    return np.concatenate([rs["eef"]["pos"], axis_angle, rs["gripper"]["qpos"]]).astype(np.float32)


def load_episodes(root: Path, episodes: set[int]) -> dict[int, pd.DataFrame]:
    cols = ["episode_index", "frame_index", "task_index", "observation.state", "action", IMG, WRIST]
    out = {}
    for f in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(f, columns=cols)
        df = df[df["episode_index"].isin(episodes)]
        for ep, g in df.groupby("episode_index"):
            out[int(ep)] = g.sort_values("frame_index").reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--suite", required=True, help="LIBERO benchmark suite the tasks belong to")
    p.add_argument("--episodes-per-task", type=int, default=3)
    p.add_argument("--task-indices", type=int, nargs="*", help="dataset task_index values (default: all)")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    env_langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    tasks = pd.read_parquet(args.root / "meta/tasks.parquet")
    frames = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in (args.root / "data").rglob("*.parquet")
    )
    ep_task = frames.groupby("episode_index")["task_index"].first()

    plan = {}
    for task, row in tasks.iterrows():
        ti = int(row.task_index)
        if args.task_indices and ti not in args.task_indices:
            continue
        eps = sorted(ep_task[ep_task == ti].index)[: args.episodes_per_task]
        if not eps:
            continue  # empty vocabulary entry (e.g. other suites' tasks in libero_10)
        env_ids = [i for i, lang in enumerate(env_langs) if lang == task]
        if len(env_ids) != 1:
            logging.info(f"skip task_index {ti}: {len(env_ids)} env matches for {task!r}")
            continue
        plan[ti] = (env_ids[0], eps, task)

    data = load_episodes(args.root, {e for _, eps, _ in plan.values() for e in eps})
    results = []
    for ti, (env_id, eps, task) in sorted(plan.items()):
        env = LiberoEnv(task_suite=suite, task_id=env_id, task_suite_name=args.suite,
                        obs_type="pixels_agent_pos", episode_index=0)
        for ep in eps:
            df = data[ep]
            init_state = env.init_state_id  # reset() uses this one, then advances (as does a success)
            obs, _ = env.reset()
            r = {"task_index": ti, "env_id": env_id, "episode": ep, "init_state": init_state, "len": len(df)}
            for key, name in ((IMG, "image"), (WRIST, "image2")):
                env_img, ds_img = obs["pixels"][name], _img(df[key].iloc[0])
                r[f"{name}_corr_raw"] = round(_corr(ds_img, env_img), 3)
                r[f"{name}_corr_rot180"] = round(_corr(ds_img, env_img[::-1, ::-1]), 3)
            s_env, s_ds = _env_state(obs), np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
            r["d_eef_pos_cm"] = round(float(np.linalg.norm(s_env[:3] - s_ds[:3])) * 100, 2)
            r["d_axis_angle"] = round(float(np.linalg.norm(s_env[3:6] - s_ds[3:6])), 3)
            r["d_gripper"] = round(float(np.abs(s_env[6:] - s_ds[6:]).max()), 4)
            ds_states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            success, steps, pos_err, grip_agree = False, 0, [], []
            for t, a in enumerate(df["action"]):
                obs, _, term, _, info = env.step(np.asarray(a, dtype=np.float32))
                steps += 1
                if info["is_success"]:
                    success = True
                    break
                if t + 1 < len(ds_states):
                    s_t = _env_state(obs)
                    pos_err.append(float(np.linalg.norm(s_t[:3] - ds_states[t + 1, :3])) * 100)
                    # gripper finger opening agrees on open (> mid) vs closed
                    mid = 0.02
                    grip_agree.append((s_t[6] > mid) == (ds_states[t + 1, 6] > mid))
            pos_err = np.asarray(pos_err)
            r.update(
                success=success,
                steps=steps,
                track_err_cm_10=round(float(pos_err[:10].mean()), 2) if len(pos_err) else None,
                track_err_cm_30=round(float(pos_err[:30].mean()), 2) if len(pos_err) else None,
                track_err_cm_all=round(float(pos_err.mean()), 2) if len(pos_err) else None,
                gripper_agree=round(float(np.mean(grip_agree)), 3) if grip_agree else None,
            )
            results.append(r)
            logging.info(
                f"task {ti:2d} env {env_id:2d} ep {ep:4d}: success={success!s:5} steps={steps:3d}/{len(df)} "
                f"img raw/rot {r['image_corr_raw']:+.2f}/{r['image_corr_rot180']:+.2f} "
                f"wrist raw/rot {r['image2_corr_raw']:+.2f}/{r['image2_corr_rot180']:+.2f} "
                f"d_pos {r['d_eef_pos_cm']}cm track10/30/all {r['track_err_cm_10']}/{r['track_err_cm_30']}/"
                f"{r['track_err_cm_all']}cm grip {r['gripper_agree']}"
            )
        env.close()

    res = pd.DataFrame(results)
    orient = Counter(np.where(res.image_corr_rot180 > res.image_corr_raw, "rot180", "raw"))
    summary = {
        "root": str(args.root),
        "suite": args.suite,
        "episodes_replayed": len(res),
        "tasks": len(plan),
        "replay_success_rate": round(float(res.success.mean()), 3),
        "tasks_with_any_success": int(res.groupby("task_index").success.any().sum()),
        "orientation_votes": dict(orient),
        "median_d_eef_pos_cm": float(res.d_eef_pos_cm.median()),
        "median_d_axis_angle": float(res.d_axis_angle.median()),
        "median_d_gripper": float(res.d_gripper.median()),
        "median_track_err_cm_10": float(res.track_err_cm_10.median()),
        "median_track_err_cm_30": float(res.track_err_cm_30.median()),
        "median_track_err_cm_all": float(res.track_err_cm_all.median()),
        "median_gripper_agree": float(res.gripper_agree.median()),
    }
    logging.info(json.dumps(summary, indent=1))
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))


if __name__ == "__main__":
    main()
