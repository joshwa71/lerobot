"""Build libero_10 / libero_goal / libero_object / libero_spatial from HuggingFaceVLA/libero.

HuggingFaceVLA/libero (LeRobot v3.0, images stored in parquet) holds all four suites: 1693 episodes,
40 tasks. Its task_index 0-9 are the ten LIBERO-Long tasks, and in that order they are LIBERO-10 envs
4,6,9,2,7,0,8,1,3,5 -- the paper's sequential order. 10-19 are goal, 20-29 object, 30-39 spatial.

Each suite dataset keeps the full 40-task table with the source task_index values (other suites' tasks
are empty vocabulary), exactly like the Nebius-era libero_10. Stats are recomputed per suite,
including the quantiles pi05 needs (the source's per-episode stats have none).
The source's stale per-episode data-file index is corrected first (see libero_common.repaired_source).

    python scripts/datasets/build_libero_suites.py \
        --src outputs/hf_src/HuggingFaceVLA/libero --out outputs [--suites libero_10 ...]
        [--work outputs/_build_libero_suites]
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from libero.libero import benchmark

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from libero_common import (
    add_quantile_stats,
    hub_revision,
    repaired_source,
    subset_keep_task_table,
    write_build_info,
)

SRC_REPO = "HuggingFaceVLA/libero"
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
# Episode counts of the suites in HuggingFaceVLA/libero (libero_10's 379 matches the research log).
EXPECTED_EPISODES = {"libero_10": 379, "libero_goal": 428, "libero_object": 454, "libero_spatial": 432}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="local copy of HuggingFaceVLA/libero")
    p.add_argument("--out", type=Path, required=True, help="parent dir; writes <out>/<suite>")
    p.add_argument("--suites", nargs="+", default=SUITES, choices=SUITES)
    p.add_argument("--work", type=Path, help="scratch dir (default <out>/_build_libero_suites)")
    p.add_argument("--allow-partial", action="store_true",
                   help="validation only: build from the episodes whose data is present (marked in BUILD_INFO)")
    args = p.parse_args()
    # force: lerobot configures logging on import, which would make a plain basicConfig a no-op
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    work = args.work or args.out / "_build_libero_suites"
    src_root = repaired_source(args.src, work / "src_repaired")
    src = LeRobotDatasetMetadata(SRC_REPO, root=src_root)
    present = set()
    for f in (args.src / "data").glob("chunk-*/file-*.parquet"):
        present.update(pd.read_parquet(f, columns=["episode_index"])["episode_index"].unique().tolist())

    bench = benchmark.get_benchmark_dict()
    lang_to_suite = {}
    for suite in SUITES:
        b = bench[suite]()
        for env_id in range(b.n_tasks):
            lang = b.get_task(env_id).language
            assert lang not in lang_to_suite, f"instruction shared across suites: {lang!r}"
            lang_to_suite[lang] = (suite, env_id)

    task_suite = {}
    for task, row in src.tasks.iterrows():
        if task not in lang_to_suite:
            raise ValueError(f"source task {row.task_index} matches no LIBERO suite: {task!r}")
        task_suite[int(row.task_index)] = lang_to_suite[task]

    episodes_by_suite = defaultdict(list)
    for ep in range(src.total_episodes):
        tasks = src.episodes[ep]["tasks"]
        assert len(tasks) == 1, f"episode {ep} has {len(tasks)} tasks"
        suite, _ = lang_to_suite[tasks[0]]
        episodes_by_suite[suite].append(ep)
    counts = {s: len(v) for s, v in episodes_by_suite.items()}
    logging.info(f"episodes per suite: {counts}")
    for suite in args.suites:
        if counts.get(suite) != EXPECTED_EPISODES[suite]:
            raise ValueError(f"{suite}: {counts.get(suite)} episodes, expected {EXPECTED_EPISODES[suite]}")
    if args.allow_partial:
        episodes_by_suite = {s: [e for e in v if e in present] for s, v in episodes_by_suite.items()}
        logging.warning(f"PARTIAL build: episodes present per suite {({s: len(v) for s, v in episodes_by_suite.items()})}")

    for suite in args.suites:
        missing = sorted(set(episodes_by_suite[suite]) - present)
        if missing:
            raise FileNotFoundError(f"{suite}: {len(missing)} episodes have no data file yet, e.g. {missing[:5]}")

    revision = hub_revision(SRC_REPO, args.src)
    for suite in args.suites:
        out_dir = args.out / suite
        if (out_dir / "BUILD_INFO.json").exists():
            logging.info(f"{suite}: already built ({out_dir}/BUILD_INFO.json), skipping")
            continue
        if out_dir.exists():
            raise FileExistsError(f"{out_dir} exists without BUILD_INFO.json (a failed build?); remove it")
        logging.info(f"building {suite} -> {out_dir}")
        ds = subset_keep_task_table(src, episodes_by_suite[suite], out_dir, repo_id=suite)

        # same task table as the source, and the frames use exactly this suite's task indices
        assert ds.meta.tasks["task_index"].to_dict() == src.tasks["task_index"].to_dict()
        frames = pd.concat(
            pd.read_parquet(f, columns=["episode_index", "task_index"])
            for f in sorted((out_dir / "data").rglob("*.parquet"))
        )
        used = sorted(frames["task_index"].unique().tolist())
        expected = sorted(ti for ti, (s, _) in task_suite.items() if s == suite)
        assert used == expected, f"{suite}: task indices {used} != {expected}"
        assert frames["episode_index"].nunique() == ds.meta.total_episodes == len(episodes_by_suite[suite])
        assert len(frames) == ds.meta.total_frames
        add_quantile_stats(out_dir, repo_id=suite)
        assert {"q01", "q99"} <= set(LeRobotDatasetMetadata(suite, root=out_dir).stats["action"])
        env_order = [task_suite[ti][1] for ti in expected]
        logging.info(f"{suite}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
                     f"task_index {expected[0]}-{expected[-1]} -> env ids {env_order}")

        write_build_info(
            out_dir,
            source=SRC_REPO,
            source_revision=revision,
            script="scripts/datasets/build_libero_suites.py",
            suite=suite,
            partial=args.allow_partial,
            source_episodes=episodes_by_suite[suite],
            task_index_to_env_id={ti: task_suite[ti][1] for ti in expected},
        )


if __name__ == "__main__":
    main()
