#!/usr/bin/env python3
"""THROWAWAY forward query-decomposition probe driver (does NOT touch training code).
Reuses lerobot's own parser.wrap + factories to build policy/dataset/preprocessor exactly as
the trainer does, then runs the query-decomposition probe (query_forward.run_query_probe) and exits.
Run like the trainer CLI, e.g.:
  python query_forward_standalone.py --policy.path=<ckpt> --dataset.repo_id=libero_10 ...
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import SequentialOnlineConfig, _collect_task_index_to_name


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    # processor setup — replicated verbatim from lerobot_sequential_train.py (1440-1469)
    processor_kwargs, postprocessor_kwargs = {}, {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = accelerator.prepare(policy)
    task_index_to_name = _collect_task_index_to_name(dataset)

    from query_forward import run_query_probe
    run_query_probe(policy, accelerator, dataset, task_index_to_name, preprocessor, device, cfg)


if __name__ == "__main__":
    main()
