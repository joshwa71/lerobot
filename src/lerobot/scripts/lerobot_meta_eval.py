#!/usr/bin/env python

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import torch.multiprocessing as mp
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from lerobot.configs import parser
from lerobot.meta.configs import MetaTrainConfig
from lerobot.meta.engine import MetaEngine
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import init_logging


mp.set_start_method("spawn", force=True)


@parser.wrap()
def meta_eval(cfg: MetaTrainConfig):
    """
    Standalone meta-eval entrypoint.

    Expects --policy.path to point to a meta-trained checkpoint directory
    (i.e. the 'pretrained_model' folder produced by meta-training).
    """
    cfg.validate()
    if cfg.output_dir is None:
        raise ValueError("Meta-eval requires --output_dir to be set.")

    # Preserve the path to the meta-trained policy weights before engine.setup(),
    # then force policy instantiation from base weights so that LoRA wrappers
    # are attached first and we can correctly load the LoRA parameters.
    policy_weight_dir = cfg.policy.pretrained_path
    if not policy_weight_dir:
        raise ValueError(
            "Meta-eval requires --policy.path to point to a meta-trained 'pretrained_model' directory."
        )
    policy_weight_dir = Path(policy_weight_dir)
    model_file = policy_weight_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"Expected model weights at {model_file}, but file does not exist.")

    logging.info("Starting meta-eval from meta checkpoint at %s", policy_weight_dir)

    # Do not let make_policy() try to load weights; we want a fresh base policy
    # + attached LoRA, then load the full LoRA checkpoint into that structure.
    cfg.policy.pretrained_path = None

    engine = MetaEngine(cfg)
    engine.setup()

    # Load the meta-trained weights (including LoRA parameters) into the
    # already-LoRA-wrapped policy that engine.setup() constructed.
    policy_cls: type[PreTrainedPolicy] = type(engine.policy)
    map_location = getattr(cfg.policy, "device", None) or "cpu"
    policy_cls._load_as_safetensor(
        model=engine.policy,
        model_file=str(model_file),
        map_location=map_location,
        strict=False,
    )

    total_steps = cfg.steps
    step = total_steps
    engine._run_meta_eval(step=step, total_outer_steps=total_steps)


def main():
    init_logging()
    meta_eval()


if __name__ == "__main__":
    main()


