#!/usr/bin/env python

# Copyright 2025

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot import envs
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.adapters.lora import LoraAttachConfig


@dataclass
class InnerOptConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0


@dataclass
class MetaAlgoConfig(draccus.ChoiceRegistry):
    pass


@MetaAlgoConfig.register_subclass("reptile")
@dataclass
class ReptileConfig(MetaAlgoConfig):
    meta_step_size: float = 0.1


@MetaAlgoConfig.register_subclass("fomaml")
@dataclass
class FOMAMLConfig(MetaAlgoConfig):
    first_order: bool = True


@dataclass
class MetaTrainConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    env: envs.EnvConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    # Outer loop control
    steps: int = 100_000
    batch_size: int = 8
    log_freq: int = 1000
    save_freq: int = 10_000
    verbose_log: bool = False

    # Tasking
    train_tasks: list[int] | None = None
    eval_tasks: list[int] | None = None
    tasks_per_outer_step: int = 4
    support_frames_per_task: int = 1024
    query_frames_per_task: int = 512

    # Loops
    inner_steps: int = 3
    inner_opt: InnerOptConfig = field(default_factory=InnerOptConfig)
    algo: MetaAlgoConfig = field(default_factory=ReptileConfig)

    # LoRA
    lora: LoraAttachConfig = field(default_factory=lambda: LoraAttachConfig(enable=True))

    # Logging / eval
    eval_freq: int = 10000
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def validate(self):
        from lerobot.configs import parser
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            # self.policy.pretrained_path = policy_path
        
        if self.policy is None:
            raise ValueError(
                "Policy configuration is required. Please provide either --policy.path "
                "to load a pretrained policy or specify --policy.type to create a new one."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


