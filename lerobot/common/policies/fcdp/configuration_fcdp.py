# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# lerobot/lerobot/common/policies/fcdp/configuration_fcdp.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("fcdp")
@dataclass
class FCDPConfig(PreTrainedConfig):
    """Configuration class for the Failure-Conditioned Diffusion Policy (FCDP)."""

    # Context & Transformer Parameters (from FCIL)
    state_dim: int | None = None
    action_dim: int | None = None
    
    image_feature_keys: List[str] = field(default_factory=lambda: ["observation.images.top", "observation.images.wrist"])
    use_embeddings: bool = True
    embedding_dim_in: int = 1024
    
    model_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    activation_function: str = "gelu"
    
    history_len: int = 10
    max_fail_traj_len: int = 50
    frame_skip_rate: int = 1
    
    recovery_loss_weight: float = 0.5
    
    # Diffusion & UNet Parameters (from Diffusion)
    horizon: int = 16
    n_action_steps: int = 8
    
    # UNet architecture
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    
    # Noise scheduler parameters
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    num_inference_steps: int | None = None
    
    # Global conditioning dimension (output of transformer encoder)
    global_cond_dim: int = 256
    
    type: str = "fcdp"
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )
    
    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 8  # horizon - n_action_steps
    
    # Training parameters
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 9000
    scheduler_decay_lr_ratio: float = 0.1
    dinov2_model_name: str = "dinov2_vitl14"
    
    # Loss computation
    do_mask_loss_for_padding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.frame_skip_rate < 1:
            raise ValueError("frame_skip_rate must be >= 1.")
        if not self.use_embeddings:
            raise ValueError("FCDP currently requires use_embeddings=True.")
        if self.embedding_dim_in % self.model_dim != 0:
            raise ValueError(f"embedding_dim_in ({self.embedding_dim_in}) must be divisible by model_dim ({self.model_dim}) for token splitting.")
        
        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
        
        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

    @property
    def policy_action_dim(self):
        if self.action_dim is None:
            return None
        return self.action_dim + 1

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * self.scheduler_decay_lr_ratio,
        )
    
    def validate_features(self) -> None:
        if self.action_dim is None and self.input_features.get("action"):
            self.action_dim = self.input_features["action"].shape[0]
        
        if self.state_dim is None and self.input_features.get("observation.state"):
            self.state_dim = self.input_features["observation.state"].shape[0]

        if self.action_dim is None or self.state_dim is None:
            raise ValueError("action_dim and state_dim must be set or inferable from dataset features.")

        self.output_features = {
            "action_pred": PolicyFeature(type=FeatureType.ACTION, shape=(self.policy_action_dim,))
        }
        
        if "observation.state" not in self.input_features:
            raise ValueError("FCDP requires 'observation.state' in input_features.")
        if not self.image_feature_keys:
            raise ValueError("FCDP requires at least one image_feature_key.")
        for key in self.image_feature_keys:
            if key not in self.input_features:
                raise ValueError(f"Image feature key '{key}' not found in inferred input_features.")
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.embedding_dim_in,))
        # For history and failure context, the 'action' key will include the done flag.
        self.input_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(self.policy_action_dim,))

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.history_len, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(0, self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None