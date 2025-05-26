# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig # Or another appropriate one
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("fcil_policy")
@dataclass
class FCILPolicyConfig(PreTrainedConfig):
    """Configuration class for the Failure-Conditioned Imitation Learning Policy."""

    # Input / output structure related
    state_dim: int | None = None  # Inferred from dataset if not provided
    action_dim: int | None = None # Original action dim, inferred from dataset. Policy will predict action_dim + 1 (for done)
    
    image_feature_keys: List[str] = field(default_factory=lambda: ["observation.images.top", "observation.images.wrist"])
    use_embeddings: bool = True # Assumed to be true for FCIL as per description
    embedding_dim_in: int = 1024 # Dimension of precomputed image embeddings (e.g., DINOv2 base)

    type: str = "fcil_policy"
    
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD, # Action targets will be normalized
            "VISUAL": NormalizationMode.IDENTITY, # Embeddings are assumed to be somewhat normalized or handled by Linear layers
        }
    )

    # Model architecture
    model_dim: int = 512  # Transformer hidden dimension
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    activation_function: str = "gelu" # Common in transformers

    history_len: int = 10 # Number of past (s, o, a) timesteps to use as context
    max_fail_traj_len: int = 50 # Max number of timesteps from the failure trajectory to use as context
    
    frame_skip_rate: int = 1 # Sample every Nth frame. 1 means no skipping (backward compatible).

    # Calculated internally, but good to know the components
    # Max tokens per state = 1
    # Max tokens per action = 1
    # Max tokens per obs = num_cameras * (embedding_dim_in / model_dim)
    # Max tokens per full_traj_timestep = state + obs + action
    # max_seq_len = max_fail_traj_len * tokens_per_ts_in_fail_traj + 1 (fail_token) + history_len * tokens_per_ts_in_hist + (state_tok + obs_tok for current)
    # This will be dynamically computed based on other params, or set to a safe upper bound
    # For now, let's estimate a reasonable upper bound.
    # Example: state=1, action=1, 2 cams, 1024 emb_in, 512 model_dim -> obs_tok = 2 * (1024/512) = 4.
    # Tokens per ts in fail traj (s,o,a): 1 + 4 + 1 = 6
    # Tokens per ts in history (s,o,a): 1 + 4 + 1 = 6
    # Max fail traj: 50 * 6 = 300
    # Fail token: 1
    # History: 10 * 6 = 60
    # Current s,o: 1 + 4 = 5
    # Total: 300 + 1 + 60 + 5 = 366. Let's set max_seq_len higher for safety / other configs.
    max_seq_len: int = 512

    # Training
    recovery_loss_weight: float = 0.5  # Lambda for L_recovery

    # Optimizer and Scheduler presets
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 9000
    scheduler_decay_lr_ratio: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.frame_skip_rate < 1:
            raise ValueError("frame_skip_rate must be >= 1.")
        if not self.use_embeddings:
            raise ValueError("FCILPolicy currently requires use_embeddings=True.")
        if self.embedding_dim_in % self.model_dim != 0:
            raise ValueError(f"embedding_dim_in ({self.embedding_dim_in}) must be divisible by model_dim ({self.model_dim}) for token splitting.")

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
            raise ValueError("FCILPolicy requires 'observation.state' in input_features.")
        if not self.image_feature_keys:
            raise ValueError("FCILPolicy requires at least one image_feature_key.")
        for key in self.image_feature_keys:
            if key not in self.input_features:
                raise ValueError(f"Image feature key '{key}' not found in inferred input_features.")
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.embedding_dim_in,))
        if "action" not in self.input_features: 
             self.input_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))


    @property
    def observation_delta_indices(self) -> None: return None
    @property
    def action_delta_indices(self) -> None: return None
    @property
    def reward_delta_indices(self) -> None: return None