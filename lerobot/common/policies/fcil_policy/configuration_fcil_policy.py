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
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig 
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("fcil_policy")
@dataclass
class FCILPolicyConfig(PreTrainedConfig):
    """Configuration class for the Failure-Conditioned Imitation Learning Policy."""

    state_dim: int | None = None 
    action_dim: int | None = None 
    
    image_feature_keys: List[str] = field(default_factory=lambda: ["observation.images.top", "observation.images.wrist"])
    use_embeddings: bool = True 
    embedding_dim_in: int = 1024 

    type: str = "fcil_policy"
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD, 
            "VISUAL": NormalizationMode.IDENTITY, 
        }
    )

    model_dim: int = 512  
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    activation_function: str = "gelu" 

    history_len: int = 10 
    max_fail_traj_len: int = 50 
    
    frame_skip_rate: int = 1 

    max_seq_len: int = 512

    recovery_loss_weight: float = 0.5  

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
        # For history and failure context, the 'action' key will include the done flag.
        self.input_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(self.policy_action_dim,))


    @property
    def observation_delta_indices(self) -> None: return None
    @property
    def action_delta_indices(self) -> None: return None
    @property
    def reward_delta_indices(self) -> None: return None