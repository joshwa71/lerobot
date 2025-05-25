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
from typing import List # Python 3.8 compatibility

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("fcil_classifier")
@dataclass
class FCILClassifierConfig(PreTrainedConfig):
    """Configuration class for the Trajectory Failure Classifier Policy."""

    state_dim: int | None = None
    action_dim: int | None = None
    image_feature_keys: List[str] = field(default_factory=list)

    use_embeddings: bool = False
    embedding_dim: int | None = None

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    vision_feature_dim: int = 512 # Output dim of vision_backbone (e.g., 512 for ResNet18) before any final FC layer.

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    dim_model: int = 256       # Transformer hidden dimension
    n_heads: int = 4
    n_encoder_layers: int = 3
    dim_feedforward: int = 1024
    feedforward_activation: str = "relu"
    dropout: float = 0.1
    max_seq_len: int = 100     # Number of original trajectory timesteps
    pre_norm: bool = True

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_config: LRSchedulerConfig | None = None

    # Calculated property, not user-set directly
    _actual_transformer_max_seq_len: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.output_features = {
            "is_failure_pred": PolicyFeature(type=FeatureType.STATE, shape=(1,))
        }
        if self.use_embeddings:
            if self.embedding_dim is None:
                raise ValueError("If use_embeddings is True, embedding_dim must be specified.")
            if self.embedding_dim % self.dim_model != 0 :
                 raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by dim_model ({self.dim_model}) for segmentation.")
        
        # Calculate actual sequence length after tokenization strategy
        tokens_per_timestep = 2 # state + action
        if self.image_feature_keys:
            if self.use_embeddings:
                num_visual_tokens_per_cam = self.embedding_dim // self.dim_model
                tokens_per_timestep += num_visual_tokens_per_cam * len(self.image_feature_keys)
            else:
                tokens_per_timestep += 1 # One fused visual token
        self._actual_transformer_max_seq_len = 1 + self.max_seq_len * tokens_per_timestep # +1 for CLS

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return self.scheduler_config

    def validate_features(self) -> None:
        if self.state_dim is None or self.action_dim is None :
            pass

        self.input_features = {}
        if self.state_dim:
            self.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        if self.action_dim:
            self.input_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))

        for key in self.image_feature_keys:
            if self.use_embeddings:
                if self.embedding_dim is None:
                    raise ValueError("embedding_dim must be set in config if use_embeddings is True and image_feature_keys are present.")
                # The input feature from dataset is embedding_dim
                self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.embedding_dim,))
            else:
                self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)) # Placeholder for raw images

        if not self.input_features and not (self.state_dim or self.action_dim or self.image_feature_keys):
             raise ValueError("FCILClassifierConfig requires state_dim, action_dim, or image_feature_keys to be set, "
                             "or input_features to be populated by the policy factory.")