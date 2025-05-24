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
    """Configuration class for the Trajectory Failure Classifier Policy with Image Processing."""

    # Input features:
    state_dim: int | None = None
    action_dim: int | None = None
    image_feature_keys: List[str] = field(default_factory=list) # If use_embeddings=True, these keys point to embeddings

    # Embedding settings (if use_embeddings=True)
    use_embeddings: bool = False
    embedding_dim: int | None = None # Dimension of pre-computed embeddings

    # Image processing (if use_embeddings=False)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    vision_feature_dim: int = 512 # Output dim of vision_backbone before projection

    # Normalization settings
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY, # For raw image pixels or embeddings
        }
    )

    # Architecture (Transformer Encoder based)
    dim_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 3
    dim_feedforward: int = 1024
    feedforward_activation: str = "relu"
    dropout: float = 0.1
    max_seq_len: int = 100
    pre_norm: bool = True

    # Training presets (can be overridden)
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_config: LRSchedulerConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        self.output_features = {
            "is_failure_pred": PolicyFeature(type=FeatureType.STATE, shape=(1,))
        }
        if self.use_embeddings and self.embedding_dim is None:
            raise ValueError("If use_embeddings is True, embedding_dim must be specified.")

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
            # This will be caught later if not set by the policy factory
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
                self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.embedding_dim,))
            else:
                # For raw images, shape is typically (C, H, W). Let's use a placeholder or infer from dataset.
                # For now, Normalize module for VISUAL uses (C,1,1) so exact H,W not critical for it.
                # The actual image shape is handled by the backbone.
                self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)) # Placeholder for raw images

        if not self.input_features and not (self.state_dim or self.action_dim or self.image_feature_keys):
             raise ValueError("FCILClassifierConfig requires state_dim, action_dim, or image_feature_keys to be set, "
                             "or input_features to be populated by the policy factory.")