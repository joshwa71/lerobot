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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter # For ResNet feature extraction
from torchvision.ops.misc import FrozenBatchNorm2d # For ResNet
import einops # For rearranging tensors, e.g. in image processing

from lerobot.common.policies.act.modeling_act import ACTEncoderLayer
from lerobot.common.policies.normalize import Normalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.fcil_classifier.configuration_fcil_classifier import FCILClassifierConfig
from lerobot.configs.types import PolicyFeature, FeatureType


class FCILClassifierModel(nn.Module):
    def __init__(self, config: FCILClassifierConfig):
        super().__init__()
        self.config = config

        # Image Backbone (ResNet-style) or direct embedding projection
        if self.config.image_feature_keys:
            if not self.config.use_embeddings:
                if not hasattr(torchvision.models, config.vision_backbone):
                    raise ValueError(f"Vision backbone {config.vision_backbone} not found in torchvision.models")

                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, False], # Standard ResNet
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d if config.pretrained_backbone_weights else nn.BatchNorm2d,
                )
                self.vision_backbone = nn.Sequential(*list(backbone_model.children())[:-1]) # Before final FC
                # Projection for image features from backbone
                image_proj_in_dim = config.vision_feature_dim * len(config.image_feature_keys)
            else: # use_embeddings is True
                self.vision_backbone = None # No online backbone needed
                if config.embedding_dim is None:
                     raise ValueError("embedding_dim must be specified in config when use_embeddings is True.")
                image_proj_in_dim = config.embedding_dim * len(config.image_feature_keys)
            
            self.image_feature_proj = nn.Linear(image_proj_in_dim, config.dim_model)
        else:
            self.vision_backbone = None
            self.image_feature_proj = None


        # Projection for state features
        self.state_proj = nn.Linear(config.state_dim, config.dim_model)

        # Projection for action features
        self.action_proj = nn.Linear(config.action_dim, config.dim_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim_model))

        # Positional embedding for CLS token + sequence tokens
        self.pos_embed = nn.Embedding(config.max_seq_len + 1, config.dim_model)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

        # Output head for binary classification
        self.output_head = nn.Linear(config.dim_model, 1)

    def forward(self, obs_state_seq, act_seq, obs_feature_seq_dict=None, padding_mask=None):
        # obs_state_seq: (batch, seq_len, state_dim)
        # act_seq: (batch, seq_len, action_dim)
        # obs_feature_seq_dict: Dict{"cam_key": (batch, seq_len, C, H, W) or (batch, seq_len, embedding_dim)}
        # padding_mask: (batch, seq_len) - True for padded elements

        batch_size, seq_len, _ = obs_state_seq.shape

        # Project states and actions
        state_embed = self.state_proj(obs_state_seq) # (batch, seq_len, dim_model)
        action_embed = self.action_proj(act_seq)     # (batch, seq_len, dim_model)

        # Process image features or embeddings
        if self.config.image_feature_keys and obs_feature_seq_dict:
            all_cam_processed_features = []
            for cam_key in self.config.image_feature_keys:
                feature_seq = obs_feature_seq_dict[cam_key] # (B, T, C, H, W) or (B, T, emb_dim)

                if not self.config.use_embeddings:
                    # Process raw images through backbone
                    # Reshape for backbone: (B*T, C, H, W)
                    bt, c, h, w = feature_seq.shape[0]*feature_seq.shape[1], feature_seq.shape[2], feature_seq.shape[3], feature_seq.shape[4]
                    feature_seq_reshaped = feature_seq.reshape(bt, c, h, w)

                    img_features_from_backbone = self.vision_backbone(feature_seq_reshaped) # (B*T, vision_feature_dim, 1, 1) for ResNet GAP
                    img_features_from_backbone = img_features_from_backbone.squeeze(-1).squeeze(-1) # (B*T, vision_feature_dim)

                    # Reshape back to (B, T, vision_feature_dim)
                    processed_cam_feature_seq = img_features_from_backbone.view(batch_size, seq_len, -1)
                else:
                    # Embeddings are already (B, T, embedding_dim)
                    processed_cam_feature_seq = feature_seq
                
                all_cam_processed_features.append(processed_cam_feature_seq)

            # Concatenate features from all cameras/embedding sources along the feature dimension
            concatenated_img_features = torch.cat(all_cam_processed_features, dim=-1) # (B, T, num_cams * (vision_feature_dim or embedding_dim))
            image_embed = self.image_feature_proj(concatenated_img_features) # (batch, seq_len, dim_model)

            # Combine modalities: simple summation
            fused_embed_per_timestep = state_embed + action_embed + image_embed
        else:
            # Combine modalities: simple summation (state + action only)
            fused_embed_per_timestep = state_embed + action_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, dim_model)
        encoder_input = torch.cat([cls_tokens, fused_embed_per_timestep], dim=1) # (batch, seq_len+1, dim_model)

        # Add positional embeddings
        positions = torch.arange(0, seq_len + 1, device=obs_state_seq.device).unsqueeze(0).expand(batch_size, -1)
        encoder_input += self.pos_embed(positions)

        if padding_mask is not None:
            cls_padding_mask = torch.full((batch_size, 1), False, device=padding_mask.device) # CLS token is never padded
            transformer_key_padding_mask = torch.cat([cls_padding_mask, padding_mask], dim=1)
        else:
            transformer_key_padding_mask = None

        x = encoder_input.permute(1, 0, 2) # (seq_len+1, batch, dim_model) for Transformer

        for layer in self.encoder_layers:
            x = layer(x, pos_embed=None, key_padding_mask=transformer_key_padding_mask)
        x = self.encoder_norm(x)

        cls_output = x[0] # (batch, dim_model) - CLS token output
        logits = self.output_head(cls_output) # (batch, 1)
        return logits


class FCILClassifierPolicy(PreTrainedPolicy):
    config_class = FCILClassifierConfig
    name = "fcil_classifier"

    def __init__(
        self,
        config: FCILClassifierConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config

        self._infer_and_set_config_dims(dataset_stats)

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.model = FCILClassifierModel(config)

    def _infer_and_set_config_dims(self, dataset_stats):
        if self.config.state_dim is None and dataset_stats and "observation.state" in dataset_stats:
            self.config.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
        if self.config.action_dim is None and dataset_stats and "action" in dataset_stats:
            self.config.action_dim = dataset_stats["action"]["mean"].shape[0]

        # If using embeddings, embedding_dim must be set in the config.
        if self.config.use_embeddings and self.config.embedding_dim is None:
             # This check is also in FCILClassifierConfig.__post_init__, but good to have here too.
            raise ValueError("embedding_dim must be set in config if use_embeddings is True.")
        
        if self.config.state_dim is None or self.config.action_dim is None :
            raise ValueError(
                "state_dim and action_dim must be provided in config or inferable from dataset_stats."
            )
        
        # This will populate self.config.input_features based on the (now set) dims
        self.config.validate_features()


    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        obs_state_seq = batch['observation.state']
        act_seq = batch['action']
        labels = batch['label']
        padding_mask = batch.get('padding_mask')

        # obs_feature_seq_dict will hold either images or embeddings based on config
        obs_feature_seq_dict = {key: batch[key] for key in self.config.image_feature_keys if key in batch}

        # Normalization:
        # Create a temporary dict for normalization, including only expected features
        # VISUAL normalization in Normalize module expects (C,H,W) and stats for (C,1,1).
        # If embeddings are (emb_dim,), this normalization will be a no-op or might need adjustment
        # if specific per-dimension embedding normalization is desired. For now, assume no-op.
        norm_input_dict = {'observation.state': obs_state_seq, 'action': act_seq}
        if obs_feature_seq_dict:
             norm_input_dict.update(obs_feature_seq_dict)

        normalized_batch = self.normalize_inputs(norm_input_dict)

        norm_obs_state_seq = normalized_batch['observation.state']
        norm_act_seq = normalized_batch['action']
        norm_obs_feature_seq_dict = {key: normalized_batch[key] for key in self.config.image_feature_keys if key in normalized_batch}


        logits = self.model(
            norm_obs_state_seq,
            norm_act_seq,
            obs_feature_seq_dict=norm_obs_feature_seq_dict if norm_obs_feature_seq_dict else None,
            padding_mask=padding_mask
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_failure_prob(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        obs_state_seq = batch['observation.state']
        act_seq = batch['action']
        padding_mask = batch.get('padding_mask')
        obs_feature_seq_dict = {key: batch[key] for key in self.config.image_feature_keys if key in batch}

        norm_input_dict = {'observation.state': obs_state_seq, 'action': act_seq}
        if obs_feature_seq_dict:
             norm_input_dict.update(obs_feature_seq_dict)

        normalized_batch = self.normalize_inputs(norm_input_dict)

        norm_obs_state_seq = normalized_batch['observation.state']
        norm_act_seq = normalized_batch['action']
        norm_obs_feature_seq_dict = {key: normalized_batch[key] for key in self.config.image_feature_keys if key in normalized_batch}

        logits = self.model(
            norm_obs_state_seq,
            norm_act_seq,
            obs_feature_seq_dict=norm_obs_feature_seq_dict if norm_obs_feature_seq_dict else None,
            padding_mask=padding_mask
        )
        probs = torch.sigmoid(logits)
        return probs

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # For a classifier, select_action might not be standard, but let's have it return the probability.
        return self.predict_failure_prob(batch)