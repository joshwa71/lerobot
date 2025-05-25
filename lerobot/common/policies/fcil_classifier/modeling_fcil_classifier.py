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
from torchvision.ops.misc import FrozenBatchNorm2d # For ResNet
import logging
# Import torchmetrics for calculating metrics
import torchmetrics # ADDED

from lerobot.common.policies.act.modeling_act import ACTEncoderLayer
from lerobot.common.policies.normalize import Normalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.fcil_classifier.configuration_fcil_classifier import FCILClassifierConfig


class FCILClassifierModel(nn.Module):
    def __init__(self, config: FCILClassifierConfig):
        super().__init__()
        self.config = config
        self.tokens_per_timestep = 0

        # Projection for state features
        self.state_proj = nn.Linear(config.state_dim, config.dim_model)
        self.tokens_per_timestep += 1

        # Projection for action features
        self.action_proj = nn.Linear(config.action_dim, config.dim_model)
        self.tokens_per_timestep += 1

        # Visual feature processing
        if self.config.image_feature_keys:
            if not self.config.use_embeddings:
                if not hasattr(torchvision.models, config.vision_backbone):
                    raise ValueError(f"Vision backbone {config.vision_backbone} not found in torchvision.models")

                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, False],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d if config.pretrained_backbone_weights else nn.BatchNorm2d,
                )
                self.vision_backbone = nn.Sequential(*list(backbone_model.children())[:-1]) # Before final FC
                
                self.image_feature_proj = nn.Linear(config.vision_feature_dim * len(config.image_feature_keys), config.dim_model)
                self.tokens_per_timestep += 1 
            else: 
                self.vision_backbone = None 
                self.image_feature_proj = None 
                if config.embedding_dim is None or config.dim_model is None:
                     raise ValueError("embedding_dim and dim_model must be specified in config when use_embeddings is True.")
                if config.embedding_dim % config.dim_model != 0:
                    raise ValueError(f"embedding_dim ({config.embedding_dim}) must be divisible by dim_model ({config.dim_model}) for segmentation.")
                self.num_visual_tokens_per_cam = config.embedding_dim // config.dim_model
                self.tokens_per_timestep += self.num_visual_tokens_per_cam * len(config.image_feature_keys)
        else:
            self.vision_backbone = None
            self.image_feature_proj = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim_model))
        self.actual_transformer_seq_len = 1 + config.max_seq_len * self.tokens_per_timestep
        logging.info(f"FCIL Classifier: Original max_seq_len (timesteps): {config.max_seq_len}")
        logging.info(f"FCIL Classifier: Tokens per original timestep: {self.tokens_per_timestep}")
        logging.info(f"FCIL Classifier: Actual Transformer sequence length (incl. CLS): {self.actual_transformer_seq_len}")
        self.pos_embed = nn.Embedding(self.actual_transformer_seq_len, config.dim_model)

        self.encoder_layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()
        self.output_head = nn.Linear(config.dim_model, 1)

    def forward(self, obs_state_seq, act_seq, obs_feature_seq_dict=None, padding_mask=None):
        batch_size, T_orig, _ = obs_state_seq.shape

        state_embed_per_t = self.state_proj(obs_state_seq) 
        action_embed_per_t = self.action_proj(act_seq)     

        visual_tokens_all_cams_per_t_list = [] 

        if self.config.image_feature_keys and obs_feature_seq_dict:
            if not self.config.use_embeddings:
                all_cam_backbone_features = []
                for cam_key in self.config.image_feature_keys:
                    raw_image_seq = obs_feature_seq_dict[cam_key] 
                    bt, c, h, w = raw_image_seq.shape[0]*raw_image_seq.shape[1], raw_image_seq.shape[2], raw_image_seq.shape[3], raw_image_seq.shape[4]
                    raw_image_seq_reshaped = raw_image_seq.reshape(bt, c, h, w)
                    img_features_from_backbone = self.vision_backbone(raw_image_seq_reshaped) 
                    img_features_from_backbone = img_features_from_backbone.squeeze(-1).squeeze(-1) 
                    processed_cam_feature_seq = img_features_from_backbone.view(batch_size, T_orig, -1)
                    all_cam_backbone_features.append(processed_cam_feature_seq)
                concatenated_raw_visual_features = torch.cat(all_cam_backbone_features, dim=-1) 
                single_visual_token_per_t = self.image_feature_proj(concatenated_raw_visual_features)
                visual_tokens_all_cams_per_t_list.append(single_visual_token_per_t.unsqueeze(2))
            else: 
                for cam_key in self.config.image_feature_keys:
                    embedding_seq = obs_feature_seq_dict[cam_key] 
                    segmented_visual_tokens = embedding_seq.view(
                        batch_size, T_orig, self.num_visual_tokens_per_cam, self.config.dim_model
                    )
                    visual_tokens_all_cams_per_t_list.append(segmented_visual_tokens)
        
        input_token_list = [self.cls_token.expand(batch_size, -1, -1)] 
        for t in range(T_orig):
            input_token_list.append(state_embed_per_t[:, t:t+1, :])    
            input_token_list.append(action_embed_per_t[:, t:t+1, :]) 
            if visual_tokens_all_cams_per_t_list:
                for cam_segmented_tokens in visual_tokens_all_cams_per_t_list:
                    input_token_list.append(cam_segmented_tokens[:, t, :, :])
        
        encoder_input = torch.cat(input_token_list, dim=1) 
        actual_seq_len_for_transformer = encoder_input.shape[1]
        if actual_seq_len_for_transformer > self.config._actual_transformer_max_seq_len:
             logging.warning(f"Input sequence length {actual_seq_len_for_transformer} exceeds configured max {self.config._actual_transformer_max_seq_len}. Truncating.")
             encoder_input = encoder_input[:, :self.config._actual_transformer_max_seq_len, :]
             actual_seq_len_for_transformer = self.config._actual_transformer_max_seq_len

        positions = torch.arange(0, actual_seq_len_for_transformer, device=obs_state_seq.device).unsqueeze(0).expand(batch_size, -1)
        encoder_input += self.pos_embed(positions)

        transformer_key_padding_mask = None
        if padding_mask is not None:
            expanded_padding_mask_per_t = padding_mask.unsqueeze(2).expand(-1, -1, self.tokens_per_timestep)
            flattened_padding_mask_main_seq = expanded_padding_mask_per_t.reshape(batch_size, T_orig * self.tokens_per_timestep)
            cls_padding_part = torch.full((batch_size, 1), False, device=padding_mask.device)
            transformer_key_padding_mask = torch.cat([cls_padding_part, flattened_padding_mask_main_seq], dim=1)
            if transformer_key_padding_mask.shape[1] > actual_seq_len_for_transformer:
                transformer_key_padding_mask = transformer_key_padding_mask[:, :actual_seq_len_for_transformer]

        x = encoder_input.permute(1, 0, 2) 
        for layer in self.encoder_layers:
            x = layer(x, pos_embed=None, key_padding_mask=transformer_key_padding_mask)
        x = self.encoder_norm(x)
        cls_output = x[0] 
        logits = self.output_head(cls_output) 
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

        # Initialize torchmetrics metrics
        # Make sure they are on the same device as the model will be
        self.accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.precision_metric = torchmetrics.Precision(task="binary")
        self.recall_metric = torchmetrics.Recall(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary")


    def _infer_and_set_config_dims(self, dataset_stats):
        if self.config.state_dim is None and dataset_stats and "observation.state" in dataset_stats:
            self.config.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
        if self.config.action_dim is None and dataset_stats and "action" in dataset_stats:
            self.config.action_dim = dataset_stats["action"]["mean"].shape[0]

        if self.config.use_embeddings and self.config.embedding_dim is None:
            raise ValueError("embedding_dim must be set in config if use_embeddings is True.")
        
        if self.config.state_dim is None or self.config.action_dim is None :
            if self.config.input_features:
                 if "observation.state" in self.config.input_features and self.config.state_dim is None:
                     self.config.state_dim = self.config.input_features["observation.state"].shape[0]
                 if "action" in self.config.input_features and self.config.action_dim is None:
                     self.config.action_dim = self.config.input_features["action"].shape[0]
            if self.config.state_dim is None or self.config.action_dim is None:
                 raise ValueError(
                    "state_dim and action_dim must be provided in config or inferable from dataset_stats."
                )
        self.config.__post_init__()
        self.config.validate_features()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        # Reset metric states if needed, e.g., at the start of an evaluation epoch
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

    def to(self, *args, **kwargs):
        # Ensure metrics are moved to the correct device along with the model
        super().to(*args, **kwargs)
        self.accuracy_metric.to(*args, **kwargs)
        self.precision_metric.to(*args, **kwargs)
        self.recall_metric.to(*args, **kwargs)
        self.f1_metric.to(*args, **kwargs)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        obs_state_seq = batch['observation.state']
        act_seq = batch['action']
        labels = batch['label'] # Expected shape: [B, 1] or [B]
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

        loss = F.binary_cross_entropy_with_logits(logits, labels) # labels should be float for BCE

        # Calculate other metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int() # Convert probabilities to binary predictions (0 or 1)
        
        # Ensure labels are also integer type for torchmetrics
        # and have the same shape as preds (squeeze if labels is [B,1] and preds is [B])
        labels_int = labels.squeeze(-1).int() if labels.ndim > 1 and labels.shape[-1] == 1 else labels.int()
        preds_squeezed = preds.squeeze(-1) if preds.ndim > 1 and preds.shape[-1] == 1 else preds
        
        # Update metrics (they accumulate if not reset)
        # It's better to compute them in an eval loop rather than per-batch for training logging,
        # but we can return batch-wise metrics here.
        # For training, these will be batch-wise values. For eval, accumulate over an epoch then compute.
        batch_accuracy = self.accuracy_metric(preds_squeezed, labels_int)
        batch_precision = self.precision_metric(preds_squeezed, labels_int)
        batch_recall = self.recall_metric(preds_squeezed, labels_int)
        batch_f1 = self.f1_metric(preds_squeezed, labels_int)

        output_dict = {
            "loss": loss.item(),
            "accuracy": batch_accuracy.item(),
            "precision": batch_precision.item(),
            "recall": batch_recall.item(),
            "f1_score": batch_f1.item(),
        }
        return loss, output_dict

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
        return self.predict_failure_prob(batch)