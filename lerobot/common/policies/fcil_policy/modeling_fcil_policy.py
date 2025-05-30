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

import logging
from collections import deque
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.fcil_policy.configuration_fcil_policy import FCILPolicyConfig
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.configs.types import FeatureType


logger = logging.getLogger(__name__)

class FCILTransformerModel(nn.Module):
    def __init__(self, config: FCILPolicyConfig):
        super().__init__()
        self.config = config
        self.model_dim = config.model_dim
        if config.embedding_dim_in is None or config.model_dim is None:
            raise ValueError("embedding_dim_in and model_dim must be set in FCILPolicyConfig.")
        self.image_tokens_per_cam = config.embedding_dim_in // config.model_dim

        self.state_embed = nn.Linear(config.state_dim, config.model_dim)
        self.action_embed = nn.Linear(config.policy_action_dim, config.model_dim) 
        
        self.image_chunk_embed = nn.Linear(config.model_dim, config.model_dim) 
        
        self.fail_token_embed = nn.Parameter(torch.randn(1, 1, config.model_dim)) 
        
        self.pad_token_embed = nn.Parameter(torch.randn(1, 1, config.model_dim))

        self.pos_embed = nn.Embedding(config.max_seq_len, config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.n_heads,
            dim_feedforward=config.model_dim * 4, 
            dropout=config.dropout,
            activation=config.activation_function,
            batch_first=True, 
            norm_first=True   
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.final_transformer_norm = nn.LayerNorm(config.model_dim)
        self.action_pred_head = nn.Linear(config.model_dim, config.policy_action_dim)


    def _tokenize_timestep(self, state: torch.Tensor, obs_visual: Dict[str, torch.Tensor], action: torch.Tensor | None) -> List[torch.Tensor]:
        tokens = []
        tokens.append(self.state_embed(state)) 
        
        for cam_key in self.config.image_feature_keys: 
            cam_embedding = obs_visual[cam_key] 
            cam_chunks = cam_embedding.view(-1, self.image_tokens_per_cam, self.model_dim)
            for i in range(self.image_tokens_per_cam):
                tokens.append(self.image_chunk_embed(cam_chunks[:, i, :]))
        
        if action is not None:
            tokens.append(self.action_embed(action))
        else: 
            batch_size = state.shape[0]
            tokens.append(self.pad_token_embed.squeeze(1).expand(batch_size, -1)) # Using pad_token as a stand-in for "action query" or missing action
            
        return tokens 

    def forward(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        history_context_collated: List[Dict[str, Any] | None] = input_dict["history_context"]
        current_state_obs_collated: Dict[str, Any] = input_dict["current_state_obs"]
        fail_traj_context_collated_optional: List[Dict[str, Any]] | None = input_dict["fail_traj_context_optional"]
        is_recovery_mode_batch: torch.Tensor = input_dict["is_recovery_mode"] 

        batch_size = current_state_obs_collated['state'].shape[0]
        device = current_state_obs_collated['state'].device
        
        all_tokens_list: List[torch.Tensor] = [] 

        if fail_traj_context_collated_optional is not None: 
            for step_data_batch_fail in fail_traj_context_collated_optional: 
                all_tokens_list.extend(self._tokenize_timestep(
                    step_data_batch_fail['state'], 
                    step_data_batch_fail['observation_visual'], 
                    step_data_batch_fail['action']
                ))
            all_tokens_list.append(self.fail_token_embed.squeeze(1).expand(batch_size, -1))

        for step_data_batch_hist in history_context_collated: 
            if step_data_batch_hist is not None:
                 all_tokens_list.extend(self._tokenize_timestep(
                    step_data_batch_hist['state'], 
                    step_data_batch_hist['observation_visual'], 
                    step_data_batch_hist['action']
                ))
            else: 
                logger.error("Encountered None in collated history_context. This should be padded by collate_fn.")
                tokens_per_step_hist = 1 + len(self.config.image_feature_keys) * self.image_tokens_per_cam + 1 
                for _ in range(tokens_per_step_hist):
                    all_tokens_list.append(self.pad_token_embed.squeeze(1).expand(batch_size, -1))
        
        current_s_o_tokens = self._tokenize_timestep(
            current_state_obs_collated['state'],
            current_state_obs_collated['observation_visual'],
            action=None 
        )
        all_tokens_list.extend(current_s_o_tokens)
        
        input_sequence_unpadded = torch.stack(all_tokens_list, dim=1)
        
        current_seq_len = input_sequence_unpadded.shape[1]
        
        # Apply positional embeddings to the valid part of the sequence
        positions_unpadded = torch.arange(0, current_seq_len, device=device).unsqueeze(0).expand(batch_size, -1) 
        input_sequence_with_pos = input_sequence_unpadded + self.pos_embed(positions_unpadded)

        # Prepare final_input_sequence and src_key_padding_mask
        if current_seq_len < self.config.max_seq_len:
            num_pads_to_add = self.config.max_seq_len - current_seq_len
            pad_tensor_vals = self.pad_token_embed.expand(batch_size, num_pads_to_add, -1)
            final_input_sequence = torch.cat([input_sequence_with_pos, pad_tensor_vals], dim=1)
            
            # Mask: False for valid tokens, True for padding
            src_key_padding_mask = torch.zeros(batch_size, self.config.max_seq_len, dtype=torch.bool, device=device)
            src_key_padding_mask[:, current_seq_len:] = True
            
        elif current_seq_len > self.config.max_seq_len:
            logger.warning(f"Input sequence length {current_seq_len} > max_seq_len {self.config.max_seq_len}. Truncating.")
            final_input_sequence = input_sequence_with_pos[:, :self.config.max_seq_len, :]
            src_key_padding_mask = torch.zeros(batch_size, self.config.max_seq_len, dtype=torch.bool, device=device) # All False, no padding
        else: # current_seq_len == self.config.max_seq_len
            final_input_sequence = input_sequence_with_pos
            src_key_padding_mask = torch.zeros(batch_size, current_seq_len, dtype=torch.bool, device=device) # All False, no padding

        transformer_output = self.transformer_encoder(final_input_sequence, src_key_padding_mask=src_key_padding_mask) 
        
        transformer_output_norm = self.final_transformer_norm(transformer_output)

        pred_token_idx = min(current_seq_len - 1, self.config.max_seq_len - 1)        
        
        action_query_output_feature = transformer_output_norm[:, pred_token_idx , :] 
        
        predicted_action_and_done = self.action_pred_head(action_query_output_feature) 
        
        return predicted_action_and_done


class FCILPolicy(PreTrainedPolicy):
    config_class = FCILPolicyConfig
    name = "fcil_policy"

    def __init__(
        self,
        config: FCILPolicyConfig, # This config instance has been modified by make_policy
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config) # Pass the modified config to super
        self.config: FCILPolicyConfig = config

        if self.config.state_dim is None:
            if dataset_stats and "observation.state" in dataset_stats:
                self.config.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
            elif self.config.input_features.get("observation.state"): # Fallback to what factory might have set
                 self.config.state_dim = self.config.input_features["observation.state"].shape[0]
            else:
                raise ValueError("FCILPolicyConfig: state_dim cannot be None and not inferable.")

        if self.config.action_dim is None:
            if dataset_stats and "action" in dataset_stats: # Check raw action from dataset
                self.config.action_dim = dataset_stats["action"]["mean"].shape[0]
            elif self.config.output_features.get("action") and self.config.output_features["action"].type == FeatureType.ACTION:
                self.config.action_dim = self.config.output_features["action"].shape[0]
            elif self.config.input_features.get("action"): # Fallback to historical action if in input_features
                self.config.action_dim = self.config.input_features["action"].shape[0]
            else:
                raise ValueError("FCILPolicyConfig: action_dim cannot be None and not inferable.")

        self.config.validate_features()

        self.normalize_inputs = Normalize(self.config.input_features, self.config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(self.config.output_features, self.config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(self.config.output_features, self.config.normalization_mapping, dataset_stats)
        
        self.model = FCILTransformerModel(self.config)
        
        self._history_queue: deque = deque(maxlen=self.config.history_len)
        self._fail_traj_for_recovery: List[Dict[str, Any]] | None = None
        self._is_in_recovery_mode_inference: bool = False
        self.reset()


    def get_optim_params(self) -> List[Dict[str, Any]]:
        return [{"params": self.parameters()}]
        
    def reset(self):
        self._history_queue.clear()
        self._fail_traj_for_recovery = None
        self._is_in_recovery_mode_inference = False
        for _ in range(self.config.history_len):
            self._history_queue.append(None) 

    def set_recovery_mode(self, failed_trajectory_data: List[Dict[str, Any]]):
        self._fail_traj_for_recovery = failed_trajectory_data
        self._is_in_recovery_mode_inference = True

    @torch.no_grad()
    def select_action(self, observation_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        device = get_device_from_parameters(self.model)
        batch_size = observation_batch["state"].shape[0]


        current_state_for_norm = {"observation.state": observation_batch["state"]}
        norm_current_state = self.normalize_inputs(current_state_for_norm)["observation.state"]
        norm_current_obs_visual = observation_batch["observation_visual"]


        current_s_o_for_policy = {
            "state": norm_current_state.to(device),
            "observation_visual": {k: v.to(device) for k,v in norm_current_obs_visual.items()}
        }
        
        policy_history_context_collated: List[Dict[str, Any] | None] = []
        
        temp_history_list = list(self._history_queue) 
        
        if temp_history_list : # Check if deque is not empty
            history_len_model = len(temp_history_list)
            for i in range(history_len_model):
                current_hist_step_data = temp_history_list[i] 

                if current_hist_step_data is not None:
                    hist_s_b = current_hist_step_data['state'].unsqueeze(0).expand(batch_size, -1).to(device)
                    hist_a_b = current_hist_step_data['action'].unsqueeze(0).expand(batch_size, -1).to(device)
                    hist_o_v_b = {
                        k: v.unsqueeze(0).expand(batch_size, *v.shape).to(device)
                        for k, v in current_hist_step_data['observation_visual'].items()
                    }
                    policy_history_context_collated.append({
                        "state": hist_s_b, "observation_visual": hist_o_v_b, "action": hist_a_b
                    })
                else: 
                    s_dim = self.config.state_dim
                    a_dim = self.config.action_dim
                    # Use norm_current_state for dtype and device for consistency if available
                    dtype_ref = norm_current_state.dtype if norm_current_state is not None else torch.float32
                    device_ref = norm_current_state.device if norm_current_state is not None else device

                    pad_s = torch.zeros(batch_size, s_dim, device=device_ref, dtype=dtype_ref)
                    pad_a = torch.zeros(batch_size, a_dim, device=device_ref, dtype=dtype_ref) 
                    pad_o_v = {
                        k: torch.zeros(batch_size, self.config.embedding_dim_in, device=device_ref, dtype=v.dtype if hasattr(v,'dtype') else torch.float32) 
                        for k,v in norm_current_obs_visual.items()
                    }
                    policy_history_context_collated.append({
                        "state": pad_s, "observation_visual": pad_o_v, "action": pad_a
                    })
        else: # history queue is empty (should be filled by Nones in reset)
             for _ in range(self.config.history_len):
                s_dim = self.config.state_dim
                a_dim = self.config.action_dim
                dtype_ref = norm_current_state.dtype if norm_current_state is not None else torch.float32
                device_ref = norm_current_state.device if norm_current_state is not None else device
                pad_s = torch.zeros(batch_size, s_dim, device=device_ref, dtype=dtype_ref)
                pad_a = torch.zeros(batch_size, a_dim, device=device_ref, dtype=dtype_ref)
                pad_o_v = {
                    k: torch.zeros(batch_size, self.config.embedding_dim_in, device=device_ref, dtype=v.dtype if hasattr(v,'dtype') else torch.float32) 
                    for k,v in norm_current_obs_visual.items()
                }
                policy_history_context_collated.append({
                    "state": pad_s, "observation_visual": pad_o_v, "action": pad_a
                })


        policy_fail_traj_context_collated = None
        if self._is_in_recovery_mode_inference and self._fail_traj_for_recovery:
            policy_fail_traj_context_collated = []
            for step_data in self._fail_traj_for_recovery: 
                fail_s_b = step_data['state'].unsqueeze(0).expand(batch_size, -1)
                fail_a_b = step_data['action'].unsqueeze(0).expand(batch_size, -1)
                fail_o_v_b = {
                    k: v.unsqueeze(0).expand(batch_size, *v.shape)
                    for k, v in step_data['observation_visual'].items()
                }
                policy_fail_traj_context_collated.append({
                    "state": fail_s_b, "observation_visual": fail_o_v_b, "action": fail_a_b
                })
            num_actual_fail_steps = len(policy_fail_traj_context_collated)
            num_pad_fail_steps = self.config.max_fail_traj_len - num_actual_fail_steps
            if num_pad_fail_steps > 0:
                s_dim = self.config.state_dim
                a_dim = self.config.action_dim
                dtype_ref = norm_current_state.dtype if norm_current_state is not None else torch.float32
                device_ref = norm_current_state.device if norm_current_state is not None else device
                pad_s_fail = torch.zeros(batch_size, s_dim, device=device_ref, dtype=dtype_ref)
                pad_a_fail = torch.zeros(batch_size, a_dim, device=device_ref, dtype=dtype_ref)
                pad_o_v_fail = {
                    k: torch.zeros(batch_size, self.config.embedding_dim_in, device=device_ref, dtype=v.dtype if hasattr(v,'dtype') else torch.float32) 
                    for k,v in norm_current_obs_visual.items()
                }
                for _ in range(num_pad_fail_steps):
                    policy_fail_traj_context_collated.append({
                        "state": pad_s_fail, "observation_visual": pad_o_v_fail, "action": pad_a_fail
                    })

        
        is_recovery_mode_tensor = torch.tensor([self._is_in_recovery_mode_inference] * batch_size, dtype=torch.bool, device=device)

        input_dict_for_policy = {
            "history_context": policy_history_context_collated,
            "current_state_obs": current_s_o_for_policy,
            "fail_traj_context_optional": policy_fail_traj_context_collated,
            "is_recovery_mode": is_recovery_mode_tensor 
        }

        predicted_action_done_normalized = self.model(input_dict_for_policy) 
        
        unnormalized_action_done = self.unnormalize_outputs({"action_pred": predicted_action_done_normalized})["action_pred"]
        
        action_pred = unnormalized_action_done[:, :-1] 
        
        action_to_store_in_history = predicted_action_done_normalized[:, :-1].detach().clone()

        # For B=1, this is fine. For B > 1, history queue needs to be a list of deques, or handle batching.
        self._history_queue.append({
            "state": norm_current_state[0].detach().clone() if batch_size == 1 else norm_current_state.detach().clone(), # Store normalized
            "observation_visual": {k:v[0].detach().clone() if batch_size == 1 else v.detach().clone() for k,v in norm_current_obs_visual.items()}, # Store normalized
            "action": action_to_store_in_history[0] if batch_size == 1 else action_to_store_in_history # Store normalized action part
        })
        
        return action_pred 

    def forward(self, batch: Tuple[Dict[str, Any], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        policy_input_batch_collated, target_action_and_done_batch = batch
        
        predicted_action_done_normalized = self.model(policy_input_batch_collated) 
        
        target_action_and_done_normalized = self.normalize_targets(
            {"action_pred": target_action_and_done_batch} 
        )["action_pred"] 

        loss = F.mse_loss(predicted_action_done_normalized, target_action_and_done_normalized, reduction="none")
        
        sample_weights = torch.ones_like(loss[:,0], device=loss.device) 
        sample_weights[policy_input_batch_collated["is_recovery_mode"]] = self.config.recovery_loss_weight
        
        # Compute unweighted loss for debugging
        unweighted_loss = loss.mean()
        
        loss = (loss.mean(dim=1) * sample_weights).mean() 
        
        # Calculate proportion of recovery samples
        recovery_proportion = policy_input_batch_collated["is_recovery_mode"].float().mean().item()
        
        loss_dict = {
            "mse_loss": loss.item(),
            "unweighted_mse_loss": unweighted_loss.item(),
            "recovery_proportion": recovery_proportion
        }
        return loss, loss_dict