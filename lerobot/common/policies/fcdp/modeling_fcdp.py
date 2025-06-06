# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# lerobot/lerobot/common/policies/fcdp/modeling_fcdp.py
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
import math
from collections import deque
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.fcdp.configuration_fcdp import FCDPConfig
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.configs.types import FeatureType

logger = logging.getLogger(__name__)


class TrajectoryTransformerEncoder(nn.Module):
    """Transformer encoder that processes trajectory context and outputs a conditioning vector."""
    
    def __init__(self, config: FCDPConfig):
        super().__init__()
        self.config = config
        self.model_dim = config.model_dim
        if config.embedding_dim_in is None or config.model_dim is None:
            raise ValueError("embedding_dim_in and model_dim must be set in FCDPConfig.")
        self.image_tokens_per_cam = config.embedding_dim_in // config.model_dim

        self.state_embed = nn.Linear(config.state_dim, config.model_dim)
        self.action_embed = nn.Linear(config.policy_action_dim, config.model_dim)
        
        self.image_chunk_embed = nn.Linear(config.model_dim, config.model_dim)
        
        self.fail_token_embed = nn.Parameter(torch.randn(1, 1, config.model_dim))
        
        self.pad_token_embed = nn.Parameter(torch.randn(1, 1, config.model_dim))

        self.pos_embed = nn.Embedding(512, config.model_dim)  # Max sequence length

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
        
        # Project to global conditioning dimension
        self.cond_proj = nn.Linear(config.model_dim, config.global_cond_dim)

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
            tokens.append(self.pad_token_embed.squeeze(1).expand(batch_size, -1))
            
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
        
        # Apply positional embeddings
        positions_unpadded = torch.arange(0, current_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        input_sequence_with_pos = input_sequence_unpadded + self.pos_embed(positions_unpadded)

        # No padding needed since we use the last token
        transformer_output = self.transformer_encoder(input_sequence_with_pos)
        transformer_output_norm = self.final_transformer_norm(transformer_output)
        
        # Take the last token as the global representation
        global_feature = transformer_output_norm[:, -1, :]
        
        # Project to conditioning dimension
        global_cond = self.cond_proj(global_feature)
        
        return global_cond


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """1D residual block with FiLM conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        
        # FiLM layers
        self.film = nn.Linear(cond_dim, out_channels * 2)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        residue = self.residual_conv(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        
        # Apply FiLM
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)  # (B, out_channels, 1)
        shift = shift.unsqueeze(-1)  # (B, out_channels, 1)
        x = x * (1 + scale) + shift
        
        x = F.mish(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.mish(x)
        
        return x + residue


class DiffusionUnet1D(nn.Module):
    """1D U-Net for diffusion policy action prediction."""
    
    def __init__(self, config: FCDPConfig):
        super().__init__()
        self.config = config
        
        # Input/output dimensions
        input_dim = config.policy_action_dim
        cond_dim = config.global_cond_dim + config.diffusion_step_embed_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv1d(input_dim, config.down_dims[0], config.kernel_size, padding=config.kernel_size // 2)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        in_channels = config.down_dims[0]
        for i, out_channels in enumerate(config.down_dims):
            self.down_blocks.append(
                ConditionalResidualBlock1D(in_channels, out_channels, cond_dim, config.kernel_size, config.n_groups)
            )
            if i < len(config.down_dims) - 1:  # Don't downsample on last block
                self.down_blocks.append(nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1))
            in_channels = out_channels
        
        # Middle block
        self.mid_block = ConditionalResidualBlock1D(
            config.down_dims[-1], config.down_dims[-1], cond_dim, config.kernel_size, config.n_groups
        )
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        reversed_dims = list(reversed(config.down_dims))
        for i in range(len(reversed_dims) - 1):
            in_channels = reversed_dims[i]
            out_channels = reversed_dims[i + 1]
            
            # Upsample
            self.up_blocks.append(nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1))
            # Residual block with skip connection
            self.up_blocks.append(
                ConditionalResidualBlock1D(in_channels + out_channels, out_channels, cond_dim, config.kernel_size, config.n_groups)
            )
        
        # Final convolution
        self.conv_out = nn.Conv1d(config.down_dims[0], input_dim, 1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: (B, horizon, action_dim + 1) - noisy action trajectory
            timestep: (B,) - diffusion timestep
            global_cond: (B, global_cond_dim) - conditioning from transformer
        Returns:
            (B, horizon, action_dim + 1) - predicted noise
        """
        # Reshape for conv1d: (B, horizon, action_dim) -> (B, action_dim, horizon)
        x = sample.transpose(1, 2)
        
        # Embed timestep
        t_emb = self.time_mlp(timestep.float())
        
        # Combine conditioning
        cond = torch.cat([global_cond, t_emb], dim=-1)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Encoder with skip connections
        skip_features = []
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, ConditionalResidualBlock1D):
                x = block(x, cond)
                if i < len(self.down_blocks) - 1:  # Save skip connection except for last block
                    skip_features.append(x)
            else:  # Downsampling conv
                x = block(x)
        
        # Middle
        x = self.mid_block(x, cond)
        
        # Decoder with skip connections
        skip_idx = len(skip_features) - 1
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, nn.ConvTranspose1d):  # Upsample
                x = block(x)
            else:  # Residual block
                # Concatenate skip connection
                x = torch.cat([x, skip_features[skip_idx]], dim=1)
                x = block(x, cond)
                skip_idx -= 1
        
        # Final conv
        x = self.conv_out(x)
        
        # Reshape back: (B, action_dim, horizon) -> (B, horizon, action_dim)
        return x.transpose(1, 2)


class FCDPModel(nn.Module):
    """Main model combining transformer encoder and diffusion UNet."""
    
    def __init__(self, config: FCDPConfig):
        super().__init__()
        self.config = config
        
        self.transformer_encoder = TrajectoryTransformerEncoder(config)
        self.unet = DiffusionUnet1D(config)
    
    def forward(self, policy_input_dict: Dict[str, Any], noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            policy_input_dict: Dictionary from FCDPDataset containing context
            noisy_actions: (B, horizon, action_dim + 1) - noisy action trajectory
            timesteps: (B,) - diffusion timesteps
        Returns:
            (B, horizon, action_dim + 1) - predicted noise
        """
        # Get global conditioning from transformer
        global_cond = self.transformer_encoder(policy_input_dict)
        
        # Predict noise with UNet
        pred_noise = self.unet(noisy_actions, timesteps, global_cond)
        
        return pred_noise
    
    def generate_actions(self, policy_input_dict: Dict[str, Any], scheduler: DDPMScheduler, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Generate action trajectory using reverse diffusion.
        
        Args:
            policy_input_dict: Dictionary from FCDPDataset containing context
            scheduler: Noise scheduler for diffusion
            generator: Random generator for reproducibility
        Returns:
            (B, horizon, action_dim + 1) - clean action trajectory
        """
        # Get global conditioning
        global_cond = self.transformer_encoder(policy_input_dict)
        
        batch_size = global_cond.shape[0]
        device = global_cond.device
        
        # Start with pure noise
        action_trajectory = torch.randn(
            (batch_size, self.config.horizon, self.config.policy_action_dim),
            device=device,
            generator=generator
        )
        
        # Reverse diffusion loop
        scheduler.set_timesteps(self.config.num_inference_steps or self.config.num_train_timesteps)
        
        for t in scheduler.timesteps:
            # Predict noise
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.unet(action_trajectory, timesteps, global_cond)
            
            # Denoise
            action_trajectory = scheduler.step(pred_noise, t, action_trajectory, generator=generator).prev_sample
        
        return action_trajectory


class FCDPPolicy(PreTrainedPolicy):
    """Failure-Conditioned Diffusion Policy."""
    
    config_class = FCDPConfig
    name = "fcdp"

    def __init__(
        self,
        config: FCDPConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config: FCDPConfig = config

        # Validate and set dimensions
        if self.config.state_dim is None:
            if dataset_stats and "observation.state" in dataset_stats:
                self.config.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
            elif self.config.input_features.get("observation.state"):
                 self.config.state_dim = self.config.input_features["observation.state"].shape[0]
            else:
                raise ValueError("FCDPConfig: state_dim cannot be None and not inferable.")

        if self.config.action_dim is None:
            if dataset_stats and "action" in dataset_stats:
                self.config.action_dim = dataset_stats["action"]["mean"].shape[0]
            elif self.config.output_features.get("action") and self.config.output_features["action"].type == FeatureType.ACTION:
                self.config.action_dim = self.config.output_features["action"].shape[0]
            elif self.config.input_features.get("action"):
                self.config.action_dim = self.config.input_features["action"].shape[0]
            else:
                raise ValueError("FCDPConfig: action_dim cannot be None and not inferable.")

        self.config.validate_features()

        # Normalization layers
        self.normalize_targets = Normalize(
            self.config.output_features, self.config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            self.config.output_features, self.config.normalization_mapping, dataset_stats
        )

        # Main model
        self.model = FCDPModel(config)
        
        # Noise scheduler
        self.noise_scheduler = self._make_noise_scheduler()
        
        # For inference
        self.reset()
        
        # DINOv2 for real-time image encoding during inference
        self.dinov2_ready = False

    def _make_noise_scheduler(self) -> DDPMScheduler | DDIMScheduler:
        """Create noise scheduler based on config."""
        if self.config.noise_scheduler_type == "DDPM":
            return DDPMScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
                beta_schedule=self.config.beta_schedule,
                prediction_type=self.config.prediction_type,
            )
        elif self.config.noise_scheduler_type == "DDIM":
            return DDIMScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
                beta_schedule=self.config.beta_schedule,
                prediction_type=self.config.prediction_type,
            )
        else:
            raise ValueError(f"Unsupported noise scheduler type {self.config.noise_scheduler_type}")

    def reset(self):
        """Reset internal state for new episode."""
        self._observation_history = deque(maxlen=self.config.history_len)
        self._action_history = deque(maxlen=self.config.history_len)
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._recovery_mode = False
        self._failure_trajectory = None

    def set_recovery_mode(self, failure_trajectory: List[Dict[str, Any]]):
        """Switch to recovery mode with failure context."""
        self._recovery_mode = True
        self._failure_trajectory = failure_trajectory

    def forward(self, batch: Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Training forward pass."""
        policy_input_dict, target_action_trajectory, target_padding_mask = batch
        
        # Normalize target actions
        normalized_targets = self.normalize_targets({"action_pred": target_action_trajectory})
        target_action_trajectory = normalized_targets["action_pred"]
        
        # Sample timesteps
        batch_size = target_action_trajectory.shape[0]
        timesteps = torch.randint(
            0, self.config.num_train_timesteps, (batch_size,), device=target_action_trajectory.device
        )
        
        # Add noise
        noise = torch.randn_like(target_action_trajectory)
        noisy_actions = self.noise_scheduler.add_noise(target_action_trajectory, noise, timesteps)
        
        # Predict noise
        pred_noise = self.model(policy_input_dict, noisy_actions, timesteps)
        
        # Compute loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = target_action_trajectory
        else:
            raise ValueError(f"Unknown prediction type {self.config.prediction_type}")
        
        # Apply padding mask if configured
        if self.config.do_mask_loss_for_padding:
            loss = F.mse_loss(pred_noise, target, reduction='none')
            loss = loss * target_padding_mask.unsqueeze(-1)  # Mask padded positions
            
            # Apply recovery loss weight for recovery samples
            is_recovery = policy_input_dict["is_recovery_mode"].float()
            sample_weights = torch.where(
                is_recovery.bool(),
                torch.ones_like(is_recovery) * self.config.recovery_loss_weight,
                torch.ones_like(is_recovery)
            )
            loss = loss.mean(dim=[1, 2]) * sample_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(pred_noise, target)
        
        output_dict = {"loss": loss}
        return loss, output_dict

    @torch.no_grad()
    def select_action(self, observation_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action for inference."""
        # Initialize DINOv2 if needed
        if not self.dinov2_ready:
            self._initialize_dinov2()
        
        # Update observation history
        self._update_observation_history(observation_batch)
        
        # Check if we need to generate new actions
        if len(self._action_queue) == 0:
            # Prepare policy input dict
            policy_input_dict = self._prepare_policy_input_dict()
            
            # Generate action trajectory
            action_trajectory = self.model.generate_actions(policy_input_dict, self.noise_scheduler)
            
            # Unnormalize
            action_trajectory = self.unnormalize_outputs({"action_pred": action_trajectory})["action_pred"]
            
            # Extract n_action_steps and queue them
            for i in range(self.config.n_action_steps):
                self._action_queue.append(action_trajectory[:, i])
        
        # Pop and return next action
        action = self._action_queue.popleft()
        
        # Update action history (without done flag for history)
        self._action_history.append(action)
        
        return action[:, :-1]  # Remove done flag for execution

    def _initialize_dinov2(self):
        """Initialize DINOv2 for real-time image encoding."""
        device = get_device_from_parameters(self.model)
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", self.config.dinov2_model_name)
        self.dinov2_model = self.dinov2_model.to(device).eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dinov2_ready = True

    def _update_observation_history(self, observation_batch: Dict[str, torch.Tensor]):
        """Update observation history with new observation."""
        # Process images through DINOv2
        visual_features = {}
        for cam_key in self.config.image_feature_keys:
            if cam_key in observation_batch:
                image = observation_batch[cam_key]
                image = self.image_transform(image)
                with torch.no_grad():
                    features = self.dinov2_model(image.unsqueeze(0))
                visual_features[cam_key] = features.squeeze(0)
        
        obs_dict = {
            "state": observation_batch["observation.state"],
            "observation_visual": visual_features
        }
        
        self._observation_history.append(obs_dict)

    def _prepare_policy_input_dict(self) -> Dict[str, Any]:
        """Prepare input dict for model from observation/action history."""
        device = get_device_from_parameters(self.model)
        
        # Prepare history context
        history_context = []
        for i in range(self.config.history_len):
            if i < len(self._observation_history):
                obs = self._observation_history[i]
                if i < len(self._action_history):
                    action = self._action_history[i]
                    # Add done flag (0 for history)
                    action_with_done = torch.cat([action, torch.zeros(1, device=device)])
                else:
                    action_with_done = None
                
                history_context.append({
                    "state": obs["state"],
                    "observation_visual": obs["observation_visual"],
                    "action": action_with_done
                })
            else:
                history_context.append(None)
        
        # Current observation
        current_obs = self._observation_history[-1]
        
        # Prepare failure context
        if self._recovery_mode and self._failure_trajectory:
            fail_traj_context = self._failure_trajectory
        else:
            # Padding
            fail_traj_context = []
            for _ in range(self.config.max_fail_traj_len):
                fail_traj_context.append({
                    'state': torch.zeros(self.config.state_dim, device=device),
                    'action': torch.zeros(self.config.policy_action_dim, device=device),
                    'observation_visual': {
                        k: torch.zeros(self.config.embedding_dim_in, device=device)
                        for k in self.config.image_feature_keys
                    }
                })
        
        # Build policy input dict
        policy_input_dict = {
            "history_context": history_context,
            "current_state_obs": current_obs,
            "fail_traj_context_optional": fail_traj_context,
            "is_recovery_mode": torch.tensor([self._recovery_mode], device=device, dtype=torch.bool),
        }
        
        # Add batch dimension
        policy_input_dict = self._add_batch_dimension(policy_input_dict)
        
        return policy_input_dict

    def _add_batch_dimension(self, policy_input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add batch dimension to all tensors in the dict."""
        batched_dict = {}
        
        for key, value in policy_input_dict.items():
            if key == "history_context":
                batched_list = []
                for item in value:
                    if item is not None:
                        batched_item = {
                            "state": item["state"].unsqueeze(0),
                            "observation_visual": {k: v.unsqueeze(0) for k, v in item["observation_visual"].items()},
                        }
                        if item.get("action") is not None:
                            batched_item["action"] = item["action"].unsqueeze(0)
                        batched_list.append(batched_item)
                    else:
                        batched_list.append(None)
                batched_dict[key] = batched_list
            elif key == "current_state_obs":
                batched_dict[key] = {
                    "state": value["state"].unsqueeze(0),
                    "observation_visual": {k: v.unsqueeze(0) for k, v in value["observation_visual"].items()},
                }
            elif key == "fail_traj_context_optional":
                batched_list = []
                for item in value:
                    batched_item = {
                        "state": item["state"].unsqueeze(0),
                        "action": item["action"].unsqueeze(0),
                        "observation_visual": {k: v.unsqueeze(0) for k, v in item["observation_visual"].items()},
                    }
                    batched_list.append(batched_item)
                batched_dict[key] = batched_list
            elif isinstance(value, torch.Tensor):
                batched_dict[key] = value.unsqueeze(0)
            else:
                batched_dict[key] = value
        
        return batched_dict

    def get_optim_params(self) -> dict:
        """Get parameters for optimizer."""
        return self.model.parameters()