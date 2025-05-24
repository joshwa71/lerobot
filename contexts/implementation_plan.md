
---
## Overall Implementation Strategy

1.  **Dataset Handling (Done):**
    *   We'll need a way to load both standard successful trajectories and paired (failure, success) trajectories.
    *   The `LeRobotDataset` will need to be adapted or wrapped to provide these two types of samples to the policy.

2.  **Failure Classifier ($f_\phi$):**
    *   This can be a separate model, trained first or jointly. For simplicity, let's assume it's trained first and then used during policy inference.
    *   It will take a trajectory (sequence of states/actions) as input and output a failure probability.
    *   We can leverage components of existing architectures (e.g., a transformer encoder similar to ACT's vision or VAE encoder).

3.  **FCIL Policy (ACT-based, $\pi_\theta$):**
    *   The ACT architecture is a good candidate. We'll need to modify its input processing to accept the conditional information ($\tau_{fail}$, $z_{fail}$) during recovery mode.
    *   `z_{fail}` will be a learnable embedding.
    *   $\tau_{fail}$ will be encoded into a fixed-size representation.

4.  **Training Script:**
    *   A new training script (`train_fcil.py`) will be the cleanest approach.
    *   It will manage loading the two types of data (standard success, paired failure-success).
    *   It will compute the combined loss $\mathcal{L}_{total} = \mathcal{L}_{standard} + \lambda \mathcal{L}_{recovery}$.

5.  **Inference/Evaluation:**
    *   The evaluation script (`eval.py`) will need to incorporate the failure classifier and the policy's mode-switching logic.

---
## Detailed Implementation Steps

### 1. Dataset Augmentation and Loading (Done)

This is the most critical part. LeRobot's `LeRobotDataset` loads individual frames and uses `delta_timestamps` to create sequences. We need to handle:
    a. Standard successful trajectories.
    b. Paired (failure, then successful recovery) trajectories.

**File to Modify/Extend:** `lerobot/common/datasets/lerobot_dataset.py` (potentially a new dataset class inheriting from it or a wrapper) and a custom sampler.

**Approach:**

*   **`FCILCombinedDataset(torch.utils.data.Dataset)`:**
    *   **Initialization:**
        *   Takes two `repo_id` arguments: `success_repo_id` (e.g., `USER/success_100`) and `mixed_repo_id` (e.g., `USER/mixed_200`).
        *   Internally creates two `LeRobotDataset` instances: `self.success_ds` and `self.mixed_ds`.
        *   Reads `episodes.jsonl` from `self.mixed_ds.meta.episodes` to identify fail/success pairs. We assume episode `2k` is a failure and `2k+1` is its corresponding success.
        *   Builds an internal index mapping:
            *   Indices `0` to `len(self.success_ds) - 1` map to standard successes.
            *   Indices `len(self.success_ds)` onwards map to the *successful recovery trajectories* from `self.mixed_ds`. For each such recovery trajectory, store the `episode_index` of its preceding failure.

    *   **`__len__`:** `len(self.success_ds) + len(self.mixed_ds.meta.episodes) // 2`. (We only sample based on the success parts of the mixed dataset).

    *   **`__getitem__(self, idx)`:**
        *   **Standard Mode Sample (from `self.success_ds`):**
            *   If `idx < len(self.success_ds)`:
                *   `item = self.success_ds[idx]` (this already returns a sequence based on `delta_timestamps`).
                *   Return `{'obs_seq': item_obs, 'act_seq': item_act, 'is_recovery': False, 'failed_traj_obs': None, 'failed_traj_act': None}`. (obs/act here means sequences for training ACT).
        *   **Recovery Mode Sample (from `self.mixed_ds`):**
            *   If `idx >= len(self.success_ds)`:
                *   Adjust `idx` to map into `self.mixed_ds`. Let `pair_idx = idx - len(self.success_ds)`.
                *   The successful recovery episode index is `success_ep_idx = 2 * pair_idx + 1`.
                *   The failed episode index is `failure_ep_idx = 2 * pair_idx`.
                *   To get the sample for ACT from the successful recovery trajectory:
                    *   Find a random frame `frame_num_in_success_ep` within `success_ep_idx`.
                    *   Map this to a global index `global_idx_s` in `self.mixed_ds.hf_dataset`.
                    *   `item_s = self.mixed_ds[global_idx_s]`. This gives `obs_seq_s` and `act_seq_s`.
                *   To get the failed trajectory $\tau_{fail}$:
                    *   Fetch the last N (state, action) pairs from `failure_ep_idx`.
                    *   Helper function `get_full_episode_data(dataset_instance, episode_idx, last_n_steps)`:
                        ```python
                        # Snippet for get_full_episode_data
                        # Inside FCILCombinedDataset or a utility
                        def get_full_episode_data(self, ds_instance: LeRobotDataset, ep_idx: int, last_n_steps: int):
                            # Find start and end global indices for this episode
                            ep_frames_start = ds_instance.episode_data_index["from"][ep_idx].item()
                            ep_frames_end = ds_instance.episode_data_index["to"][ep_idx].item()
                            
                            actual_start = max(ep_frames_start, ep_frames_end - last_n_steps)
                            
                            # Extract all relevant columns for these frames
                            # This part can be slow if not optimized, consider direct hf_dataset access.
                            obs_list = []
                            act_list = []
                            # Iterate through global indices for the episode segment
                            for i in range(actual_start, ep_frames_end):
                                # __getitem__ is designed for sequences, so this is a bit inefficient
                                # if we only need raw state/action.
                                # A more direct way might be to access ds_instance.hf_dataset
                                # and then apply normalization.
                                # For simplicity here, let's assume we can get individual frames easily
                                # or adapt __getitem__ to support fetching single (s,a) after normalization.
                                frame_data = ds_instance.hf_dataset[i] # Gets raw data
                                # Manually build what's needed (simplified)
                                # This needs careful handling of normalization and feature extraction
                                # matching what the policy expects.
                                obs_state = torch.tensor(frame_data['observation.state'])
                                # If images are needed for failure context, they need to be loaded and processed.
                                # For now, let's assume failure context only uses 'observation.state'.
                                action = torch.tensor(frame_data['action'])
                                obs_list.append(obs_state) # Or a dict if multiple obs modalities
                                act_list.append(action)
                            
                            # Pad if shorter than last_n_steps
                            num_missing = last_n_steps - len(obs_list)
                            if num_missing > 0:
                                # Pad with first state/action
                                obs_padding = [obs_list[0]] * num_missing if obs_list else [torch.zeros_like(self._get_dummy_state_shape())] * num_missing
                                act_padding = [act_list[0]] * num_missing if act_list else [torch.zeros_like(self._get_dummy_action_shape())] * num_missing
                                obs_list = obs_padding + obs_list
                                act_list = act_padding + act_list
                                
                            return torch.stack(obs_list), torch.stack(act_list)
                        ```
                    *   `failed_obs_seq, failed_act_seq = self.get_full_episode_data(self.mixed_ds, failure_ep_idx, N_context_window)`
                *   Return `{'obs_seq': item_s_obs, 'act_seq': item_s_act, 'is_recovery': True, 'failed_traj_obs': failed_obs_seq, 'failed_traj_act': failed_act_seq}`.

    *   **Collate Function:** The default `torch.utils.data.dataloader.default_collate` should mostly work if `None` values for `failed_traj_*` are handled correctly (e.g., batched as `None` or a batch of sentinel values). You might need a custom collate function to pad `failed_traj_*` if their lengths vary or to handle the `None`s.

---
### 2. Failure Classifier $f_\phi$

**New File:** `lerobot/common/policies/failure_classifier.py` (or similar)

```python
# Snippet for lerobot/common/policies/failure_classifier.py
import torch
import torch.nn as nn
from lerobot.common.policies.act.modeling_act import ACTEncoder # Reusing ACT's encoder

class TrajectoryFailureClassifier(nn.Module):
    def __init__(self, state_dim, action_dim, config): # config for transformer hyperparams
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config # ACTConfig or a dedicated ClassifierConfig

        # Project (state, action) to model dimension
        self.input_proj = nn.Linear(state_dim + action_dim, config.dim_model)
        
        # Transformer encoder (reusing ACT's encoder structure)
        self.encoder = ACTEncoder(config) # Pass appropriate ACTConfig or subset
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim_model))
        
        # Positional encoding (fixed or learned)
        # Assuming max_seq_len is known or handled by ACTEncoder's pos_enc if it's fixed
        self.pos_embed = nn.Embedding(config.max_classifier_seq_len + 1, config.dim_model) # +1 for CLS

        # Output head
        self.output_head = nn.Linear(config.dim_model, 1) # Binary classification

    def forward(self, obs_seq, act_seq, key_padding_mask=None):
        # obs_seq: (batch, seq_len, state_dim)
        # act_seq: (batch, seq_len, action_dim)
        # key_padding_mask: (batch, seq_len) - True for padded elements
        
        batch_size, seq_len, _ = obs_seq.shape
        
        # Concatenate state and action, then project
        sa_concat = torch.cat([obs_seq, act_seq], dim=-1)
        sa_embed = self.input_proj(sa_concat) # (batch, seq_len, dim_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_tokens, sa_embed], dim=1) # (batch, seq_len+1, dim_model)
        
        # Add positional embeddings
        positions = torch.arange(0, seq_len + 1, device=obs_seq.device).unsqueeze(0).expand(batch_size, -1)
        encoder_input += self.pos_embed(positions)
        
        # Prepare key_padding_mask for transformer if provided
        if key_padding_mask is not None:
            # Add a False for the CLS token (CLS token is never padding)
            cls_padding_mask = torch.full((batch_size, 1), False, device=key_padding_mask.device)
            full_key_padding_mask = torch.cat([cls_padding_mask, key_padding_mask], dim=1)
        else:
            full_key_padding_mask = None
            
        # Transformer expects (seq_len, batch, dim)
        encoder_input = encoder_input.permute(1, 0, 2)
        
        # Forward through encoder
        # Note: ACTEncoder's positional embedding is added internally if its pos_embed arg is None,
        # or it uses the passed pos_embed. Here we added it manually, so pass it.
        # If ACTEncoder adds its own, ensure no double addition or pass pos_embed=None.
        # For simplicity, let's assume ACTEncoder can take key_padding_mask.
        encoded_features = self.encoder(encoder_input, key_padding_mask=full_key_padding_mask)
        
        # Use the output of the CLS token
        cls_output = encoded_features[0] # (batch, dim_model)
        
        # Prediction
        logits = self.output_head(cls_output) # (batch, 1)
        return torch.sigmoid(logits)
```

**Training Script for Classifier:** `lerobot/scripts/train_failure_classifier.py`
This would be a standard binary classification training script:
*   Load data using `LeRobotDataset` (from `mixed_200` and `success_100`).
*   For each episode, extract all (state, action) pairs.
*   Pad/truncate sequences to `max_classifier_seq_len`.
*   Labels come from the `success` flag in `episodes.jsonl`.
*   Use `BCELoss`.
---
### 3. FCIL Policy Model

**File to Modify:** `lerobot/common/policies/act/modeling_act.py`
Create `FCILACTPolicy(ACTPolicy)` and potentially `FCILACT(ACT)`.

```python
# Snippet for lerobot/common/policies/act/modeling_act.py

# ... (imports) ...
# from .configuration_act import FCILACTConfig (You'll need a new config dataclass)

class FCILACT(ACT): # Or modify ACT directly if config can handle the new inputs
    def __init__(self, config: ACTConfig): # Should be FCILACTConfig
        super().__init__(config)
        self.config = config # Store the FCIL-specific config
        
        # Learnable failure token
        self.failure_token_embed = nn.Embedding(1, config.dim_model) # Or nn.Parameter

        # Encoder for the failed trajectory (if different from VAE encoder)
        # This could be another instance of ACTEncoder or a simpler LSTM/GRU
        if config.use_dedicated_failure_encoder: # Add this to FCILACTConfig
            self.failure_trajectory_encoder = ACTEncoder(config.failure_encoder_config) # Needs separate config
            self.failure_feature_proj = nn.Linear(config.dim_model, config.dim_model) # Project to main dim_model
        
        # How to combine failure context with other inputs?
        # Option 1: Concatenate to latent_sample (if failure_trajectory_encoder outputs fixed size)
        # Option 2: Add as additional tokens to the main transformer encoder
        # Let's go with Option 2 as it's more flexible with transformer attention
        
        # Adjust positional embeddings for encoder if adding new tokens
        # self.encoder_1d_feature_pos_embed needs to account for failure token and failure context token
        # Or, handle positional embeddings for these new tokens separately.
        
        # Original ACT has self.encoder_1d_feature_pos_embed for [latent, robot_state, env_state]
        # We will add [failure_context_embedding, failure_token_embedding]
        # Total 1D tokens: 1 (latent) + 1 (robot_state) + 1 (env_state) + 1 (fail_context) + 1 (fail_token)
        # This needs careful management of indices for positional embeddings.
        # An easier way: failure context and token get their own pos embeds or are simply added.
        # For now, let's assume they are concatenated to the `latent_sample` before projection.
        # This requires the failure context to be a fixed-size vector.
        
        # If concatenating to latent_sample:
        # Adjust self.encoder_latent_input_proj if latent_dim changes due to concatenation
        # new_latent_dim = config.latent_dim + config.dim_model (for failure context) + config.dim_model (for failure token)
        # self.encoder_latent_input_proj = nn.Linear(new_latent_dim, config.dim_model)

    def encode_failure_trajectory(self, failed_obs_seq, failed_act_seq, key_padding_mask=None):
        # Simplified: uses the VAE encoder logic (which is actually an ACTEncoder)
        # Input: failed_obs_seq (B, N, obs_dim), failed_act_seq (B, N, act_dim)
        # Output: (B, dim_model)
        if hasattr(self, 'failure_trajectory_encoder'):
            # Use dedicated encoder
            # Similar to classifier: concat, project, add pos_embed, encode, take CLS or mean pool
            # This is a placeholder for a more complete implementation.
            # For now, let's assume it reuses part of the VAE encoder logic from original ACT.
            # This part needs careful implementation matching your choice of failure encoding.
            # A simple approach:
            sa_concat = torch.cat([failed_obs_seq, failed_act_seq], dim=-1)
            sa_embed_fail = self.vae_encoder_action_input_proj(sa_concat) # Reusing projection
            
            # Create a cls token for failure encoding
            cls_embed_fail = self.vae_encoder_cls_embed.weight.expand(sa_embed_fail.shape[0], -1, -1)
            fail_encoder_input = torch.cat([cls_embed_fail, sa_embed_fail], dim=1)
            
            # Add positional embeddings
            # Reusing vae_encoder_pos_enc; ensure dimensions match context window N + 1
            pos_embed_fail = self.vae_encoder_pos_enc[:, :fail_encoder_input.shape[1], :].clone().detach()
            
            # Key padding for failure trajectory
            # Add a False for the CLS token
            if key_padding_mask is not None:
                cls_padding_mask_fail = torch.full((key_padding_mask.shape[0], 1), False, device=key_padding_mask.device)
                full_key_padding_mask_fail = torch.cat([cls_padding_mask_fail, key_padding_mask], dim=1)
            else:
                full_key_padding_mask_fail = None

            fail_context_embedding = self.vae_encoder( # reusing VAE encoder
                fail_encoder_input.permute(1,0,2),
                pos_embed=pos_embed_fail.permute(1,0,2),
                key_padding_mask=full_key_padding_mask_fail
            )[0] # Select CLS token output: (B, dim_model)
            fail_context_embedding = self.failure_feature_proj(fail_context_embedding)

        else: # Fallback or if not using dedicated encoder (e.g. if failure context is just last state)
            # This would be a very simplified failure context, e.g., just the last state of failure
            # For the full trajectory, an encoder as above is needed.
            # Placeholder: If not using a dedicated encoder, assume failed_obs_seq is already an embedding
            fail_context_embedding = torch.zeros(failed_obs_seq.shape[0], self.config.dim_model, device=failed_obs_seq.device)


        return fail_context_embedding


    def forward(self, batch: dict[str, Tensor], is_recovery_mode: bool, failed_obs_seq=None, failed_act_seq=None, failed_traj_padding_mask=None) -> tuple[Tensor, dict]:
        # ... (standard ACT batch processing and VAE part if use_vae) ...
        # batch["action"] is the target action sequence for ACT (from successful recovery trajectory)
        # batch["observation.state"] is the current observation sequence (from successful recovery trajectory)
        
        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and self.training:
            # ... (original ACT VAE logic to get mu, log_sigma_x2, latent_sample) ...
            # This uses batch["action"] which should be the *target* actions from the successful recovery.
            # (Code omitted for brevity, same as original ACT)
            pass # placeholder for VAE logic
        else:
            mu = log_sigma_x2 = None
            batch_size_eff = batch["observation.state"].shape[0]
            latent_sample = torch.zeros([batch_size_eff, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # === FCIL Modification: Incorporate failure context ===
        if is_recovery_mode:
            assert failed_obs_seq is not None and failed_act_seq is not None
            # Encode the failed trajectory
            fail_context_embedding = self.encode_failure_trajectory(failed_obs_seq, failed_act_seq, failed_traj_padding_mask) # (B, dim_model)
            
            # Get the failure token embedding
            failure_token = self.failure_token_embed.weight.squeeze(0).expand(latent_sample.shape[0], -1) # (B, dim_model)
            
            # Option 1: Concatenate to latent_sample (requires adjusting self.encoder_latent_input_proj)
            # combined_latent = torch.cat([latent_sample, fail_context_embedding, failure_token], dim=-1)
            # encoder_cond_input = self.encoder_latent_input_proj(combined_latent) # (B, dim_model)

            # Option 2 (More Transformer-like): Add as separate tokens to the transformer encoder
            # This requires modifying how encoder_in_tokens and encoder_in_pos_embed are built
            # For now, let's assume a simpler concatenation to latent_sample (Option 1 style) for this snippet
            # which means the `combined_latent` will be projected by `encoder_latent_input_proj`.
            # This is a simplification and adding them as distinct tokens to the transformer might be better.
            # For distinct tokens:
            # encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample),
            #                      fail_context_embedding, # Assuming already projected
            #                      failure_token]
            # And update positional embeddings accordingly.

            # For this snippet, let's stick to modifying the `latent_sample` that goes into projection:
            # This is a placeholder - a proper modification to `encoder_in_tokens` list in ACT's main encoder is better.
            # The simplest for now is to sum them into the latent_sample after its projection.
            projected_latent = self.encoder_latent_input_proj(latent_sample)
            projected_latent += self.failure_feature_proj(fail_context_embedding) # if not already projected
            projected_latent += failure_token 
            encoder_cond_input_token = projected_latent
        else:
            encoder_cond_input_token = self.encoder_latent_input_proj(latent_sample)
        
        # --- Resume standard ACT transformer encoder input preparation ---
        # encoder_in_tokens will start with encoder_cond_input_token
        encoder_in_tokens = [encoder_cond_input_token]
        # ... (rest of ACT's self.encoder input preparation with robot_state, env_state, image_features)
        # ... (ACT's self.encoder and self.decoder forward pass) ...
        # ... (ACT's action_head) ...

        # This is a high-level sketch. The actual integration into ACT's `forward` needs to
        # carefully manage the token sequences and positional embeddings if choosing Option 2.
        # If Option 1 (concat to latent), then `self.encoder_latent_input_proj` input dim changes.
        
        # Placeholder for actual ACT forward pass logic adapted for FCIL
        # The original ACT.forward method logic for building `encoder_in_tokens`, 
        # `encoder_in_pos_embed`, and calling encoder/decoder should be here.
        # The key change is that `encoder_in_tokens[0]` is now `encoder_cond_input_token`.
        # If using Option 2 (separate tokens), `encoder_in_tokens` would be extended,
        # and `encoder_in_pos_embed` would need corresponding positional embeddings.
        
        # For this example, assume original ACT's forward logic is called here using `encoder_cond_input_token`
        # as the first element of `encoder_in_tokens` list.
        # The rest of the ACT.forward() method would largely remain the same,
        # taking `encoder_cond_input_token` as the first token in `encoder_in_tokens`.

        # Example of what ACT.forward does (simplified, assuming encoder_cond_input_token is the "latent" part):
        _encoder_in_tokens_list = [encoder_cond_input_token]
        _encoder_in_pos_embed_list = [self.encoder_1d_feature_pos_embed.weight[0].unsqueeze(0)] # For the combined latent/cond

        if self.config.robot_state_feature:
            _encoder_in_tokens_list.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
            _encoder_in_pos_embed_list.append(self.encoder_1d_feature_pos_embed.weight[1].unsqueeze(0))
        # ... add env_state, image_features as in original ACT ...

        final_encoder_in_tokens = torch.stack(_encoder_in_tokens_list, axis=0) # Seq, Batch, Dim
        final_encoder_in_pos_embed = torch.stack(_encoder_in_pos_embed_list, axis=0) # Seq, Batch, Dim
        
        encoder_out = self.encoder(final_encoder_in_tokens, pos_embed=final_encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch["observation.state"].shape[0], self.config.dim_model),
            dtype=final_encoder_in_pos_embed.dtype,
            device=final_encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=final_encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions_hat = self.action_head(decoder_out)
        # --- End of ACT forward pass logic ---

        return actions_hat, (mu, log_sigma_x2)


class FCILACTPolicy(ACTPolicy):
    def __init__(self, config: ACTConfig, dataset_stats=None): # Should be FCILACTConfig
        super().__init__(config, dataset_stats)
        # Override self.model with FCILACT
        self.model = FCILACT(config) # This should be FCILACT
        # The rest of ACTPolicy (normalization, reset, select_action, etc.) can be inherited
        # or overridden if FCIL behavior differs significantly.
        # `select_action` will need to be overridden for inference.

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Unpack batch based on is_recovery
        obs_seq = batch['obs_seq']
        act_seq = batch['act_seq'] # This is the target for ACT
        is_recovery = batch['is_recovery'] # (B,) boolean tensor

        # Create a dummy observation structure for normalize_inputs
        # This structure depends on what normalize_inputs expects.
        # Typically, it's like {"observation.state": obs_state_seq_for_act, "action": target_act_for_act}
        # obs_state_seq_for_act would be obs_seq[:, -1] if n_obs_steps=1
        # target_act_for_act would be act_seq
        
        # Construct the input batch for the underlying ACT model's normalization and forward pass
        # Assume n_obs_steps = 1 for simplicity for now. ACT expects current obs, target actions.
        current_obs_for_act = {"observation.state": obs_seq[:, -1, :]} # (B, obs_dim)
                                                                       # Add images if used
        
        # Add other observation modalities if present (e.g., images)
        # current_obs_for_act["observation.images"] = ...
        
        normalized_obs = self.normalize_inputs(current_obs_for_act)
        
        # For target normalization, ACT expects "action" key
        normalized_targets = self.normalize_targets({"action": act_seq, "action_is_pad": batch.get("action_is_pad")})
        
        # Prepare inputs for FCILACT.forward()
        # This part is tricky because ACT's VAE path also uses batch['action'].
        # For recovery mode, batch['action'] is from tau_s. For standard, it's also from tau_s.
        # The VAE part of ACT should always use the target actions from the successful trajectory.
        
        # We need to pass all necessary components to self.model.forward
        # Standard ACT batch for VAE encoder (if used)
        act_model_input_batch = {
            "observation.state": normalized_obs["observation.state"], # Current state from success traj
            # Add images if used
            # "observation.images": normalized_obs["observation.images"],
            "action": normalized_targets["action"], # Target actions from success traj
            "action_is_pad": normalized_targets["action_is_pad"]
        }

        # Handle recovery mode specific inputs
        # This assumes that failed_traj_obs and failed_traj_act are already normalized if needed,
        # or normalization happens inside FCILACT.encode_failure_trajectory
        failed_obs_seq = batch.get('failed_traj_obs')
        failed_act_seq = batch.get('failed_traj_act')
        failed_traj_padding_mask = batch.get('failed_traj_padding_mask') # Important for variable length failure contexts

        # The `is_recovery_mode` flag needs to be handled.
        # Since the model's forward processes a whole batch, and a batch can be mixed,
        # we need a way to pass this. A simple way is to pass the boolean tensor
        # and let FCILACT handle it.
        # However, ACT's design isn't for mixed batches typically.
        # It's easier if the batch is *either* all standard or all recovery.
        # The current FCILCombinedDataset creates mixed batches.
        #
        # Simplification: Assume FCILACT.forward takes a single is_recovery_mode for the whole batch.
        # This means the dataloader should provide homogeneous batches or we process them separately.
        # Let's assume the script ensures homogeneous batches for now for simplicity.
        
        # For a mixed batch, you'd need to process standard and recovery samples separately
        # or make FCILACT handle masks for conditioning.
        # For this snippet, let's assume a single `is_recovery_mode_for_batch` derived from `is_recovery`.
        is_recovery_mode_for_batch = is_recovery[0].item() # if batch is homogeneous

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(
            act_model_input_batch, 
            is_recovery_mode=is_recovery_mode_for_batch, # Pass this flag
            failed_obs_seq=failed_obs_seq,      # Pass failure context
            failed_act_seq=failed_act_seq,
            failed_traj_padding_mask=failed_traj_padding_mask
        )

        # Loss calculation (same as ACTPolicy, using target actions from success trajectory)
        # Target actions are in normalized_targets["action"]
        l1_loss = (
            F.l1_loss(normalized_targets["action"], actions_hat, reduction="none") * ~normalized_targets["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae: # Assuming config is FCILACTConfig, check if VAE is used
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        
        return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], failed_trajectory_context=None) -> Tensor:
        self.eval()
        
        # Normalize current observation
        current_obs_for_act = {"observation.state": batch["observation.state"]} # Assuming n_obs_steps=1
        # Add images if used
        # if self.config.image_features:
        #     current_obs_for_act["observation.images"] = [batch[key] for key in self.config.image_features]

        normalized_obs = self.normalize_inputs(current_obs_for_act)

        # Prepare batch for model
        model_input_batch = {"observation.state": normalized_obs["observation.state"]}
        # if self.config.image_features:
        #     model_input_batch["observation.images"] = normalized_obs["observation.images"]

        is_recovery = failed_trajectory_context is not None
        failed_obs_seq, failed_act_seq, failed_pad_mask = None, None, None
        if is_recovery:
            failed_obs_seq = failed_trajectory_context['obs'] # (1, N, obs_dim)
            failed_act_seq = failed_trajectory_context['act'] # (1, N, act_dim)
            # Potentially pass padding mask for the failed trajectory if N is fixed but actual length varies
            # failed_pad_mask = failed_trajectory_context['pad_mask'] # (1, N)
        
        # Get action predictions
        # Note: VAE path is skipped during inference by not passing "action" to model.
        actions_chunk, _ = self.model(
            model_input_batch,
            is_recovery_mode=is_recovery,
            failed_obs_seq=failed_obs_seq,
            failed_act_seq=failed_act_seq,
            failed_traj_padding_mask=failed_pad_mask 
        ) # (B, chunk_size, action_dim), B=1 for inference

        # We only care about the first n_action_steps from the chunk
        actions = actions_chunk[:, : self.config.n_action_steps]
        
        actions = self.unnormalize_outputs({"action": actions})["action"]

        # Action queue logic (if n_action_steps > 1 for the policy config)
        # This part is similar to original ACTPolicy.select_action for queue management
        if len(self._action_queue) == 0:
             self._action_queue.extend(actions.transpose(0, 1)) # if batch_size=1
        return self._action_queue.popleft()

```

**Configuration:** `lerobot/common/policies/act/configuration_act.py`
You'll need to create `FCILACTConfig(ACTConfig)` and add parameters like `failure_encoder_config`, `use_dedicated_failure_encoder`, `max_failure_context_len (N)`.

### 4. New Training Script

**New File:** `lerobot/scripts/train_fcil.py`

This script will be similar to `lerobot/scripts/train.py` but with modifications:

*   **Dataset/Dataloader Setup:**
    *   Instantiate `FCILCombinedDataset`.
    *   Use a standard `torch.utils.data.DataLoader`. If batches can be mixed (standard and recovery), the policy's `forward` needs to handle this. If batches are homogeneous (all standard or all recovery), you might sample from two dataloaders or use a custom batch sampler.
*   **Policy Instantiation:** Instantiate `FCILACTPolicy`.
*   **Loss Calculation:**
    *   The `FCILACTPolicy.forward` already computes the correct loss based on the sample type (implicitly, as the `batch` contents will differ).
    *   The $\lambda$ hyperparameter needs to be applied if you train with separate losses and then combine them. If the policy's forward directly computes the combined loss based on `is_recovery`, then $\lambda$ is handled internally or by weighting samples. A simpler way: if using two dataloaders, fetch one batch for $\mathcal{L}_{standard}$ and one for $\mathcal{L}_{recovery}$, compute losses, then `total_loss = standard_loss + lambda_ * recovery_loss`.
---

```python

# Snippet for lerobot/scripts/train_fcil.py (conceptual)

# ... (imports, setup like train.py) ...

def train_fcil(cfg: FCILTrainPipelineConfig): # New config for this script
    # ... (wandb, device, seed setup) ...

    # 1. Datasets and Dataloaders
    # Option A: Two dataloaders
    # success_only_dataset = LeRobotDataset(cfg.dataset.success_repo_id, ...)
    # mixed_episodes_dataset = LeRobotDataset(cfg.dataset.mixed_repo_id, ...)
    # This requires a more complex sampler/dataloader setup to get paired (fail, success) items for recovery.
    # Using FCILCombinedDataset is preferred.
    
    fcil_dataset = FCILCombinedDataset(
        success_repo_id=cfg.dataset.success_repo_id,
        mixed_repo_id=cfg.dataset.mixed_repo_id,
        # ... other dataset args like delta_timestamps, image_transforms ...
        context_window_N=cfg.policy.max_failure_context_len 
    )
    
    # Dataloader might need a custom collate_fn if `failed_traj_*` can be None or have variable lengths.
    # For variable lengths, padding is needed. For None, default_collate might work if Nones are batched as Nones.
    dataloader = torch.utils.data.DataLoader(
        fcil_dataset,
        # ... (batch_size, num_workers, etc.) ...
        # collate_fn=custom_fcil_collate_fn # if needed
    )
    dl_iter = cycle(dataloader)

    # 2. Policy
    # policy_cfg should be FCILACTConfig
    policy = FCILACTPolicy(cfg.policy, dataset_stats=fcil_dataset.get_combined_stats()) # You'll need a way to get stats
    policy.to(device)

    # 3. Optimizer, Scheduler, GradScaler
    # ... (as in train.py, using policy.get_optim_params()) ...
    
    # Training loop
    for step in range(cfg.steps):
        batch = next(dl_iter)
        # Move batch to device
        # batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        # Handle None for failed_traj parts if not on device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device, non_blocking=True)
            # Note: failed_traj_obs/act might be None for standard samples.
            # The policy.forward method must handle this.
            
        # Policy update (FCILACTPolicy.forward computes the loss)
        # The loss calculation within FCILACTPolicy should handle the lambda weighting implicitly
        # by how it processes standard vs. recovery samples, OR the loss components
        # could be returned separately and combined here.
        # Assuming FCILACTPolicy.forward computes the appropriate loss based on batch content:
        
        # The lambda weighting can be done by ensuring the batch has a mix of standard and recovery
        # data according to 1:lambda ratio, or by having policy.forward return two loss values.
        # Let's assume policy.forward returns a single, possibly weighted, loss.
        
        loss, loss_info_dict = update_policy_fcil( # A new update_policy or adapt existing
            train_tracker, # or some metrics tracker
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler,
            lr_scheduler,
            use_amp=cfg.policy.use_amp,
            lambda_recovery=cfg.lambda_recovery # Pass lambda if combining losses here
        )
        # ... (logging, checkpointing, eval) ...

# update_policy_fcil would be similar to the one in train.py,
# but calls policy.forward(batch) where batch is prepared by FCILCombinedDataset
# and FCILACTPolicy.forward computes the loss based on 'is_recovery'.

# If FCILACTPolicy.forward returns separate losses:
# loss_standard, loss_recovery, output_dict = policy.forward(batch)
# total_loss = loss_standard + cfg.lambda_recovery * loss_recovery

```
---
### 5. Inference and Evaluation

**File to Modify:** `lerobot/scripts/eval.py` (or create `eval_fcil.py`)
The `rollout` function and `eval_policy` function need to be adapted.

```python

def rollout_fcil(
    env: gym.vector.VectorEnv,
    policy: FCILACTPolicy, # Expects FCILACTPolicy
    failure_classifier: TrajectoryFailureClassifier,
    # ... (other args from original rollout) ...
    max_retries: int = 1 # How many times to attempt recovery
) -> dict:
    # ... (reset env, policy) ...
    
    current_retry = 0
    failed_trajectory_for_context = None # Store (obs_seq, act_seq) of the failure

    while not np.all(done) and current_retry <= max_retries:
        # Preprocess observation for policy
        # obs_for_policy = preprocess_observation(observation) # From LeRobot
        # obs_for_policy = {k: v.to(device) for k, v in obs_for_policy.items()}

        # Prepare batch for select_action (simplified for B=1)
        current_obs_batch = {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float().to(device).unsqueeze(0)
            # Add images if used
        }

        with torch.inference_mode():
            action_tensor = policy.select_action(current_obs_batch, failed_trajectory_for_context)
        
        action_np = action_tensor.cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action_np)

        # Store s, a for current trajectory attempt
        # current_trajectory_buffer.append((observation, action_np))

        done_now = terminated | truncated
        
        if np.any(done_now): # If any env in the batch is done
            # Check for failures in those that are done
            # This is simplified. In a batched env, you'd check each env that finished.
            if not info.get("is_success", False) and current_retry < max_retries : # Assuming batched_env sets this
                # Use failure_classifier (needs full trajectory or last N steps)
                # trajectory_to_classify = format_for_classifier(current_trajectory_buffer)
                # failure_prob = failure_classifier(trajectory_to_classify_obs, trajectory_to_classify_act)
                # if failure_prob > threshold:
                if True: # Simplified: assume failure if not success and retries left
                    print(f"Failure detected. Attempting recovery ({current_retry + 1}/{max_retries}).")
                    # Store context for next attempt (last N steps of failed obs and actions)
                    # failed_trajectory_for_context = get_last_n_steps_from_buffer(current_trajectory_buffer, N_context_window)
                    # Reset this specific env within the batch if possible, or manage flags
                    # env.reset(options={'seed': specific_seed_for_retry}) # if env supports this
                    # For VectorEnv, if one env fails, it might auto-reset or need specific handling.
                    # The policy.reset() might be needed for that specific env's policy state.
                    
                    # For the *next* iteration of this *same episode*:
                    # failed_trajectory_for_context must be assembled from current_trajectory_buffer
                    # Then reset current_trajectory_buffer for the recovery attempt.
                    # The `done` flag for this specific environment in the batch should be set to False
                    # to allow continuation. This logic gets complex with VectorEnvs.
                    pass # Placeholder for actual recovery logic setup

            # If successful or max_retries reached for this episode segment
            # current_trajectory_buffer.clear()
            failed_trajectory_for_context = None # Reset for next independent episode
            # current_retry = 0 # Reset for next independent episode

        # Update observation, done, etc.
        observation = next_observation
        # ... (logging, metrics update) ...
    # ... (return rollout data) ..
```
---
## Potential Issues and Considerations

*   **Dataset Size for Paired Data:** You'll need a good number of (failure, success) pairs for the recovery mode to learn effectively.
*   **Failure Trajectory Encoding ($N$):** The choice of $N$ (context window) is crucial. Too small, and it lacks context. Too large, and it's computationally expensive and might overfit. ACT's VAE encoder (an `ACTEncoder`) could be a good starting point for encoding $\tau_{fail}$ into a fixed vector.
*   **Computational Cost:** Training will be more expensive due to processing two types of samples and potentially encoding failed trajectories.
*   **Normalization:** Ensure consistent normalization across states, actions, and failed trajectory contexts. `normalize_inputs` and `unnormalize_outputs` in `ACTPolicy` will handle current states/actions. The failed trajectory context might need separate handling or to be passed through a similar normalization scheme.
*   **`z_{fail}` Token:** Experiment with how this token is incorporated. Concatenating it to the observation embedding or treating it as a special token in the transformer sequence are options.
*   **Mixed Batches:** If your dataloader produces batches with a mix of standard and recovery samples, `FCILACTPolicy.forward` needs to be robust. It might be simpler to have the dataloader yield homogeneous batches (all standard or all recovery) by using two separate iterators and alternating.
*   **Failure Classifier Accuracy:** The performance of FCIL will depend on how well $f_\phi$ detects relevant failures.
*   **Defining "Same Task" for Pairs:** Ensure that $\tau_f$ and $\tau_s$ in a pair truly correspond to the same task instance and starting conditions (or as close as possible).
*   **ACT's VAE Objective:** The original ACT uses a VAE for smoother action spaces. You need to decide if/how this interacts with the recovery conditioning. The $\mathcal{L}_{standard}$ and $\mathcal{L}_{recovery}$ would likely both include the VAE's KL divergence term if `use_vae=True` in your ACT config, applied to the *target successful actions*.
