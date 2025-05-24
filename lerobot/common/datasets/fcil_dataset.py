import logging
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_hf_features_from_features
from lerobot.configs.policies import PreTrainedConfig # For delta_timestamps resolution helper


class FCILCombinedDataset(Dataset):
    def __init__(
        self,
        success_repo_id: str,
        mixed_repo_id: str,
        policy_config: PreTrainedConfig, # Used to resolve delta_timestamps
        context_window_N: int,
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
    ):
        self.context_window_N = context_window_N

        self.success_ds_meta = LeRobotDatasetMetadata(
            success_repo_id, root=Path(root) / success_repo_id if root else None, revision=revision, force_cache_sync=force_cache_sync
        )
        success_delta_timestamps = self._resolve_delta_timestamps(policy_config, self.success_ds_meta)
        self.success_ds = LeRobotDataset(
            repo_id=success_repo_id,
            root=Path(root) / success_repo_id if root else None,
            episodes=None,
            image_transforms=image_transforms,
            delta_timestamps=success_delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        self.mixed_ds_meta = LeRobotDatasetMetadata(
            mixed_repo_id, root=Path(root) / mixed_repo_id if root else None, revision=revision, force_cache_sync=force_cache_sync
        )
        mixed_delta_timestamps = self._resolve_delta_timestamps(policy_config, self.mixed_ds_meta)
        self.mixed_ds = LeRobotDataset(
            repo_id=mixed_repo_id,
            root=Path(root) / mixed_repo_id if root else None,
            episodes=None, # Use all episodes from mixed_ds
            image_transforms=image_transforms,
            delta_timestamps=mixed_delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        self.mixed_episodes_info = self.mixed_ds.meta.episodes
        self.num_pairs = self.mixed_ds.meta.total_episodes // 2

        # Build internal index mapping
        # Indices 0 to len(success_ds)-1 map to standard successes
        # Indices len(success_ds) onwards map to recovery trajectories from mixed_ds
        self._index_map = []
        for i in range(len(self.success_ds)):
            self._index_map.append({"type": "standard", "original_idx": i})

        for i in range(self.num_pairs):
            success_ep_idx_in_mixed = 2 * i + 1
            failure_ep_idx_in_mixed = 2 * i
            # map to each frame in the successful recovery trajectory
            success_ep_global_start = self.mixed_ds.episode_data_index["from"][success_ep_idx_in_mixed].item()
            success_ep_global_end = self.mixed_ds.episode_data_index["to"][success_ep_idx_in_mixed].item()
            for global_frame_idx in range(success_ep_global_start, success_ep_global_end):
                 self._index_map.append({
                    "type": "recovery", 
                    "original_idx": global_frame_idx, # global frame index in mixed_ds.hf_dataset for the success part
                    "failure_ep_idx": failure_ep_idx_in_mixed 
                })

    def _resolve_delta_timestamps(
        self, cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
    ) -> dict[str, list] | None:
        """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig."""
        delta_timestamps = {}
        for key in ds_meta.features:
            if key == "next.reward" and cfg.reward_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
            if key == "action" and cfg.action_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
            if key.startswith("observation.") and cfg.observation_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

        if len(delta_timestamps) == 0:
            delta_timestamps = None
        return delta_timestamps

    def __len__(self):
        return len(self._index_map)

    def _get_dummy_obs_state(self, ds_instance: LeRobotDataset) -> torch.Tensor:
        obs_state_shape = ds_instance.meta.features.get("observation.state", {}).get("shape")
        if obs_state_shape is None:
             # Fallback if "observation.state" is not in features, though it's expected for ACT
            logging.warning("'observation.state' not found in dataset features. Padding with zeros of shape (1,).")
            obs_state_shape = (1,)
        return torch.zeros(obs_state_shape, dtype=torch.float32)

    def _get_dummy_action(self, ds_instance: LeRobotDataset) -> torch.Tensor:
        action_shape = ds_instance.meta.features.get("action", {}).get("shape")
        if action_shape is None:
            # Fallback if "action" is not in features
            logging.warning("'action' not found in dataset features. Padding with zeros of shape (1,).")
            action_shape = (1,)
        return torch.zeros(action_shape, dtype=torch.float32)


    def get_full_episode_data(self, ds_instance: LeRobotDataset, ep_idx: int, last_n_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_frames_start = ds_instance.episode_data_index["from"][ep_idx].item()
        ep_frames_end = ds_instance.episode_data_index["to"][ep_idx].item()
        
        actual_start_global_idx = max(ep_frames_start, ep_frames_end - last_n_steps)
        
        obs_list = []
        act_list = []
        
        num_actual_frames = ep_frames_end - actual_start_global_idx
        
        for i in range(actual_start_global_idx, ep_frames_end):
            frame_data = ds_instance.hf_dataset[i]
            # Assuming 'observation.state' and 'action' are primary features for failure context
            # and are directly available as numpy arrays or lists in hf_dataset items.
            # LeRobotDataset.__getitem__ converts these to tensors.
            # Here, we access raw data and then convert.
            if "observation.state" in frame_data and frame_data["observation.state"] is not None:
                obs_list.append(torch.tensor(frame_data['observation.state'], dtype=torch.float32))
            else:
                # This case should ideally not happen if 'observation.state' is always present.
                obs_list.append(self._get_dummy_obs_state(ds_instance))

            if "action" in frame_data and frame_data["action"] is not None:
                act_list.append(torch.tensor(frame_data['action'], dtype=torch.float32))
            else:
                act_list.append(self._get_dummy_action(ds_instance))

        padding_mask = torch.ones(last_n_steps, dtype=torch.bool) # True for padded elements

        num_padding = last_n_steps - num_actual_frames
        if num_padding > 0:
            dummy_obs = self._get_dummy_obs_state(ds_instance)
            dummy_act = self._get_dummy_action(ds_instance)
            
            obs_padding = [dummy_obs] * num_padding
            act_padding = [dummy_act] * num_padding
            
            obs_list = obs_padding + obs_list
            act_list = act_padding + act_list
            padding_mask[:num_padding] = True # Padded elements are at the beginning
            padding_mask[num_padding:] = False # Actual data
        else:
            padding_mask[:] = False # No padding

        return torch.stack(obs_list), torch.stack(act_list), padding_mask


    def __getitem__(self, idx):
        map_info = self._index_map[idx]

        if map_info["type"] == "standard":
            item = self.success_ds[map_info["original_idx"]]
            # Ensure obs_seq and act_seq are present
            obs_seq = item.get('observation.state', self._get_dummy_obs_state(self.success_ds))
            act_seq = item.get('action', self._get_dummy_action(self.success_ds))

            return {
                'obs_seq': obs_seq,
                'act_seq': act_seq,
                'is_recovery': False,
                'failed_traj_obs': None,
                'failed_traj_act': None,
                'failed_traj_padding_mask': None,
            }
        
        elif map_info["type"] == "recovery":
            # original_idx is the global frame index for the success part in mixed_ds
            global_idx_s = map_info["original_idx"]
            failure_ep_idx = map_info["failure_ep_idx"]

            item_s = self.mixed_ds[global_idx_s]
            obs_seq_s = item_s.get('observation.state', self._get_dummy_obs_state(self.mixed_ds))
            act_seq_s = item_s.get('action', self._get_dummy_action(self.mixed_ds))

            failed_obs_seq, failed_act_seq, failed_padding_mask = self.get_full_episode_data(
                self.mixed_ds, failure_ep_idx, self.context_window_N
            )
            
            return {
                'obs_seq': obs_seq_s,
                'act_seq': act_seq_s,
                'is_recovery': True,
                'failed_traj_obs': failed_obs_seq,
                'failed_traj_act': failed_act_seq,
                'failed_traj_padding_mask': failed_padding_mask,
            }
        else:
            raise ValueError(f"Unknown sample type: {map_info['type']}")

    def get_combined_stats(self):
        # Aggregate stats from both datasets
        # Ensure that stats are loaded for both datasets
        if self.success_ds.meta.stats is None:
            logging.warning("Stats for success_ds are not loaded. Combined stats might be incomplete.")
            success_stats = {}
        else:
            success_stats = self.success_ds.meta.stats

        if self.mixed_ds.meta.stats is None:
            logging.warning("Stats for mixed_ds are not loaded. Combined stats might be incomplete.")
            mixed_stats = {}
        else:
            mixed_stats = self.mixed_ds.meta.stats
        
        # Handle cases where one or both stats are empty
        if not success_stats and not mixed_stats:
            return {}
        if not success_stats:
            return mixed_stats
        if not mixed_stats:
            return success_stats
            
        return aggregate_stats([success_stats, mixed_stats])

    @property
    def features(self) -> dict:
        # Return features from the success_ds as a representative.
        # Assumes features are compatible for the policy.
        return self.success_ds_meta.features

    @property
    def hf_features(self) -> dict:
         # Return hf_features from the success_ds as a representative.
        return get_hf_features_from_features(self.features)

    @property
    def fps(self) -> int:
        # Assuming fps is the same for both datasets
        return self.success_ds.fps 