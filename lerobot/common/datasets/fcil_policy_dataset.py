import logging
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any

import torch
import datasets # type: ignore
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data._utils.collate import default_collate 


from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_episode_data_index
from lerobot.common.policies.fcil_policy.configuration_fcil_policy import FCILPolicyConfig 

logger = logging.getLogger(__name__)

def fcil_policy_collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor]], config: FCILPolicyConfig) -> Tuple[Dict[str, Any], torch.Tensor]: # Added config argument
    """
    Custom collate function for FCILPolicyDataset.
    Handles 'history_context' and 'fail_traj_context_optional' which can contain lists of (dict or None).
    Ensures that the model always receives a list of batched dictionaries, even if padding is needed.
    """
    policy_input_dicts, target_actions_and_dones = zip(*batch)

    collated_policy_input_dict: Dict[str, Any] = {}
    
    first_input_dict = policy_input_dicts[0]
    batch_size = len(policy_input_dicts)

    for key in first_input_dict:
        if key not in ["history_context", "fail_traj_context_optional"]: # Removed "config" from here
            collated_policy_input_dict[key] = default_collate([d[key] for d in policy_input_dicts])

    # --- Custom collation for history_context ---
    history_contexts = [d["history_context"] for d in policy_input_dicts] 
    history_len = 0
    first_valid_history_sample = next((hc for hc in history_contexts if hc is not None and len(hc) > 0), None)
    if first_valid_history_sample is not None:
        history_len = len(first_valid_history_sample) # Should be config.history_len
    
    collated_history_context_list: List[Dict[str, Any]] = [] 

    if history_len > 0:
        template_item_hist = None
        for hc_sample in history_contexts:
            if hc_sample:
                for step_data in hc_sample:
                    if step_data:
                        template_item_hist = step_data
                        break
            if template_item_hist:
                break
        
        if template_item_hist is None:
            logger.debug("No valid history item found for template, using fallback for padding shapes based on config.")
            # Use config for shapes if no live template
            template_item_hist_fallback = {
                'state': torch.zeros((config.state_dim,), dtype=torch.float32),
                'action': torch.zeros((config.action_dim,), dtype=torch.float32),
                'observation_visual': {
                    k: torch.zeros((config.embedding_dim_in,), dtype=torch.float32) 
                    for k in config.image_feature_keys
                }
            }
            template_item_hist = template_item_hist_fallback


        for i in range(history_len):
            history_step_batch_items = [] 
            for sample_idx in range(batch_size):
                hc_sample = history_contexts[sample_idx]
                if hc_sample is not None and i < len(hc_sample) and hc_sample[i] is not None:
                    history_step_batch_items.append(hc_sample[i])
                else:
                    padding_dict = {
                        'state': torch.zeros_like(template_item_hist['state']), 
                        'action': torch.zeros_like(template_item_hist['action']),
                        'observation_visual': {
                            k: torch.zeros_like(v) for k, v in template_item_hist['observation_visual'].items()
                        }
                    }
                    history_step_batch_items.append(padding_dict)
            collated_history_context_list.append(default_collate(history_step_batch_items))
    collated_policy_input_dict["history_context"] = collated_history_context_list


    # --- Custom collation for fail_traj_context_optional ---
    fail_traj_contexts = [d.get("fail_traj_context_optional") for d in policy_input_dicts] 
    
    collated_fail_traj_context_list: List[Dict[str, Any]] | None = None # Initialize as None

    if any(ftc is not None for ftc in fail_traj_contexts):
        collated_fail_traj_context_list = [] # Will become a list of dicts
        first_valid_fail_traj = next((ftc for ftc in fail_traj_contexts if ftc is not None and len(ftc) > 0), None)
        
        fail_traj_len_for_batch = config.max_fail_traj_len # Use config for consistent length
        
        template_item_fail_step = None
        if first_valid_fail_traj is not None and len(first_valid_fail_traj) > 0:
             template_item_fail_step = first_valid_fail_traj[0] # Template for a single step
        else: # Fallback template if no valid fail traj in batch
            logger.debug("No valid fail_traj item found for template, using fallback for padding shapes based on config.")
            template_item_fail_step = {
                'state': torch.zeros((config.state_dim,), dtype=torch.float32),
                'action': torch.zeros((config.action_dim,), dtype=torch.float32),
                'observation_visual': {
                    k: torch.zeros((config.embedding_dim_in,), dtype=torch.float32) 
                    for k in config.image_feature_keys
                }
            }


        for i in range(fail_traj_len_for_batch):
            fail_step_batch_items = []
            for sample_idx in range(batch_size):
                ftc_sample = fail_traj_contexts[sample_idx]
                # Check if the sample is in recovery and has this specific step
                if ftc_sample is not None and i < len(ftc_sample) and ftc_sample[i] is not None : 
                    fail_step_batch_items.append(ftc_sample[i])
                else: 
                    padding_dict_fail = {
                        'state': torch.zeros_like(template_item_fail_step['state']),
                        'action': torch.zeros_like(template_item_fail_step['action']),
                        'observation_visual': {
                            k: torch.zeros_like(v) for k,v in template_item_fail_step['observation_visual'].items()
                        }
                    }
                    fail_step_batch_items.append(padding_dict_fail)
            collated_fail_traj_context_list.append(default_collate(fail_step_batch_items))
    collated_policy_input_dict["fail_traj_context_optional"] = collated_fail_traj_context_list


    collated_target_actions_and_dones = default_collate(target_actions_and_dones)

    return collated_policy_input_dict, collated_target_actions_and_dones


class FCILPolicyDataset(Dataset):
    def __init__(
        self,
        config: FCILPolicyConfig,
        success_dataset_repo_id: str,
        mixed_dataset_repo_id: str,
        root: str | Path | None = None,
        image_transforms: Callable | None = None, 
        revision: str | None = None,
    ):
        self.config = config
        self.root = Path(root) if root else None
        self.image_transforms = image_transforms 
        
        self.success_ds_meta = LeRobotDatasetMetadata(
            success_dataset_repo_id, 
            root=self.root / success_dataset_repo_id if self.root else None, 
            revision=revision
        )
        self.mixed_ds_meta = LeRobotDatasetMetadata(
            mixed_dataset_repo_id, 
            root=self.root / mixed_dataset_repo_id if self.root else None, 
            revision=revision
        )

        self.success_hf_ds = self._load_hf_dataset_for_meta(self.success_ds_meta)
        self.mixed_hf_ds = self._load_hf_dataset_for_meta(self.mixed_ds_meta)
        
        self.success_ep_data_idx = get_episode_data_index(self.success_ds_meta.episodes)
        self.mixed_ep_data_idx = get_episode_data_index(self.mixed_ds_meta.episodes)

        self.image_tokens_per_cam = self.config.embedding_dim_in // self.config.model_dim
        
        self.episode_map: List[Dict[str, Any]] = []
        self._build_episode_map()

        if not self.episode_map:
            raise ValueError("FCILPolicyDataset init resulted in no valid samples. Check dataset contents and pairing.")
        
    def _load_hf_dataset_for_meta(self, meta: LeRobotDatasetMetadata) -> datasets.Dataset:
        parquet_files_in_meta = [meta.root / meta.get_data_file_path(ep_idx) for ep_idx in range(meta.total_episodes)]
        parquet_files_exist = [str(p) for p in parquet_files_in_meta if p.exists()]

        if not parquet_files_exist:
            logger.info(f"Parquet files for {meta.repo_id} not found locally at expected paths. Attempting to pull data files.")
            meta.pull_from_repo(allow_patterns="data/*/*.parquet", ignore_patterns=None)
            parquet_files_exist = [str(p) for p in parquet_files_in_meta if p.exists()]
            if not parquet_files_exist:
                raise FileNotFoundError(f"No parquet files found for {meta.repo_id} at {meta.root} even after attempting pull. Ensure data is downloaded/present.")
        
        files_to_download = []
        if self.config.use_embeddings: 
            for ep_idx_in_meta in range(meta.total_episodes):
                for key_emb in self.config.image_feature_keys:
                    chunk_num = ep_idx_in_meta // meta.chunks_size
                    ep_embedding_dir_name = f"episode_{ep_idx_in_meta:06d}"
                    embedding_ep_dir_path_str = f"embeddings/chunk-{chunk_num:03d}/{key_emb}/{ep_embedding_dir_name}"
                    
                    current_ep_dir_path = meta.root / embedding_ep_dir_path_str
                    if not current_ep_dir_path.exists():
                        files_to_download.append(f"{embedding_ep_dir_path_str}/*") 
        
        if files_to_download:
            logger.info(f"Downloading {len(files_to_download)} embedding file patterns for {meta.repo_id}...")
            meta.pull_from_repo(allow_patterns=list(set(files_to_download)), ignore_patterns=None)

        cache_dir_path_str = None
        if meta.root: # Only set cache_dir if root is local, otherwise datasets library handles HF cache
            cache_dir_path = Path(meta.root).parent / ".cache" / meta.repo_id.split("/")[-1]
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_dir_path_str = str(cache_dir_path)

        hf_ds = datasets.load_dataset(
            "parquet", 
            data_files=parquet_files_exist, 
            cache_dir=cache_dir_path_str, 
            split="train"
        )
        return hf_ds

    def _build_episode_map(self):
        for ep_idx in range(self.success_ds_meta.total_episodes):
            ep_len = self.success_ds_meta.episodes[ep_idx]['length']
            if ep_len == 0: continue
            for t in range(ep_len): 
                self.episode_map.append({
                    "ds_idx": 0, 
                    "ep_idx_in_ds": ep_idx, 
                    "timestep_in_ep": t,
                    "mode": "standard"
                })

        for ep_idx in range(self.mixed_ds_meta.total_episodes):
            ep_meta = self.mixed_ds_meta.episodes[ep_idx]
            ep_len = ep_meta['length']
            if ep_len == 0: continue
            is_success_ep = ep_meta.get('success', True) 

            if is_success_ep:
                for t in range(ep_len):
                    self.episode_map.append({
                        "ds_idx": 1, 
                        "ep_idx_in_ds": ep_idx,
                        "timestep_in_ep": t,
                        "mode": "standard"
                    })
            else: 
                if ep_idx + 1 < self.mixed_ds_meta.total_episodes:
                    next_ep_meta = self.mixed_ds_meta.episodes[ep_idx + 1]
                    if next_ep_meta.get('success', False): 
                        succ_ep_len = next_ep_meta['length']
                        if succ_ep_len == 0: continue
                        for t_succ in range(succ_ep_len):
                            self.episode_map.append({
                                "ds_idx": 1, 
                                "fail_ep_idx_in_ds": ep_idx,
                                "succ_ep_idx_in_ds": ep_idx + 1,
                                "timestep_in_succ_ep": t_succ, 
                                "mode": "recovery"
                            })
                    else:
                        logger.debug(f"Failure ep {ep_idx} in {self.mixed_ds_meta.repo_id} not followed by a success ep. Skipping for recovery.")
                else:
                    logger.debug(f"Failure ep {ep_idx} in {self.mixed_ds_meta.repo_id} is last ep. Skipping for recovery.")
        
        logger.info(f"Built episode map with {len(self.episode_map)} samples.")


    def _load_trajectory_timestep_data(self, ds_meta: LeRobotDatasetMetadata, hf_ds: datasets.Dataset, 
                                      ep_data_idx: Dict, ep_idx_in_ds: int, t_in_ep: int) -> Dict[str, Any]:
        ep_global_start = ep_data_idx["from"][ep_idx_in_ds].item()
        
        frame_data = hf_ds[ep_global_start + t_in_ep]
        
        data_out = {}
        data_out['state'] = torch.tensor(frame_data['observation.state'], dtype=torch.float32)
        data_out['action'] = torch.tensor(frame_data['action'], dtype=torch.float32)
        
        is_done = (t_in_ep == ds_meta.episodes[ep_idx_in_ds]['length'] - 1)
        done_signal = torch.tensor([1.0 if is_done else 0.0], dtype=torch.float32)
        data_out['action_done_target'] = torch.cat([data_out['action'], done_signal])

        visual_features = {}
        for cam_key in self.config.image_feature_keys:
            chunk_num = ep_idx_in_ds // ds_meta.chunks_size
            ep_embedding_dir_name = f"episode_{ep_idx_in_ds:06d}"
            base_embedding_path = ds_meta.root / f"embeddings/chunk-{chunk_num:03d}/{cam_key}/{ep_embedding_dir_name}"
            embedding_file = base_embedding_path / f"frame_{t_in_ep}.npy"
            try:
                emb = np.load(embedding_file)
                visual_features[cam_key] = torch.from_numpy(emb.astype(np.float32))
            except FileNotFoundError:
                logger.warning(f"Embedding file not found: {embedding_file}. Using zeros.")
                visual_features[cam_key] = torch.zeros(self.config.embedding_dim_in, dtype=torch.float32)
        data_out['observation_visual'] = visual_features 
        
        return data_out

    def _get_history_and_current(self, ds_meta, hf_ds, ep_data_idx, ep_idx_in_ds, t_in_ep) -> Tuple[List[Dict[str, Any] | None], Dict[str, Any], torch.Tensor]:
        history_data_list: List[Dict[str, Any] | None] = []
        for k_hist_offset in range(self.config.history_len, 0, -1): 
            hist_t = t_in_ep - k_hist_offset
            if hist_t >= 0:
                step_data = self._load_trajectory_timestep_data(ds_meta, hf_ds, ep_data_idx, ep_idx_in_ds, hist_t)
                history_data_list.append({
                    "state": step_data['state'],
                    "observation_visual": step_data['observation_visual'],
                    "action": step_data['action'] 
                })
            else: 
                history_data_list.append(None) 

        current_step_data = self._load_trajectory_timestep_data(ds_meta, hf_ds, ep_data_idx, ep_idx_in_ds, t_in_ep)
        current_s_o = {
            "state": current_step_data['state'],
            "observation_visual": current_step_data['observation_visual']
        }
        target_action_done = current_step_data['action_done_target']
        
        return history_data_list, current_s_o, target_action_done

    def _get_fail_trajectory_context(self, ds_meta, hf_ds, ep_data_idx, fail_ep_idx_in_ds) -> List[Dict[str, Any]]:
        fail_traj_data_list: List[Dict[str, Any]] = []
        fail_ep_len = ds_meta.episodes[fail_ep_idx_in_ds]['length']
        
        start_t_fail = max(0, fail_ep_len - self.config.max_fail_traj_len)
        
        for t_fail in range(start_t_fail, fail_ep_len):
            step_data = self._load_trajectory_timestep_data(ds_meta, hf_ds, ep_data_idx, fail_ep_idx_in_ds, t_fail)
            fail_traj_data_list.append({
                "state": step_data['state'],
                "observation_visual": step_data['observation_visual'],
                "action": step_data['action'] 
            })
        
        num_actual_fail_steps = len(fail_traj_data_list)
        num_pad_fail_steps = self.config.max_fail_traj_len - num_actual_fail_steps
        if num_pad_fail_steps > 0:
            if num_actual_fail_steps > 0:
                template_step = fail_traj_data_list[0] # Use first actual step as template
            else: 
                template_step = {
                    'state': torch.zeros(self.config.state_dim, dtype=torch.float32),
                    'action': torch.zeros(self.config.action_dim, dtype=torch.float32),
                    'observation_visual': {
                        k: torch.zeros(self.config.embedding_dim_in, dtype=torch.float32)
                        for k in self.config.image_feature_keys
                    }
                }
            
            for _ in range(num_pad_fail_steps):
                padding_step_data = {
                    'state': torch.zeros_like(template_step['state']),
                    'action': torch.zeros_like(template_step['action']),
                    'observation_visual': {
                        k: torch.zeros_like(v) for k,v in template_step['observation_visual'].items()
                    }
                }
                fail_traj_data_list.insert(0, padding_step_data) # Prepend padding

        return fail_traj_data_list


    def __len__(self):
        return len(self.episode_map)

    def __getitem__(self, idx):
        sample_info = self.episode_map[idx]
        mode = sample_info["mode"]

        target_action_and_done: torch.Tensor
        history_context: List[Dict[str, Any] | None] 
        current_state_obs: Dict[str, Any] 
        fail_traj_context_optional: List[Dict[str, Any]] | None = None 

        if mode == "standard":
            ds_idx = sample_info["ds_idx"]
            ep_idx_in_ds = sample_info["ep_idx_in_ds"]
            t_in_ep = sample_info["timestep_in_ep"]

            ds_meta = self.success_ds_meta if ds_idx == 0 else self.mixed_ds_meta
            hf_ds = self.success_hf_ds if ds_idx == 0 else self.mixed_hf_ds
            ep_data_idx = self.success_ep_data_idx if ds_idx == 0 else self.mixed_ep_data_idx

            history_context, current_state_obs, target_action_and_done = \
                self._get_history_and_current(ds_meta, hf_ds, ep_data_idx, ep_idx_in_ds, t_in_ep)
            is_recovery_mode = False
            # For standard mode, create a fully padded fail_traj_context for consistent structure
            fail_traj_context_optional = []
            template_step_fail_pad = {
                    'state': torch.zeros(self.config.state_dim, dtype=torch.float32),
                    'action': torch.zeros(self.config.action_dim, dtype=torch.float32),
                    'observation_visual': {
                        k: torch.zeros(self.config.embedding_dim_in, dtype=torch.float32)
                        for k in self.config.image_feature_keys
                    }
                }
            for _ in range(self.config.max_fail_traj_len):
                fail_traj_context_optional.append(template_step_fail_pad)


        elif mode == "recovery":
            fail_ep_idx = sample_info["fail_ep_idx_in_ds"]
            succ_ep_idx = sample_info["succ_ep_idx_in_ds"]
            t_in_succ_ep = sample_info["timestep_in_succ_ep"]

            ds_meta = self.mixed_ds_meta
            hf_ds = self.mixed_hf_ds
            ep_data_idx = self.mixed_ep_data_idx 

            fail_traj_context_optional = self._get_fail_trajectory_context(ds_meta, hf_ds, ep_data_idx, fail_ep_idx)
            
            history_context, current_state_obs, target_action_and_done = \
                self._get_history_and_current(ds_meta, hf_ds, ep_data_idx, succ_ep_idx, t_in_succ_ep)
            is_recovery_mode = True
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

        policy_input_dict = {
            "history_context": history_context, 
            "current_state_obs": current_state_obs, 
            "fail_traj_context_optional": fail_traj_context_optional, # Now always a list of dicts
            "is_recovery_mode": is_recovery_mode,
        }
        
        return policy_input_dict, target_action_and_done