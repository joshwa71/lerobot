# File: /home/josh/phddev/lerobot/common/datasets/trajectory_dataset.py

import logging
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import datasets # type: ignore
from torch.utils.data import Dataset
import torchvision # To check for transforms
import numpy as np # Added for loading .npy files

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_episode_data_index, hf_transform_to_torch
from lerobot.common.datasets.video_utils import decode_video_frames, get_safe_default_codec
from lerobot.configs.policies import PreTrainedConfig # For delta_timestamps resolution helper


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        repo_ids: List[str],
        max_seq_len: int,
        image_feature_keys: List[str],
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True, # This might be less relevant if use_embeddings is True
        video_backend: str | None = None,
        mixed_ds_identifier: str = "mixed",
        use_embeddings: bool = False, # New flag
        embedding_dim: int | None = None, # New flag, needed if use_embeddings is True
    ):
        self.max_seq_len = max_seq_len
        self.mixed_ds_identifier = mixed_ds_identifier
        self.image_feature_keys = image_feature_keys
        self.image_transforms = image_transforms
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.tolerance_s = tolerance_s
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim

        if self.use_embeddings and not self.embedding_dim:
            raise ValueError("embedding_dim must be provided when use_embeddings is True.")
        if self.use_embeddings and self.image_transforms:
            logging.warning("image_transforms are provided but use_embeddings is True. Transforms will be ignored for embeddings.")
            self.image_transforms = None # Do not apply transforms to embeddings

        self.dataset_metadatas: List[LeRobotDatasetMetadata] = []
        self.hf_datasets_list: List[datasets.Dataset | None] = []
        self.episode_data_indices_list: List[dict | None] = []
        self.episode_map = []

        self.final_feature_shapes: dict[str, tuple[int, ...]] = {} # Stores final C,H,W or embedding_dim

        for ds_list_idx, repo_id in enumerate(repo_ids):
            ds_root = Path(root) / repo_id if root else None
            meta = LeRobotDatasetMetadata(repo_id, root=ds_root, revision=revision, force_cache_sync=force_cache_sync)
            self.dataset_metadatas.append(meta)

            if meta.total_episodes == 0:
                 logging.warning(f"Dataset {repo_id} has 0 episodes in its metadata. Skipping.")
                 self.hf_datasets_list.append(None)
                 self.episode_data_indices_list.append(None)
                 continue

            # Determine final feature shapes (either image or embedding)
            if self.image_feature_keys:
                for key in self.image_feature_keys:
                    if key not in self.final_feature_shapes: # Only determine once per key globally
                        if self.use_embeddings:
                            self.final_feature_shapes[key] = (self.embedding_dim,)
                        elif key in meta.features and meta.features[key]['dtype'] in ['image', 'video']:
                            h_orig, w_orig, c_orig = meta.features[key]['shape']
                            c_final, h_final, w_final = c_orig, h_orig, w_orig

                            if self.image_transforms is not None:
                                try:
                                    dummy_frame = torch.zeros(1, c_orig, h_orig, w_orig)
                                    transformed_dummy = self.image_transforms(dummy_frame)
                                    c_final, h_final, w_final = transformed_dummy.shape[1], transformed_dummy.shape[2], transformed_dummy.shape[3]
                                except Exception as e_transform:
                                    logging.warning(f"Could not infer shape after transforms for {key} from {repo_id}: {e_transform}. Checking for explicit Resize.")
                                    has_resize = False
                                    if isinstance(self.image_transforms, (torch.nn.Sequential, torchvision.transforms.v2.Compose)):
                                        for t_comp in self.image_transforms.transforms:
                                            if isinstance(t_comp, torchvision.transforms.v2.Resize):
                                                size = t_comp.size
                                                if isinstance(size, int): h_final, w_final = size, size
                                                elif isinstance(size, (list,tuple)) and len(size)==2: h_final, w_final = size[0], size[1]
                                                has_resize = True; break
                                    if not has_resize:
                                         logging.warning(f"No Resize found, using original CHW {c_orig, h_orig, w_orig} for {key}.")
                            self.final_feature_shapes[key] = (c_final, h_final, w_final)
                        else: # Fallback if key not in meta or not image/video
                            if self.use_embeddings:
                                logging.warning(f"Embedding key {key} not processed. Using configured embedding_dim: {self.embedding_dim}.")
                                self.final_feature_shapes[key] = (self.embedding_dim,)
                            else:
                                logging.warning(f"Image key {key} from config not in metadata features of {repo_id} or not image/video. Using default (3,224,224).")
                                self.final_feature_shapes[key] = (3, 224, 224)


            parquet_files_for_this_ds = []
            files_to_download = [] # Can be videos or embeddings based on use_embeddings
            if self.use_embeddings:
                # Logic to identify embedding files to download (if not local)
                for ep_idx_in_meta in range(meta.total_episodes):
                     try:
                        # Parquet paths are always needed
                        parquet_path = meta.root / meta.get_data_file_path(ep_idx_in_meta)
                        parquet_files_for_this_ds.append(str(parquet_path) if ds_root else str(meta.get_data_file_path(ep_idx_in_meta)))
                        
                        # Embedding paths
                        for key_emb in self.image_feature_keys:
                            # Construct base path for episode embeddings for this camera
                            # e.g. my_dataset_root/embeddings/chunk-000/observation.images.head/episode_000000/
                            chunk_num = ep_idx_in_meta // meta.chunks_size
                            ep_embedding_dir_name = f"episode_{ep_idx_in_meta:06d}"
                            embedding_ep_dir_path_str = f"embeddings/chunk-{chunk_num:03d}/{key_emb}/{ep_embedding_dir_name}"
                            
                            if ds_root: # Local dataset
                                embedding_ep_dir_abs = meta.root / embedding_ep_dir_path_str
                                if not embedding_ep_dir_abs.exists(): # Check if the whole episode embedding dir needs download
                                     files_to_download.append(f"{embedding_ep_dir_path_str}/*") # Glob pattern for all frames
                            else:
                                files_to_download.append(f"{embedding_ep_dir_path_str}/*")
                     except Exception as e:
                        logging.warning(f"Error getting paths for episode {ep_idx_in_meta} in {repo_id} (embeddings mode): {e}. Skipping.")
            else: # Original video/image logic
                for ep_idx_in_meta in range(meta.total_episodes):
                    try:
                        parquet_path = meta.root / meta.get_data_file_path(ep_idx_in_meta)
                        parquet_files_for_this_ds.append(str(parquet_path) if ds_root else str(meta.get_data_file_path(ep_idx_in_meta)))

                        if download_videos and self.image_feature_keys: # download_videos flag remains relevant for non-embedding mode
                            for key_img in self.image_feature_keys:
                                 if key_img in meta.video_keys:
                                    video_path_obj = meta.root / meta.get_video_file_path(ep_idx_in_meta, key_img)
                                    if not video_path_obj.exists():
                                         files_to_download.append(str(meta.get_video_file_path(ep_idx_in_meta, key_img)))
                    except Exception as e:
                        logging.warning(f"Error getting paths for episode {ep_idx_in_meta} in {repo_id} (video mode): {e}. Skipping.")
            
            if files_to_download: # For both video and embedding files
                download_type = "embedding" if self.use_embeddings else "video"
                logging.info(f"Downloading {len(files_to_download)} {download_type} file patterns for {repo_id}...")
                meta.pull_from_repo(allow_patterns=list(set(files_to_download)), ignore_patterns=None)

            if not parquet_files_for_this_ds:
                logging.warning(f"No parquet files found or downloadable for {repo_id}. Skipping.")
                self.hf_datasets_list.append(None); self.episode_data_indices_list.append(None); continue

            try:
                data_files_arg = parquet_files_for_this_ds
                if not ds_root:
                    data_files_arg = { "data_files": parquet_files_for_this_ds }

                hf_ds = datasets.load_dataset("parquet", data_files=data_files_arg, cache_dir=ds_root.parent / ".cache" if ds_root else None, split="train")
                hf_ds.set_transform(hf_transform_to_torch)
                self.hf_datasets_list.append(hf_ds)
                current_episode_data_index = get_episode_data_index(meta.episodes, list(range(meta.total_episodes)))
                self.episode_data_indices_list.append(current_episode_data_index)

                for ep_idx_in_meta in range(meta.total_episodes):
                    if ep_idx_in_meta < len(current_episode_data_index["from"]):
                        self.episode_map.append((ds_list_idx, ep_idx_in_meta))
                    else:
                        logging.warning(f"Ep {ep_idx_in_meta} of {repo_id} not in loaded parquet. Skipping.")
            except Exception as e:
                logging.error(f"Failed to load/process parquet for {repo_id}: {e}")
                self.hf_datasets_list.append(None); self.episode_data_indices_list.append(None)

        if not self.episode_map:
            raise ValueError("TrajectoryDataset init resulted in no episodes. Check paths/contents.")

    def __len__(self):
        return len(self.episode_map)

    def _load_features_for_episode(self, meta, hf_ds_for_repo, ep_idx_in_original_dataset, ep_global_start, ep_global_end):
        """Loads either images/videos or pre-computed embeddings for an episode."""
        episode_timestamps = torch.stack(hf_ds_for_repo[ep_global_start:ep_global_end]['timestamp']).tolist()
        all_cam_features_for_ep = {}

        for cam_key in self.image_feature_keys:
            if self.use_embeddings:
                # Load .npy embeddings
                # Path: root / embeddings / chunk-XXX / obs.img.cam_key / episode_YYYYYY / frame_ZZZ.npy
                chunk_num = ep_idx_in_original_dataset // meta.chunks_size
                ep_embedding_dir_name = f"episode_{ep_idx_in_original_dataset:06d}"
                
                base_embedding_path = meta.root / f"embeddings/chunk-{chunk_num:03d}/{cam_key}/{ep_embedding_dir_name}"
                
                frame_embeddings = []
                # Frames in an episode are indexed 0 to N-1 relative to the episode start
                num_frames_in_episode = ep_global_end - ep_global_start
                for frame_idx_in_ep in range(num_frames_in_episode):
                    embedding_file = base_embedding_path / f"frame_{frame_idx_in_ep}.npy"
                    try:
                        emb = np.load(embedding_file)
                        frame_embeddings.append(torch.from_numpy(emb.astype(np.float32))) # Ensure float32 for consistency
                    except FileNotFoundError:
                        logging.warning(f"Embedding file not found: {embedding_file}. Using zeros.")
                        frame_embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                
                if not frame_embeddings: # Should not happen if episode has frames
                     c_final = self.final_feature_shapes.get(cam_key, (self.embedding_dim,))[0]
                     features = torch.zeros(len(episode_timestamps), c_final, dtype=torch.float32)
                else:
                    features = torch.stack(frame_embeddings)

            else: # Load images/videos
                if cam_key in meta.video_keys:
                    video_path = meta.root / meta.get_video_file_path(ep_idx_in_original_dataset, cam_key)
                    features = decode_video_frames(video_path, episode_timestamps, self.tolerance_s, self.video_backend)
                elif cam_key in meta.image_keys:
                    features = torch.stack(hf_ds_for_repo[ep_global_start:ep_global_end][cam_key])
                else:
                    c,h,w = self.final_feature_shapes.get(cam_key, (3,224,224)) # Use determined/default shape
                    features = torch.zeros(len(episode_timestamps), c, h, w, dtype=torch.float32)
                    logging.warning(f"Image key {cam_key} not found in {meta.repo_id} for ep {ep_idx_in_original_dataset}. Using zeros.")

                if self.image_transforms: # Apply transforms only if not using embeddings
                    features = torch.stack([self.image_transforms(f) for f in features])

            all_cam_features_for_ep[cam_key] = features
        return all_cam_features_for_ep


    def __getitem__(self, flat_idx):
        ds_list_idx, ep_idx_in_original_dataset = self.episode_map[flat_idx]

        meta = self.dataset_metadatas[ds_list_idx]
        hf_ds_for_repo = self.hf_datasets_list[ds_list_idx]
        episode_data_index_for_repo = self.episode_data_indices_list[ds_list_idx]

        # Default dummy values
        state_dim = meta.features.get("observation.state", {}).get("shape", [6])[0]
        action_dim = meta.features.get("action", {}).get("shape", [6])[0]

        obs_state_dummy_full = torch.zeros(self.max_seq_len, state_dim, dtype=torch.float32)
        act_dummy_full = torch.zeros(self.max_seq_len, action_dim, dtype=torch.float32)

        obs_feature_dummy_dict_full = {}
        for key in self.image_feature_keys:
            feature_shape = self.final_feature_shapes.get(key) # Get pre-determined shape
            if not feature_shape: # Fallback
                 feature_shape = (self.embedding_dim,) if self.use_embeddings else (3,224,224)
                 logging.warning(f"Could not determine final shape for {key}, using fallback {feature_shape}")
            obs_feature_dummy_dict_full[key] = torch.zeros(self.max_seq_len, *feature_shape, dtype=torch.float32)

        padding_mask_dummy = torch.ones(self.max_seq_len, dtype=torch.bool)
        label_dummy = torch.tensor([0.0], dtype=torch.float32)

        item_dict = {
            'observation.state': obs_state_dummy_full,
            'action': act_dummy_full,
            'label': label_dummy,
            'padding_mask': padding_mask_dummy,
            **obs_feature_dummy_dict_full
        }

        if hf_ds_for_repo is None or episode_data_index_for_repo is None:
            logging.error(f"Dataset {meta.repo_id} not loaded. Dummy data for flat_idx {flat_idx}.")
            return item_dict

        try:
            ep_global_start = episode_data_index_for_repo["from"][ep_idx_in_original_dataset].item()
            ep_global_end = episode_data_index_for_repo["to"][ep_idx_in_original_dataset].item()
        except IndexError:
            logging.error(f"IndexError for {meta.repo_id}, ep_idx {ep_idx_in_original_dataset}. Dummy data.")
            return item_dict

        episode_frames_data = hf_ds_for_repo[ep_global_start:ep_global_end]

        obs_state_seq_tensor = torch.stack(episode_frames_data['observation.state'])
        act_seq_tensor = torch.stack(episode_frames_data['action'])

        obs_feature_seq_dict_processed = {}
        if self.image_feature_keys:
            obs_feature_seq_dict_processed = self._load_features_for_episode(
                meta, hf_ds_for_repo, ep_idx_in_original_dataset, ep_global_start, ep_global_end
            )

        current_len = obs_state_seq_tensor.shape[0]
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        if current_len == 0:
            logging.warning(f"Empty ep: {meta.repo_id}, ep_idx={ep_idx_in_original_dataset}")
        elif current_len >= self.max_seq_len:
            item_dict['observation.state'] = obs_state_seq_tensor[:self.max_seq_len]
            item_dict['action'] = act_seq_tensor[:self.max_seq_len]
            for key_cam in self.image_feature_keys:
                 if key_cam in obs_feature_seq_dict_processed:
                    item_dict[key_cam] = obs_feature_seq_dict_processed[key_cam][:self.max_seq_len]
                 # else: it remains the full dummy tensor from item_dict init
            padding_mask[:] = False
        else: # Pad
            pad_len = self.max_seq_len - current_len
            item_dict['observation.state'] = torch.cat([obs_state_seq_tensor, obs_state_dummy_full[:pad_len]], dim=0)
            item_dict['action'] = torch.cat([act_seq_tensor, act_dummy_full[:pad_len]], dim=0)

            for key_cam in self.image_feature_keys:
                if key_cam in obs_feature_seq_dict_processed:
                    actual_cam_data = obs_feature_seq_dict_processed[key_cam]
                    # Use the shape from self.final_feature_shapes for consistency for this cam_key
                    final_shape_this_cam = self.final_feature_shapes.get(key_cam, actual_cam_data.shape[1:]) # fallback to actual data's shape if not found
                    dummy_cam_pad = torch.zeros(pad_len, *final_shape_this_cam, dtype=actual_cam_data.dtype)
                    item_dict[key_cam] = torch.cat([actual_cam_data, dummy_cam_pad], dim=0)
                # else, if key_cam was not processed, it remains the full dummy tensor

            padding_mask[:current_len] = False

        item_dict['padding_mask'] = padding_mask

        is_failure = False
        episode_meta_info = meta.episodes.get(ep_idx_in_original_dataset)
        success_flag = episode_meta_info.get('success') if episode_meta_info else None

        if success_flag is False: is_failure = True
        elif success_flag is True: is_failure = False
        else:
            if self.mixed_ds_identifier and self.mixed_ds_identifier in meta.repo_id:
                is_failure = (ep_idx_in_original_dataset % 2 == 0)
            else:
                 is_failure = False
                 if success_flag is None and episode_meta_info is not None :
                     logging.debug(f"Ep {ep_idx_in_original_dataset} in {meta.repo_id} missing 'success' flag, assuming success.")

        item_dict['label'] = torch.tensor([float(is_failure)], dtype=torch.float32)
        return item_dict