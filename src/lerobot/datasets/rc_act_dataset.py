#!/usr/bin/env python

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class RCACTDataset(LeRobotDataset):
    """Dataset wrapper that augments samples with a per-episode success flag.

    Expects the dataset metadata to include a boolean "success" field in each
    entry of `episodes.jsonl` under `meta/`.
    """

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)

        episode_index = item["episode_index"].item()
        episode_info = self.meta.episodes.get(episode_index)
        if episode_info is None or "success" not in episode_info:
            raise ValueError(
                f"Episode {episode_index} in dataset {self.repo_id} does not have a 'success' flag. "
                "Please ensure `meta/episodes.jsonl` contains a 'success' boolean per episode."
            )

        success = float(episode_info["success"])  # cast to {0.0,1.0}
        item["success"] = torch.tensor([success], dtype=torch.float32)
        return item


