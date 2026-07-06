from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class NuScenesTrajectoryDataset(Dataset):
    """
    Dataset for CTG-style trajectory generation.

    trajectories.npy:
        (S, T, A, D)

    agent_masks.npy:
        (S, T, A)

    where

        S = number of scenes
        T = timesteps
        A = max agents
        D = state dimension

    Default state:

        [x, y, vx, vy]
    """

    def __init__(
        self,
        data_root: str = "data/processed",
        past_steps: int = 20,
        future_steps: int = 21,
        normalize: bool = True,
    ):

        self.data_root = Path(data_root)

        self.trajectories = np.load(
            self.data_root / "trajectories.npy"
        ).astype(np.float32)

        self.agent_masks = np.load(
            self.data_root / "agent_masks.npy"
        ).astype(bool)

        assert self.trajectories.shape[:3] == self.agent_masks.shape

        self.past_steps = past_steps
        self.future_steps = future_steps

        total_steps = self.trajectories.shape[1]

        if total_steps < past_steps + future_steps:
            raise ValueError(
                f"Dataset contains only {total_steps} timesteps."
            )

        self.total_steps = past_steps + future_steps

        self.normalize = normalize

        if normalize:
            self._compute_statistics()

    def _compute_statistics(self):
        """
        Compute dataset statistics using only valid agents.
        """

        valid = self.agent_masks[..., None]

        valid_values = self.trajectories[valid.repeat(4, axis=-1)]

        valid_values = valid_values.reshape(-1, 4)

        self.mean = valid_values.mean(axis=0)

        self.std = valid_values.std(axis=0)

        self.std[self.std < 1e-6] = 1.0

    def __len__(self):

        return self.trajectories.shape[0]

    def _normalize(self, x):

        return (x - self.mean) / self.std

    def _denormalize(self, x):

        return x * self.std + self.mean

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        scene = self.trajectories[idx]

        mask = self.agent_masks[idx]

        scene = scene[: self.total_steps]

        mask = mask[: self.total_steps]

        if self.normalize:
            scene = self._normalize(scene)

        past = scene[: self.past_steps]

        future = scene[self.past_steps :]

        past_mask = mask[: self.past_steps]

        future_mask = mask[self.past_steps :]

        sample = {

            "past": torch.from_numpy(past).float(),

            "future": torch.from_numpy(future).float(),

            "past_mask": torch.from_numpy(past_mask),

            "future_mask": torch.from_numpy(future_mask),

            "scene_index": torch.tensor(idx),

        }

        return sample

    def get_statistics(self):

        if not self.normalize:
            return None

        return {

            "mean": torch.tensor(self.mean),

            "std": torch.tensor(self.std),

        }