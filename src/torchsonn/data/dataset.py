from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset

# Anything that supports `arr.shape[0]` and integer indexing — numpy arrays
# and torch tensors are the two real use cases.
ArrayLike = np.ndarray | torch.Tensor


class SONNDataset(Dataset):
    def __init__(
        self,
        x: ArrayLike,
        target: ArrayLike | None,
        split: int | None = None,
    ) -> None:
        self.x = x
        self.target = target
        self.split = split
        assert split in [0, 1, None]

    def __len__(self) -> int:
        length = self.x.shape[0]
        if self.split == 0:
            length = (length + 1) // 2
        elif self.split == 1:
            length = length // 2
        return length

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        if self.split == 0:
            idx = 2 * idx
        elif self.split == 1:
            idx = 2 * idx + 1
        x = self.x[idx]
        y = self.target[idx] if self.target is not None else None
        return x, y
