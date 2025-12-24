"""Base dataset classes for skeleton-based action recognition.

This module provides abstract base classes for skeleton datasets used
in training action recognition models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..skeleton import (
    augment_skeleton,
    interpolate_temporal,
    normalize_skeleton,
)


@dataclass
class SkeletonSample:
    """A single skeleton sample for action recognition.

    Attributes:
        skeleton: Skeleton tensor (C, T, V, M) format for model input.
        label: Numeric class label.
        label_name: Human-readable label string.
        video_id: Source video identifier.
        clip_id: Clip index within video.
    """

    skeleton: torch.Tensor
    label: int
    label_name: str
    video_id: Optional[str] = None
    clip_id: Optional[int] = None


class BaseSkeletonDataset(Dataset, ABC):
    """Abstract base class for skeleton action recognition datasets.

    Subclasses must implement:
        - _load_annotations(): Load dataset annotations
        - _load_skeleton(): Load skeleton data for a specific sample
        - __len__(): Return dataset size
        - __getitem__(): Return a sample

    Attributes:
        root_dir: Dataset root directory.
        split: Dataset split ('train' or 'val').
        target_frames: Number of frames to interpolate to.
        normalize: Whether to apply skeleton normalization.
        augment: Whether to apply augmentation (training only).
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        target_frames: int = 64,
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        """Initialize the base dataset.

        Args:
            root_dir: Path to dataset root directory.
            split: Dataset split, one of 'train' or 'val'.
            target_frames: Number of frames to interpolate to.
            normalize: Whether to apply skeleton normalization.
            augment: Whether to apply data augmentation.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_frames = target_frames
        self.normalize = normalize
        self.augment = augment and (split == "train")

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self._load_annotations()

    @abstractmethod
    def _load_annotations(self) -> None:
        """Load dataset annotations. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _load_skeleton(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """Load skeleton data for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (skeleton array, label id, label name).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return list of class names."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of classes."""
        pass

    def _preprocess_skeleton(self, skeleton: np.ndarray) -> torch.Tensor:
        """Apply preprocessing to skeleton sequence.

        Args:
            skeleton: Raw skeleton array (T, V, C) where C is 2 or 3.

        Returns:
            Preprocessed tensor (C, T, V, M) ready for model input.
        """
        # Ensure we only use x, y coordinates
        if skeleton.shape[-1] == 3:
            skeleton = skeleton[..., :2]

        # Normalize skeleton (center on hips, scale by torso)
        if self.normalize:
            skeleton = normalize_skeleton(skeleton)

        # Apply augmentation (training only)
        if self.augment:
            skeleton = augment_skeleton(skeleton)

        # Interpolate to target frames
        T = skeleton.shape[0]
        if T != self.target_frames:
            skeleton = interpolate_temporal(skeleton, self.target_frames)

        # Convert to tensor: (T, V, C) -> (C, T, V, M)
        tensor = torch.from_numpy(skeleton).float()
        tensor = tensor.permute(2, 0, 1)  # (C, T, V)
        tensor = tensor.unsqueeze(-1)  # (C, T, V, 1)

        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (skeleton tensor, label).
        """
        skeleton, label, _ = self._load_skeleton(idx)
        tensor = self._preprocess_skeleton(skeleton)
        return tensor, label
