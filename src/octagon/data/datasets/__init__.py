"""Dataset loaders for skeleton-based action recognition.

This module provides PyTorch Dataset implementations for various
skeleton action recognition datasets.
"""
from .base import BaseSkeletonDataset, SkeletonSample
from .boxingvi import (
    BOXINGVI_CLASSES,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    BoxingVIDataset,
)

__all__ = [
    # Base classes
    "BaseSkeletonDataset",
    "SkeletonSample",
    # BoxingVI dataset
    "BoxingVIDataset",
    "BOXINGVI_CLASSES",
    "TRAIN_SUBJECTS",
    "VAL_SUBJECTS",
]
