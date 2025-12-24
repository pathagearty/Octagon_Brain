"""Data utilities for OctagonBrain.

This module provides skeleton processing utilities and dataset loaders
for training action recognition models.
"""
from .datasets import (
    BOXINGVI_CLASSES,
    BaseSkeletonDataset,
    BoxingVIDataset,
    SkeletonSample,
)
from .skeleton import (
    SkeletonSequence,
    augment_skeleton,
    compute_bone_features,
    compute_motion_features,
    create_skeleton_windows,
    interpolate_temporal,
    normalize_skeleton,
    skeleton_to_tensor,
)

__all__ = [
    # Data structures
    "SkeletonSequence",
    "SkeletonSample",
    # Processing functions
    "normalize_skeleton",
    "interpolate_temporal",
    "create_skeleton_windows",
    "augment_skeleton",
    "skeleton_to_tensor",
    # Feature extraction
    "compute_bone_features",
    "compute_motion_features",
    # Datasets
    "BaseSkeletonDataset",
    "BoxingVIDataset",
    "BOXINGVI_CLASSES",
]
