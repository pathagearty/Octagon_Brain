"""Skeleton sequence utilities for action recognition.

This module provides utilities for processing skeleton sequences including
normalization, temporal interpolation, windowing, and augmentation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..models.pose.base import COCO_FLIP_INDICES


@dataclass
class SkeletonSequence:
    """Container for skeleton sequence data.

    Attributes:
        keypoints: Skeleton keypoints (T, 17, 2) or (T, 17, 3).
        label: Optional action label string.
        label_id: Optional numeric label ID.
        subject_id: Optional subject identifier.
        video_id: Optional video identifier.
        start_frame: Start frame in original video.
        end_frame: End frame in original video.

    Example:
        >>> keypoints = np.random.rand(25, 17, 2)
        >>> seq = SkeletonSequence(keypoints, label="jab", label_id=0)
        >>> print(f"Frames: {seq.num_frames}, Keypoints: {seq.num_keypoints}")
    """

    keypoints: np.ndarray
    label: Optional[str] = None
    label_id: Optional[int] = None
    subject_id: Optional[int] = None
    video_id: Optional[str] = None
    start_frame: int = 0
    end_frame: Optional[int] = None

    def __post_init__(self) -> None:
        """Set end_frame if not provided."""
        if self.end_frame is None:
            self.end_frame = self.start_frame + len(self.keypoints)

    @property
    def num_frames(self) -> int:
        """Return number of frames in sequence."""
        return self.keypoints.shape[0]

    @property
    def num_keypoints(self) -> int:
        """Return number of keypoints (should be 17 for COCO)."""
        return self.keypoints.shape[1]

    @property
    def num_channels(self) -> int:
        """Return number of channels (2 for x,y or 3 for x,y,conf)."""
        return self.keypoints.shape[2]

    @property
    def has_confidence(self) -> bool:
        """Return True if keypoints include confidence scores."""
        return self.keypoints.shape[-1] == 3

    def get_xy(self) -> np.ndarray:
        """Get only x,y coordinates."""
        return self.keypoints[..., :2]

    def get_confidence(self) -> Optional[np.ndarray]:
        """Get confidence scores if available."""
        if self.has_confidence:
            return self.keypoints[..., 2]
        return None


def normalize_skeleton(
    keypoints: np.ndarray,
    center_joint: int = 11,  # left hip for single reference
    scale_joints: Tuple[int, int, int, int] = (5, 6, 11, 12),  # shoulders and hips
) -> np.ndarray:
    """Normalize skeleton to be position and scale invariant.

    Centers on hip midpoint and scales by torso size.

    Args:
        keypoints: Array of shape (T, 17, 2) or (T, 17, 3).
        center_joint: Joint index to use as center reference (unused, uses hip midpoint).
        scale_joints: Four joints (shoulder_l, shoulder_r, hip_l, hip_r) for scale.

    Returns:
        Normalized keypoints with same shape.

    Example:
        >>> skeleton = np.random.rand(25, 17, 2) * 100
        >>> normalized = normalize_skeleton(skeleton)
        >>> print(f"Max value before: {skeleton.max():.1f}, after: {normalized.max():.1f}")
    """
    has_conf = keypoints.shape[-1] == 3
    if has_conf:
        xy = keypoints[..., :2].copy()
        conf = keypoints[..., 2:3]
    else:
        xy = keypoints.copy()

    # Center on hip midpoint
    hip_l = xy[:, scale_joints[2], :]
    hip_r = xy[:, scale_joints[3], :]
    center = (hip_l + hip_r) / 2

    centered = xy - center[:, np.newaxis, :]

    # Scale by torso length
    shoulder_l = xy[:, scale_joints[0], :]
    shoulder_r = xy[:, scale_joints[1], :]
    shoulder_center = (shoulder_l + shoulder_r) / 2
    hip_center = center

    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)
    torso_length = np.maximum(torso_length, 1e-6)

    normalized = centered / torso_length[:, np.newaxis, :]

    if has_conf:
        normalized = np.concatenate([normalized, conf], axis=-1)

    return normalized.astype(np.float32)


def interpolate_temporal(
    keypoints: np.ndarray,
    target_length: int,
    mode: str = "linear",
) -> np.ndarray:
    """Interpolate skeleton sequence to target temporal length.

    Args:
        keypoints: Array of shape (T, 17, C) where C is 2 or 3.
        target_length: Desired number of frames.
        mode: Interpolation mode ('linear' or 'nearest').

    Returns:
        Interpolated keypoints of shape (target_length, 17, C).

    Example:
        >>> skeleton = np.random.rand(25, 17, 2)
        >>> interpolated = interpolate_temporal(skeleton, 64)
        >>> print(f"Shape: {skeleton.shape} -> {interpolated.shape}")
    """
    T, V, C = keypoints.shape

    if T == target_length:
        return keypoints.astype(np.float32)

    # Convert to tensor for interpolation: (1, C*V, T)
    tensor = torch.from_numpy(keypoints).float()
    tensor = tensor.permute(2, 1, 0).reshape(1, C * V, T)

    # Interpolate
    if mode == "linear":
        interpolated = F.interpolate(
            tensor, size=target_length, mode="linear", align_corners=True
        )
    else:
        interpolated = F.interpolate(tensor, size=target_length, mode="nearest")

    # Reshape back: (target_length, V, C)
    result = interpolated.reshape(C, V, target_length).permute(2, 1, 0)

    return result.numpy().astype(np.float32)


def create_skeleton_windows(
    keypoints: np.ndarray,
    window_size: int,
    stride: int,
    pad_mode: str = "replicate",
) -> list[np.ndarray]:
    """Create sliding windows from skeleton sequence.

    Args:
        keypoints: Full sequence of shape (T, 17, C).
        window_size: Size of each window.
        stride: Step between windows.
        pad_mode: How to handle sequences shorter than window_size.
                  'replicate' pads with last frame, 'zero' pads with zeros.

    Returns:
        List of windows, each of shape (window_size, 17, C).

    Example:
        >>> skeleton = np.random.rand(100, 17, 2)
        >>> windows = create_skeleton_windows(skeleton, window_size=25, stride=10)
        >>> print(f"Created {len(windows)} windows")
    """
    T = keypoints.shape[0]
    windows = []

    # Handle short sequences
    if T < window_size:
        if pad_mode == "replicate":
            padding = np.tile(keypoints[-1:], (window_size - T, 1, 1))
            padded = np.concatenate([keypoints, padding], axis=0)
        else:
            padded = np.zeros((window_size, *keypoints.shape[1:]), dtype=keypoints.dtype)
            padded[:T] = keypoints
        windows.append(padded.astype(np.float32))
        return windows

    # Sliding windows
    for start in range(0, T - window_size + 1, stride):
        windows.append(keypoints[start : start + window_size].astype(np.float32))

    return windows


def augment_skeleton(
    keypoints: np.ndarray,
    rotation_range: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    flip_prob: float = 0.5,
    temporal_crop_range: Tuple[float, float] = (0.8, 1.0),
    joint_noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply random augmentations to skeleton sequence.

    Args:
        keypoints: Array of shape (T, 17, 2) - x, y coordinates only.
        rotation_range: Maximum rotation angle in degrees.
        scale_range: Range for random scaling.
        flip_prob: Probability of horizontal flip.
        temporal_crop_range: Range for random temporal cropping ratio.
        joint_noise_std: Standard deviation for joint position noise.
        seed: Optional random seed for reproducibility.

    Returns:
        Augmented keypoints with same shape as input.

    Example:
        >>> skeleton = np.random.rand(25, 17, 2)
        >>> augmented = augment_skeleton(skeleton, flip_prob=1.0)
        >>> # Keypoints are now horizontally flipped
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T, V, C = keypoints.shape
    augmented = keypoints.copy()

    # 1. Random rotation (around center)
    if rotation_range > 0 and random.random() < 0.5:
        angle = random.uniform(-rotation_range, rotation_range) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        center = augmented.mean(axis=(0, 1))  # (C,)
        centered = augmented - center

        rotated = np.zeros_like(centered)
        rotated[..., 0] = centered[..., 0] * cos_a - centered[..., 1] * sin_a
        rotated[..., 1] = centered[..., 0] * sin_a + centered[..., 1] * cos_a

        augmented = rotated + center

    # 2. Random scale
    if scale_range != (1.0, 1.0) and random.random() < 0.5:
        scale = random.uniform(*scale_range)
        center = augmented.mean(axis=(0, 1))
        augmented = (augmented - center) * scale + center

    # 3. Horizontal flip
    if random.random() < flip_prob:
        # Flip x-coordinates
        augmented[..., 0] = -augmented[..., 0]
        # Swap left/right joints
        augmented = augmented[:, COCO_FLIP_INDICES, :]

    # 4. Random temporal crop and resize back
    if temporal_crop_range != (1.0, 1.0) and random.random() < 0.5:
        crop_ratio = random.uniform(*temporal_crop_range)
        crop_len = max(1, int(T * crop_ratio))
        start = random.randint(0, max(0, T - crop_len))
        cropped = augmented[start : start + crop_len]
        augmented = interpolate_temporal(cropped, T)

    # 5. Joint position noise
    if joint_noise_std > 0 and random.random() < 0.5:
        noise = np.random.normal(0, joint_noise_std, augmented.shape)
        augmented = augmented + noise

    return augmented.astype(np.float32)


def skeleton_to_tensor(
    keypoints: np.ndarray,
    target_frames: int = 64,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert skeleton to tensor format for SkateFormer.

    Args:
        keypoints: Array of shape (T, 17, 2) or (T, 17, 3).
        target_frames: Number of frames to interpolate to.
        normalize: Whether to apply normalization.

    Returns:
        Tensor of shape (C, T, V, M) = (2, target_frames, 17, 1).

    Example:
        >>> skeleton = np.random.rand(25, 17, 2)
        >>> tensor = skeleton_to_tensor(skeleton, target_frames=64)
        >>> print(f"Tensor shape: {tensor.shape}")  # (2, 64, 17, 1)
    """
    # Use only x, y coordinates
    if keypoints.shape[-1] == 3:
        keypoints = keypoints[..., :2]

    # Normalize if requested
    if normalize:
        keypoints = normalize_skeleton(keypoints)

    # Interpolate to target frames
    T = keypoints.shape[0]
    if T != target_frames:
        keypoints = interpolate_temporal(keypoints, target_frames)

    # Convert to tensor: (C, T, V, M)
    tensor = torch.from_numpy(keypoints).float()
    tensor = tensor.permute(2, 0, 1)  # (C, T, V)
    tensor = tensor.unsqueeze(-1)  # (C, T, V, 1)

    return tensor


def compute_bone_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute bone vectors from joint positions.

    Bone features are vectors between connected joints, useful for
    additional modalities in action recognition.

    Args:
        keypoints: Array of shape (T, 17, 2) or (T, 17, 3).

    Returns:
        Bone vectors of shape (T, 16, 2) or (T, 16, 3).
        (16 bones connecting 17 joints)

    Example:
        >>> skeleton = np.random.rand(25, 17, 2)
        >>> bones = compute_bone_features(skeleton)
        >>> print(f"Bone features shape: {bones.shape}")  # (25, 16, 2)
    """
    # Define bone connections (parent -> child)
    bone_pairs = [
        (0, 1), (0, 2),  # nose -> eyes
        (1, 3), (2, 4),  # eyes -> ears
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10),  # right arm
        (5, 6),  # shoulders
        (5, 11), (6, 12),  # shoulders -> hips
        (11, 12),  # hips
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
    ]

    T = keypoints.shape[0]
    C = keypoints.shape[-1]
    bones = np.zeros((T, len(bone_pairs), C), dtype=keypoints.dtype)

    for i, (parent, child) in enumerate(bone_pairs):
        bones[:, i, :] = keypoints[:, child, :] - keypoints[:, parent, :]

    return bones.astype(np.float32)


def compute_motion_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute temporal motion (velocity) features.

    Motion features are temporal differences between consecutive frames.

    Args:
        keypoints: Array of shape (T, 17, 2) or (T, 17, 3).

    Returns:
        Motion vectors of shape (T-1, 17, 2) or (T-1, 17, 3).

    Example:
        >>> skeleton = np.random.rand(25, 17, 2)
        >>> motion = compute_motion_features(skeleton)
        >>> print(f"Motion features shape: {motion.shape}")  # (24, 17, 2)
    """
    motion = np.diff(keypoints, axis=0)
    return motion.astype(np.float32)
