"""Pose estimation models for OctagonBrain.

This module provides pose estimation wrappers for various models including
YOLO11-Pose (primary) and BlazePose (mobile).
"""
from .base import (
    COCO_FLIP_INDICES,
    COCO_KEYPOINT_NAMES,
    COCO_KEYPOINTS,
    COCO_SKELETON,
    COMBAT_JOINTS,
    BasePoseEstimator,
    PoseResult,
)
from .yolo_pose import YOLO11Pose

__all__ = [
    # Models
    "YOLO11Pose",
    # Data structures
    "PoseResult",
    # Base class
    "BasePoseEstimator",
    # Constants
    "COCO_KEYPOINTS",
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON",
    "COCO_FLIP_INDICES",
    "COMBAT_JOINTS",
]
