"""Base classes for pose estimation models.

This module defines the core data structures and abstract base classes
for pose estimation in OctagonBrain.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# COCO 17 keypoint indices and names
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Keypoint name to index mapping
COCO_KEYPOINT_NAMES = {v: k for k, v in COCO_KEYPOINTS.items()}

# Skeleton connections for visualization (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

# Flip indices for horizontal augmentation (left <-> right swap)
COCO_FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Key joint indices for combat sports analysis
COMBAT_JOINTS = {
    "guard": (9, 10),  # wrists for guard position
    "hips": (11, 12),  # hip rotation
    "stance": (15, 16),  # ankles for stance width
    "head": (0, 3, 4),  # nose and ears for head movement
    "punching_arm_left": (5, 7, 9),  # shoulder, elbow, wrist
    "punching_arm_right": (6, 8, 10),
    "kicking_leg_left": (11, 13, 15),  # hip, knee, ankle
    "kicking_leg_right": (12, 14, 16),
}


@dataclass
class PoseResult:
    """Result from pose estimation inference.

    Attributes:
        keypoints: Detected keypoints with shape (N, 17, 3) where last dim is (x, y, conf).
        boxes: Bounding boxes with shape (N, 4) in format (x1, y1, x2, y2).
        scores: Detection confidence scores with shape (N,).
        frame_id: Optional frame identifier for video processing.
        timestamp: Optional timestamp in seconds.

    Example:
        >>> result = PoseResult(
        ...     keypoints=np.zeros((2, 17, 3)),
        ...     boxes=np.zeros((2, 4)),
        ...     scores=np.array([0.95, 0.87]),
        ... )
        >>> result.num_detections
        2
    """

    keypoints: np.ndarray  # (N, 17, 3) - x, y, confidence
    boxes: np.ndarray  # (N, 4) - x1, y1, x2, y2
    scores: np.ndarray  # (N,) - detection confidence
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate array shapes after initialization."""
        if self.keypoints.ndim == 2:
            # Single person: (17, 3) -> (1, 17, 3)
            self.keypoints = self.keypoints[np.newaxis, ...]
        if self.boxes.ndim == 1:
            # Single box: (4,) -> (1, 4)
            self.boxes = self.boxes[np.newaxis, ...]
        if self.scores.ndim == 0:
            # Single score: scalar -> (1,)
            self.scores = np.array([self.scores])

    @property
    def num_detections(self) -> int:
        """Return number of detected persons."""
        return len(self.scores)

    @property
    def is_empty(self) -> bool:
        """Return True if no detections."""
        return self.num_detections == 0

    @property
    def num_keypoints(self) -> int:
        """Return number of keypoints per person (always 17 for COCO)."""
        return 17

    def get_keypoint_xy(self, person_idx: int = 0) -> np.ndarray:
        """Get (x, y) coordinates for a specific person.

        Args:
            person_idx: Index of the person (0-indexed).

        Returns:
            Array of shape (17, 2) with x, y coordinates.
        """
        if person_idx >= self.num_detections:
            raise IndexError(f"Person index {person_idx} out of range (max: {self.num_detections - 1})")
        return self.keypoints[person_idx, :, :2]

    def get_keypoint_confidence(self, person_idx: int = 0) -> np.ndarray:
        """Get confidence scores for each keypoint of a person.

        Args:
            person_idx: Index of the person (0-indexed).

        Returns:
            Array of shape (17,) with confidence scores.
        """
        if person_idx >= self.num_detections:
            raise IndexError(f"Person index {person_idx} out of range (max: {self.num_detections - 1})")
        return self.keypoints[person_idx, :, 2]

    def get_box(self, person_idx: int = 0) -> np.ndarray:
        """Get bounding box for a specific person.

        Args:
            person_idx: Index of the person (0-indexed).

        Returns:
            Array of shape (4,) with x1, y1, x2, y2.
        """
        if person_idx >= self.num_detections:
            raise IndexError(f"Person index {person_idx} out of range (max: {self.num_detections - 1})")
        return self.boxes[person_idx]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> "PoseResult":
        """Filter detections by confidence threshold.

        Args:
            min_confidence: Minimum detection confidence.

        Returns:
            New PoseResult with only high-confidence detections.
        """
        mask = self.scores >= min_confidence
        return PoseResult(
            keypoints=self.keypoints[mask],
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def filter_by_score(self, min_score: float = 0.5) -> "PoseResult":
        """Alias for filter_by_confidence for backward compatibility."""
        return self.filter_by_confidence(min_score)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "keypoints": self.keypoints.tolist(),
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimation models.

    All pose estimators must implement detect() and detect_batch() methods.

    Example:
        >>> class MyPoseEstimator(BasePoseEstimator):
        ...     def detect(self, frame):
        ...         # Implementation here
        ...         pass
        ...     def detect_batch(self, frames):
        ...         return [self.detect(f) for f in frames]
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> PoseResult:
        """Detect poses in a single frame.

        Args:
            frame: Input image as numpy array (H, W, 3) in BGR format.

        Returns:
            PoseResult containing detected poses.
        """
        pass

    @abstractmethod
    def detect_batch(self, frames: list[np.ndarray]) -> list[PoseResult]:
        """Detect poses in multiple frames.

        Args:
            frames: List of input images.

        Returns:
            List of PoseResult, one per frame.
        """
        pass

    @staticmethod
    def normalize_keypoints(
        keypoints: np.ndarray,
        reference_indices: tuple[int, int] = (11, 12),  # hips
        scale_indices: tuple[tuple[int, int], tuple[int, int]] = ((5, 6), (11, 12)),
    ) -> np.ndarray:
        """Normalize keypoints to be position and scale invariant.

        Centers on hip midpoint and scales by torso length (shoulder to hip distance).

        Args:
            keypoints: Keypoints array of shape (..., 17, 2) or (..., 17, 3).
            reference_indices: Tuple of keypoint indices to compute center (default: hips).
            scale_indices: Two pairs of indices for scale: shoulders and hips.

        Returns:
            Normalized keypoints with same shape as input.

        Example:
            >>> kpts = np.random.rand(17, 2) * 100
            >>> normalized = BasePoseEstimator.normalize_keypoints(kpts)
            >>> # Now centered on hip midpoint and scaled by torso
        """
        # Get original shape
        original_shape = keypoints.shape
        has_confidence = original_shape[-1] == 3

        # Work with (x, y) only
        if has_confidence:
            xy = keypoints[..., :2].copy()
            conf = keypoints[..., 2:3]
        else:
            xy = keypoints.copy()

        # Compute center (hip midpoint)
        left_ref = xy[..., reference_indices[0], :]
        right_ref = xy[..., reference_indices[1], :]
        center = (left_ref + right_ref) / 2

        # Center the keypoints
        centered = xy - center[..., np.newaxis, :]

        # Compute scale (torso length)
        shoulder_l = xy[..., scale_indices[0][0], :]
        shoulder_r = xy[..., scale_indices[0][1], :]
        hip_l = xy[..., scale_indices[1][0], :]
        hip_r = xy[..., scale_indices[1][1], :]

        shoulder_center = (shoulder_l + shoulder_r) / 2
        hip_center = (hip_l + hip_r) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)

        # Avoid division by zero
        torso_length = np.maximum(torso_length, 1e-6)

        # Scale
        normalized = centered / torso_length[..., np.newaxis, :]

        # Recombine with confidence if present
        if has_confidence:
            normalized = np.concatenate([normalized, conf], axis=-1)

        return normalized.astype(np.float32)

    @staticmethod
    def flip_keypoints(keypoints: np.ndarray, image_width: Optional[int] = None) -> np.ndarray:
        """Flip keypoints horizontally.

        Args:
            keypoints: Keypoints array of shape (..., 17, 2) or (..., 17, 3).
            image_width: Image width for flipping x coordinates. If None, assumes normalized.

        Returns:
            Flipped keypoints with left/right swapped.
        """
        flipped = keypoints.copy()

        # Swap left/right indices
        flipped = flipped[..., COCO_FLIP_INDICES, :]

        # Flip x coordinates
        if image_width is not None:
            flipped[..., 0] = image_width - flipped[..., 0]
        else:
            flipped[..., 0] = -flipped[..., 0]

        return flipped
