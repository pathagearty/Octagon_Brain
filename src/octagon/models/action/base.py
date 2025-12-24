"""Base classes for action recognition models.

This module defines the core data structures and abstract base classes
for action recognition in OctagonBrain.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class BoxingAction(IntEnum):
    """BoxingVI dataset action classes.

    These are the 6 punch types from the BoxingVI dataset.
    """

    JAB = 0
    CROSS = 1
    LEAD_HOOK = 2
    REAR_HOOK = 3
    LEAD_UPPERCUT = 4
    REAR_UPPERCUT = 5

    @classmethod
    def from_string(cls, name: str) -> "BoxingAction":
        """Convert string name to enum.

        Args:
            name: Action name (case-insensitive).

        Returns:
            Corresponding BoxingAction enum.

        Raises:
            KeyError: If name is not a valid action.

        Example:
            >>> BoxingAction.from_string("jab")
            <BoxingAction.JAB: 0>
        """
        name_map = {
            "jab": cls.JAB,
            "cross": cls.CROSS,
            "lead_hook": cls.LEAD_HOOK,
            "lead hook": cls.LEAD_HOOK,
            "rear_hook": cls.REAR_HOOK,
            "rear hook": cls.REAR_HOOK,
            "lead_uppercut": cls.LEAD_UPPERCUT,
            "lead uppercut": cls.LEAD_UPPERCUT,
            "rear_uppercut": cls.REAR_UPPERCUT,
            "rear uppercut": cls.REAR_UPPERCUT,
        }
        return name_map[name.lower()]

    def __str__(self) -> str:
        """Return human-readable action name."""
        return self.name.lower().replace("_", " ").title()

    @property
    def display_name(self) -> str:
        """Return display-friendly name."""
        return str(self)


# All action class names for BoxingVI
BOXING_CLASS_NAMES = [
    "Jab",
    "Cross",
    "Lead Hook",
    "Rear Hook",
    "Lead Uppercut",
    "Rear Uppercut",
]

# Number of classes in BoxingVI
NUM_BOXING_CLASSES = len(BOXING_CLASS_NAMES)


@dataclass
class ActionResult:
    """Result from action recognition inference.

    Attributes:
        action: Predicted action class name (e.g., "Jab").
        action_id: Numeric class ID (0-5 for BoxingVI).
        confidence: Prediction confidence (0-1).
        probabilities: Full probability distribution over all classes.
        start_frame: Start frame of the action window.
        end_frame: End frame of the action window.
        fighter_id: Optional fighter/person ID for multi-person tracking.

    Example:
        >>> result = ActionResult(
        ...     action="Jab",
        ...     action_id=0,
        ...     confidence=0.92,
        ...     probabilities=np.array([0.92, 0.03, 0.02, 0.01, 0.01, 0.01]),
        ...     start_frame=0,
        ...     end_frame=25,
        ... )
        >>> result.num_frames
        25
    """

    action: str
    action_id: int
    confidence: float
    probabilities: np.ndarray  # (num_classes,)
    start_frame: int
    end_frame: int
    fighter_id: Optional[int] = None

    @property
    def num_frames(self) -> int:
        """Return the number of frames in this action window."""
        return self.end_frame - self.start_frame

    @property
    def boxing_action(self) -> Optional[BoxingAction]:
        """Return the BoxingAction enum if valid, else None."""
        if 0 <= self.action_id < NUM_BOXING_CLASSES:
            return BoxingAction(self.action_id)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "action_id": self.action_id,
            "confidence": float(self.confidence),
            "probabilities": self.probabilities.tolist(),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fighter_id": self.fighter_id,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ActionResult(action='{self.action}', "
            f"confidence={self.confidence:.3f}, "
            f"frames={self.start_frame}-{self.end_frame})"
        )


class BaseActionRecognizer(ABC):
    """Abstract base class for action recognition models.

    All action recognizers must implement predict() and predict_batch() methods.

    Example:
        >>> class MyActionRecognizer(BaseActionRecognizer):
        ...     @property
        ...     def num_classes(self):
        ...         return 6
        ...     @property
        ...     def class_names(self):
        ...         return BOXING_CLASS_NAMES
        ...     def predict(self, skeleton_sequence, start_frame=0):
        ...         # Implementation here
        ...         pass
        ...     def predict_batch(self, skeleton_sequences, start_frames=None):
        ...         return [self.predict(s) for s in skeleton_sequences]
    """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of action classes."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return list of class names."""
        pass

    @abstractmethod
    def predict(
        self,
        skeleton_sequence: np.ndarray,
        start_frame: int = 0,
    ) -> ActionResult:
        """Predict action from skeleton sequence.

        Args:
            skeleton_sequence: Skeleton data of shape (T, 17, 2) or (T, 17, 3).
                T = number of frames
                17 = COCO keypoints
                2/3 = x, y (and optionally confidence)
            start_frame: Starting frame index for the sequence.

        Returns:
            ActionResult with predicted action.
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        skeleton_sequences: np.ndarray,
        start_frames: Optional[list[int]] = None,
    ) -> list[ActionResult]:
        """Predict actions from multiple skeleton sequences.

        Args:
            skeleton_sequences: Batch of skeletons (B, T, 17, 2) or (B, T, 17, 3).
            start_frames: Optional list of starting frame indices.

        Returns:
            List of ActionResult, one per sequence.
        """
        pass

    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID.

        Args:
            class_id: Numeric class ID.

        Returns:
            Class name string.

        Raises:
            IndexError: If class_id is out of range.
        """
        return self.class_names[class_id]

    def get_class_id(self, class_name: str) -> int:
        """Get class ID from name.

        Args:
            class_name: Class name string (case-insensitive).

        Returns:
            Numeric class ID.

        Raises:
            ValueError: If class_name is not found.
        """
        class_name_lower = class_name.lower()
        for idx, name in enumerate(self.class_names):
            if name.lower() == class_name_lower:
                return idx
        raise ValueError(f"Unknown class: {class_name}")
