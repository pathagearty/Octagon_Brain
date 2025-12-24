"""Action recognition models for OctagonBrain.

This module provides action recognition models for classifying combat sports
techniques from skeleton sequences.
"""
from .base import (
    BOXING_CLASS_NAMES,
    NUM_BOXING_CLASSES,
    ActionResult,
    BaseActionRecognizer,
    BoxingAction,
)
from .skateformer import SkateFormer, SkateFormerWrapper

__all__ = [
    # Models
    "SkateFormer",
    "SkateFormerWrapper",
    # Data structures
    "ActionResult",
    "BoxingAction",
    # Base class
    "BaseActionRecognizer",
    # Constants
    "BOXING_CLASS_NAMES",
    "NUM_BOXING_CLASSES",
]
