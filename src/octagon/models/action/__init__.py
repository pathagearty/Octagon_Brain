"""Action recognition models for OctagonBrain.

This module provides action recognition models for classifying combat sports
techniques from skeleton sequences.

Main Classes:
    - SkateFormer: Official KAIST SkateFormer architecture (ECCV 2024)
      with pretrained weight support for transfer learning
    - SkateFormerConfig: Configuration dataclass for SkateFormer
    - SkateFormerWrapper: Inference wrapper for SkateFormer

Legacy Classes (deprecated):
    - SkateFormerLegacy: Original simplified implementation
    - SkateFormerWrapperLegacy: Original wrapper
"""
from .base import (
    BOXING_CLASS_NAMES,
    NUM_BOXING_CLASSES,
    ActionResult,
    BaseActionRecognizer,
    BoxingAction,
)
from .skateformer import SkateFormer, SkateFormerWrapper
from .skateformer_config import SkateFormerConfig
from .skateformer_legacy import SkateFormerLegacy, SkateFormerWrapperLegacy

__all__ = [
    # Models - Official Architecture
    "SkateFormer",
    "SkateFormerWrapper",
    "SkateFormerConfig",
    # Models - Legacy (deprecated)
    "SkateFormerLegacy",
    "SkateFormerWrapperLegacy",
    # Data structures
    "ActionResult",
    "BoxingAction",
    # Base class
    "BaseActionRecognizer",
    # Constants
    "BOXING_CLASS_NAMES",
    "NUM_BOXING_CLASSES",
]
