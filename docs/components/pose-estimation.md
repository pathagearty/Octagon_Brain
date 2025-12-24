# Pose Estimation Component

> **Status:** 🔴 Not Started | **Progress:** 0%

---

## Overview

The pose estimation component extracts human body keypoints from video frames. It's the foundation for all other analysis.

**Models:**
- YOLO11-Pose (desktop, real-time)
- BlazePose (mobile, 33 keypoints)
- ViTPose (teacher, auto-labeling)

---

## Files

```
src/octagon/models/pose/
├── __init__.py
├── base.py           # Abstract base class
├── yolo_pose.py      # YOLO11-Pose wrapper
├── blazepose.py      # BlazePose wrapper
└── vitpose.py        # ViTPose wrapper (teacher)
```

---

## API

### Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class PoseResult:
    keypoints: np.ndarray      # (N, K, 3) - x, y, confidence
    boxes: np.ndarray          # (N, 4) - bounding boxes
    scores: np.ndarray         # (N,) - detection confidence
    num_keypoints: int         # K (17 for COCO, 33 for BlazePose)

class BasePoseEstimator(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> PoseResult:
        """Detect poses in a single frame."""
        pass

    @abstractmethod
    def detect_batch(self, frames: list[np.ndarray]) -> list[PoseResult]:
        """Detect poses in a batch of frames."""
        pass
```

### YOLO11-Pose

```python
from octagon.models.pose import YOLO11Pose

estimator = YOLO11Pose(
    model_size='m',      # n, s, m, l, x
    device='cuda',
    conf_threshold=0.25,
)

result = estimator.detect(frame)
```

### BlazePose

```python
from octagon.models.pose import BlazePose

estimator = BlazePose(
    model_complexity=1,   # 0=lite, 1=full, 2=heavy
)

result = estimator.detect(frame)
# result.num_keypoints = 33
```

---

## Dependencies

- `ultralytics` - YOLO11
- `mediapipe` - BlazePose
- `mmpose` - ViTPose (optional)
- `numpy`
- `opencv-python`

---

## Implementation Status

### Completed
- [ ] Nothing yet

### In Progress
- [ ] Nothing yet

### TODO
- [ ] Define base class interface (`BasePoseEstimator`)
- [ ] Implement YOLO11-Pose wrapper
- [ ] Implement BlazePose wrapper
- [ ] Implement ViTPose wrapper (teacher)
- [ ] Add pose normalization utilities
- [ ] Add keypoint format conversion (COCO ↔ BlazePose)
- [ ] Add batch processing support
- [ ] Add GPU/TensorRT optimization
- [ ] Write unit tests
- [ ] Add benchmarking script

---

## Known Issues

None (not yet implemented).

---

## References

- [docs/model-docs/yolo11-pose.md](../model-docs/yolo11-pose.md)
- [docs/model-docs/blazepose.md](../model-docs/blazepose.md)
- [docs/model-docs/mediapipe.md](../model-docs/mediapipe.md)
