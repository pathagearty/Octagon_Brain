# Action Recognition Component

> **Status:** 🔴 Not Started | **Progress:** 0%

---

## Overview

The action recognition component classifies combat techniques (punches, kicks, defense) from skeleton sequences.

**Models:**
- SkateFormer (primary, SOTA)
- Rule-based classifier (fallback)

---

## Files

```
src/octagon/models/action/
├── __init__.py
├── base.py             # Abstract base class
├── skateformer.py      # SkateFormer wrapper
└── rule_based.py       # Heuristic classifier
```

---

## Target Classes (20)

| ID | Class | Category |
|----|-------|----------|
| 0-7 | jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut, overhand, body_shot | Punches |
| 8-15 | lead_roundhouse, rear_roundhouse, lead_teep, rear_teep, side_kick, back_kick, axe_kick, low_kick | Kicks |
| 16-19 | slip, bob_weave, check, stance_change | Defense |

---

## API

### Base Class

```python
@dataclass
class ActionResult:
    action: str
    confidence: float
    start_frame: int
    end_frame: int
    hand_or_leg: str  # 'left' or 'right'

class BaseActionRecognizer(ABC):
    @abstractmethod
    def classify(self, skeleton_sequence: np.ndarray) -> ActionResult:
        """Classify action from skeleton sequence."""
        pass
```

### SkateFormer

```python
from octagon.models.action import SkateFormer

recognizer = SkateFormer(
    num_classes=20,
    num_keypoints=17,
    sequence_length=64,
    checkpoint='weights/skateformer_combat.pth',
)

result = recognizer.classify(skeleton_sequence)
# skeleton_sequence: (3, 64, 17, 1) - channels, frames, joints, persons
```

### Rule-Based

```python
from octagon.models.action import RuleBasedClassifier

recognizer = RuleBasedClassifier()
result = recognizer.classify(skeleton_sequence)
```

---

## Dependencies

- `torch`
- `einops`
- `numpy`
- SkateFormer source code

---

## Implementation Status

### Completed
- [ ] Nothing yet

### TODO
- [ ] Set up SkateFormer as submodule/dependency
- [ ] Implement base class interface
- [ ] Implement SkateFormer wrapper
- [ ] Implement rule-based classifier
- [ ] Add COCO to NTU keypoint mapping
- [ ] Create training data pipeline
- [ ] Fine-tune on combat sports data
- [ ] Write unit tests

---

## References

- [docs/model-docs/skateformer.md](../model-docs/skateformer.md)
