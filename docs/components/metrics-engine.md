# Metrics Engine Component

> **Status:** 🔴 Not Started | **Progress:** 0%

---

## Overview

The metrics engine calculates combat-sports-specific metrics from pose data, including biomechanics, technique quality, and fight statistics.

---

## Files

```
src/octagon/metrics/
├── __init__.py
├── static.py           # Single-frame metrics
├── dynamic.py          # Sequence metrics
└── biomechanics.py     # CoM, weight distribution
```

---

## Metrics Categories

### Static Metrics (per frame)

| Metric | Description | Formula |
|--------|-------------|---------|
| Stance Width | Feet distance / shoulder width | `dist(ankle_l, ankle_r) / dist(shoulder_l, shoulder_r)` |
| Guard Height | Wrist position relative to chin | `(wrist_y - nose_y)` |
| Hip Rotation | Angle of hip line from frontal | `atan2(hip_r.x - hip_l.x, hip_r.z - hip_l.z)` |
| Shoulder Rotation | Angle of shoulder line | Similar to hip |
| Weight Distribution | Front/back foot loading | CoM projection |

### Dynamic Metrics (per sequence)

| Metric | Description |
|--------|-------------|
| Strike Velocity | Wrist/ankle displacement over time |
| Acceleration | Velocity derivative |
| Strike Duration | Frames from start to contact |
| Recovery Time | Frames from contact to guard |

### Biomechanics

| Metric | Description |
|--------|-------------|
| Center of Mass | Weighted average of segment centers |
| Base of Support | Polygon formed by feet |
| Balance Score | CoM relative to BoS |

---

## API

```python
from octagon.metrics import MetricsEngine

engine = MetricsEngine()

# Static metrics from single pose
static = engine.calculate_static(pose)
# Returns: StanceWidth, GuardHeight, HipRotation, etc.

# Dynamic metrics from sequence
dynamic = engine.calculate_dynamic(pose_sequence)
# Returns: StrikeVelocity, Acceleration, etc.

# Biomechanics
bio = engine.calculate_biomechanics(pose)
# Returns: CenterOfMass, WeightDistribution
```

---

## Center of Mass Calculation

```python
SEGMENT_MASSES = {
    'head': 0.08,
    'trunk': 0.50,
    'upper_arm': 0.03,
    'forearm': 0.02,
    'thigh': 0.10,
    'lower_leg': 0.05,
}

def calculate_com(pose: np.ndarray) -> np.ndarray:
    """Calculate center of mass from pose."""
    # Segment centers as midpoints
    segments = {
        'head': pose[0],  # nose
        'trunk': (pose[5] + pose[6] + pose[11] + pose[12]) / 4,
        'left_upper_arm': (pose[5] + pose[7]) / 2,
        # ... etc
    }

    com = np.zeros(3)
    for segment, center in segments.items():
        mass = SEGMENT_MASSES[segment.replace('left_', '').replace('right_', '')]
        com += center * mass

    return com
```

---

## Implementation Status

### TODO
- [ ] Define metrics data classes
- [ ] Implement static metrics
- [ ] Implement dynamic metrics
- [ ] Implement CoM estimation
- [ ] Implement weight distribution
- [ ] Add velocity/acceleration calculation
- [ ] Write unit tests

---

## References

- [docs/RESEARCH.md - Biomechanics Section](../RESEARCH.md)
