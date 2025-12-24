# Multi-Person Tracking Component

> **Status:** 🔴 Not Started | **Progress:** 0%

---

## Overview

The tracking component maintains consistent identities for fighters across video frames. Used in OctagonAnalytics for fight analysis.

**Models:**
- ByteTrack (primary)
- BoT-SORT (alternative with ReID)

---

## Files

```
src/octagon/models/tracking/
├── __init__.py
├── base.py           # Abstract base class
├── byte_track.py     # ByteTrack wrapper
└── fighter_tracker.py # Fighter-specific logic
```

---

## API

```python
from octagon.models.tracking import FighterTracker

tracker = FighterTracker(
    tracker_type='bytetrack',
    track_buffer=30,
)

# Initialize with first frame positions
tracker.initialize({
    'red': (x1, y1),    # Red corner initial position
    'blue': (x2, y2),   # Blue corner initial position
})

# Update each frame
fighters = tracker.update(detections)
# Returns: {'red': Detection, 'blue': Detection}
```

---

## Features

- **Consistent IDs:** Maintain fighter identity through occlusions
- **Corner Assignment:** Assign red/blue based on initial positions
- **Clinch Handling:** Special logic for close contact
- **ID Recovery:** Re-acquire tracks after separation

---

## Implementation Status

### TODO
- [ ] Implement ByteTrack wrapper
- [ ] Implement fighter assignment logic
- [ ] Add clinch detection
- [ ] Add ID recovery mechanism
- [ ] Integrate with pose estimation
- [ ] Write unit tests

---

## References

- [docs/model-docs/bytetrack.md](../model-docs/bytetrack.md)
