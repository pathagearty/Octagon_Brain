# ByteTrack Documentation

> **Source:** [Ultralytics Tracking Mode](https://docs.ultralytics.com/modes/track/)
> **GitHub:** [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
> **Paper:** [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
> **Ultralytics Config:** [bytetrack.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
> **Last Updated:** 2025-12-23

---

## Overview

ByteTrack is a multi-object tracking algorithm that associates **every detection box** (including low-confidence ones) to improve tracking through occlusions. Unlike traditional trackers that discard low-score detections, ByteTrack leverages them to maintain track continuity.

### Key Innovations ([Source](https://arxiv.org/abs/2110.06864))

- **Two-stage Association:** First matches high-confidence detections, then low-confidence ones
- **Occlusion Recovery:** Low-confidence detections help maintain tracks during occlusions
- **Simple & Fast:** Pure motion-based tracking without appearance features (ReID)
- **Universal Improvement:** Increases IDF1 by 1-10 points when applied to any tracker

**Authors:** Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, et al.

**Use in OctagonBrain:** Track two fighters consistently throughout fight videos for OctagonAnalytics.

---

## Installation

### Via Ultralytics (Recommended)

```bash
pip install ultralytics
```

ByteTrack is built into Ultralytics YOLO - no separate installation required.

### Standalone Installation

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop

# Additional dependencies
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

**License:** MIT

---

## Algorithm Overview

### Traditional Tracker Problem

Most trackers discard detections below a confidence threshold (e.g., 0.5), causing:
- Lost tracks during occlusions
- Fragmented trajectories
- ID switches

### ByteTrack Solution ([Source](https://arxiv.org/abs/2110.06864))

ByteTrack uses a **two-stage association** strategy:

```
Stage 1: Associate HIGH-confidence detections with existing tracks
         ├─ Use IoU-based Kalman filter prediction
         └─ Match using Hungarian algorithm

Stage 2: Associate remaining tracks with LOW-confidence detections
         ├─ Recover occluded objects
         └─ Filter background noise using tracklet history
```

**Key Insight:** Low-confidence detections from occluded objects have high similarity with existing tracklets, while background noise does not.

---

## Ultralytics Integration

### Basic Usage ([Source](https://docs.ultralytics.com/modes/track/))

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')

# Track with ByteTrack
results = model.track(
    source='fight_video.mp4',
    tracker='bytetrack.yaml',  # Use ByteTrack (vs 'botsort.yaml')
    stream=True,
    persist=True,              # Maintain tracks across frames
)

for result in results:
    boxes = result.boxes
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)
        # track_ids[i] is the persistent ID for detection i
```

### CLI Usage

```bash
# Basic tracking
yolo track model=yolo11m-pose.pt source="video.mp4" tracker="bytetrack.yaml"

# With custom parameters
yolo track model=yolo11m-pose.pt source="video.mp4" tracker="bytetrack.yaml" conf=0.3 iou=0.5
```

---

## Configuration (bytetrack.yaml)

### Complete Default Configuration ([Source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml))

```yaml
# ByteTrack tracker defaults for mode="track"
tracker_type: bytetrack      # Tracker backend: botsort|bytetrack

# Detection thresholds
track_high_thresh: 0.25      # First-stage match threshold (high-confidence)
track_low_thresh: 0.1        # Second-stage threshold (low-confidence)
new_track_thresh: 0.25       # Threshold for creating new tracks

# Track management
track_buffer: 30             # Frames to keep lost tracks alive
match_thresh: 0.8            # IoU threshold for matching

# Score fusion
fuse_score: true             # Fuse confidence with IoU for matching
```

### Parameter Details

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tracker_type` | bytetrack | botsort, bytetrack | Tracking algorithm |
| `track_high_thresh` | 0.25 | 0.0-1.0 | High-confidence detection threshold |
| `track_low_thresh` | 0.1 | 0.0-1.0 | Low-confidence detection threshold |
| `new_track_thresh` | 0.25 | 0.0-1.0 | Threshold to start new track |
| `track_buffer` | 30 | ≥0 | Frames to keep lost tracks |
| `match_thresh` | 0.8 | 0.0-1.0 | IoU matching threshold |
| `fuse_score` | True | bool | Fuse score with IoU distance |

### Custom Configuration

```yaml
# custom_bytetrack.yaml - Optimized for combat sports
tracker_type: bytetrack

# Lower thresholds to catch fast movements
track_high_thresh: 0.2
track_low_thresh: 0.05
new_track_thresh: 0.3

# Longer buffer for clinch/grappling occlusions
track_buffer: 60

# Lower match threshold for fast-moving fighters
match_thresh: 0.6

fuse_score: true
```

Usage:
```python
results = model.track(source='video.mp4', tracker='custom_bytetrack.yaml')
```

---

## API Reference

### Accessing Track Results

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')

results = model.track(source='video.mp4', stream=True, persist=True)

for result in results:
    # Check if tracking IDs exist
    if result.boxes.id is not None:
        # Get tracking data
        track_ids = result.boxes.id.int().cpu().tolist()      # List of track IDs
        boxes = result.boxes.xyxy.cpu().numpy()               # (N, 4) bounding boxes
        confidences = result.boxes.conf.cpu().numpy()         # (N,) confidences
        keypoints = result.keypoints.xy.cpu().numpy()         # (N, 17, 2) for pose

        # Process each tracked person
        for i, track_id in enumerate(track_ids):
            print(f"Track {track_id}: box={boxes[i]}, conf={confidences[i]:.2f}")
```

### Track History

```python
from collections import defaultdict
import numpy as np

class TrackHistory:
    """Maintain position history for each track."""

    def __init__(self, max_length: int = 30):
        self.history = defaultdict(lambda: [])
        self.max_length = max_length

    def update(self, track_id: int, position: tuple):
        """Add position to track history."""
        track = self.history[track_id]
        track.append(position)
        if len(track) > self.max_length:
            track.pop(0)

    def get_trajectory(self, track_id: int) -> np.ndarray:
        """Get trajectory as numpy array."""
        return np.array(self.history[track_id])

# Usage
history = TrackHistory(max_length=60)

for result in results:
    if result.boxes.id is not None:
        for box, track_id in zip(result.boxes.xywh, result.boxes.id):
            center = (float(box[0]), float(box[1]))
            history.update(int(track_id), center)
```

---

## Code Examples

### Basic Pose Tracking

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolo11m-pose.pt')

cap = cv2.VideoCapture('fight_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track with ByteTrack
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
    result = results[0]

    # Process tracked poses
    if result.boxes.id is not None:
        for i, track_id in enumerate(result.boxes.id.int().tolist()):
            keypoints = result.keypoints.xy[i].cpu().numpy()
            box = result.boxes.xyxy[i].cpu().numpy()

            print(f"Person {track_id}: {keypoints.shape}")

    # Visualize
    annotated = result.plot()
    cv2.imshow('Tracking', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Fighter Assignment

For combat sports, assign consistent fighter identities:

```python
from typing import Dict, Optional
import numpy as np

class FighterTracker:
    """Assign and maintain fighter identities (red/blue corner)."""

    def __init__(self):
        self.fighter_ids: Dict[str, Optional[int]] = {'red': None, 'blue': None}
        self.initial_positions: Dict[str, tuple] = {}
        self.initialized = False

    def initialize(self, track_ids: list, boxes: np.ndarray, frame_width: int):
        """
        Assign fighters based on initial horizontal position.
        Left side = red corner, Right side = blue corner.
        """
        if len(track_ids) < 2:
            return

        centers = [(boxes[i][0] + boxes[i][2]) / 2 for i in range(len(track_ids))]
        sorted_indices = np.argsort(centers)

        # Left fighter = red, Right fighter = blue
        self.fighter_ids['red'] = track_ids[sorted_indices[0]]
        self.fighter_ids['blue'] = track_ids[sorted_indices[1]]

        self.initial_positions['red'] = centers[sorted_indices[0]]
        self.initial_positions['blue'] = centers[sorted_indices[1]]

        self.initialized = True

    def get_fighter(self, track_id: int) -> Optional[str]:
        """Get fighter label for a track ID."""
        for fighter, tid in self.fighter_ids.items():
            if tid == track_id:
                return fighter
        return None

    def update(self, result):
        """Update fighter tracking from YOLO result."""
        if result.boxes.id is None:
            return {}

        track_ids = result.boxes.id.int().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()

        if not self.initialized:
            self.initialize(track_ids, boxes, 1920)  # Assume HD

        assignments = {}
        for i, track_id in enumerate(track_ids):
            fighter = self.get_fighter(track_id)
            if fighter:
                assignments[fighter] = {
                    'track_id': track_id,
                    'box': boxes[i],
                    'keypoints': result.keypoints.xy[i].cpu().numpy() if result.keypoints else None
                }

        return assignments

# Usage
fighter_tracker = FighterTracker()

for result in model.track(source='fight.mp4', persist=True, stream=True):
    fighters = fighter_tracker.update(result)

    if 'red' in fighters:
        print(f"Red corner: {fighters['red']['box']}")
    if 'blue' in fighters:
        print(f"Blue corner: {fighters['blue']['box']}")
```

### Handling Track Switches

Fighters may switch positions (ID swap during clinch):

```python
class RobustFighterTracker:
    """Fighter tracker with ID switch recovery."""

    def __init__(self):
        self.fighter_tracks = {'red': [], 'blue': []}
        self.last_positions = {'red': None, 'blue': None}
        self.current_ids = {'red': None, 'blue': None}

    def update(self, track_ids, boxes, keypoints=None):
        """Update with position continuity check."""
        if len(track_ids) < 2:
            return self.current_ids

        # Get current centers
        centers = {}
        for i, tid in enumerate(track_ids):
            center_x = (boxes[i][0] + boxes[i][2]) / 2
            center_y = (boxes[i][1] + boxes[i][3]) / 2
            centers[tid] = (center_x, center_y)

        # Check for ID switches using position continuity
        for fighter in ['red', 'blue']:
            if self.last_positions[fighter] is None:
                continue

            last_pos = self.last_positions[fighter]
            current_id = self.current_ids[fighter]

            # Find closest track to last position
            min_dist = float('inf')
            closest_id = None

            for tid, pos in centers.items():
                dist = np.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_id = tid

            # If closest track is different, might be ID switch
            if closest_id != current_id and min_dist < 100:  # Threshold in pixels
                print(f"Potential ID switch for {fighter}: {current_id} -> {closest_id}")
                self.current_ids[fighter] = closest_id

        # Update last positions
        for fighter, tid in self.current_ids.items():
            if tid in centers:
                self.last_positions[fighter] = centers[tid]

        return self.current_ids
```

---

## ByteTrack vs BoT-SORT

| Feature | ByteTrack | BoT-SORT |
|---------|-----------|----------|
| Algorithm | Two-stage IoU matching | IoU + ReID |
| ReID Support | No | Yes (`with_reid=True`) |
| Speed | Faster | Slower (with ReID) |
| Occlusion Handling | Good (low-conf recovery) | Better (appearance matching) |
| ID Switch Resistance | Good | Better (with ReID) |
| Memory Usage | Lower | Higher (ReID features) |
| Best For | Fast tracking, no appearance | Re-identification needed |

### When to Use Each

- **ByteTrack:** Fast tracking, similar-looking objects, real-time requirements
- **BoT-SORT:** Object re-identification after long occlusions, distinguishing similar objects

---

## Performance Benchmarks

### MOT17 Test Set ([Source](https://arxiv.org/abs/2110.06864))

| Metric | Value |
|--------|-------|
| MOTA | 80.3 |
| IDF1 | 77.3 |
| HOTA | 63.1 |
| FPS | 29.6 (V100 GPU) |
| ID Switches | 2,196 |

### MOT20 Test Set

| Metric | Value |
|--------|-------|
| MOTA | 77.8 |
| IDF1 | 75.2 |
| HOTA | 61.3 |
| FPS | 13.7 |
| ID Switches | 1,223 |

### Model Variants (MOT17)

| Variant | Parameters | MOTA | FPS |
|---------|------------|------|-----|
| Nano | 0.90M | 69.0 | 80+ |
| Small | 9.0M | 78.0 | 60+ |
| Medium | 25.3M | 85.0 | 40+ |
| X-Large | 99.0M | 90.0 | 29.6 |

---

## Common Issues & Solutions

### 1. ID Switches During Clinch

```yaml
# Increase track buffer for clinch scenarios
track_buffer: 60  # Keep tracks longer (default: 30)

# Or use BoT-SORT with ReID
# botsort.yaml
with_reid: true
```

### 2. New Track Creation on Occlusion Exit

```yaml
# Increase new track threshold
new_track_thresh: 0.4  # Higher = harder to create new tracks
```

### 3. Multiple Fighters Detected as One

```yaml
# Lower IoU threshold for NMS
match_thresh: 0.6  # Lower = stricter matching
```

### 4. Lost Tracks During Fast Movement

```yaml
# Lower thresholds
track_high_thresh: 0.15
track_low_thresh: 0.05
```

### 5. Persist Flag Not Working

```python
# Ensure persist=True for continuous tracking
results = model.track(frame, persist=True)  # REQUIRED for frame-to-frame tracking

# Without persist, each frame is tracked independently
```

---

## Combat Sports Configuration

### Recommended Settings for OctagonBrain

```yaml
# combat_bytetrack.yaml
tracker_type: bytetrack

# Lower thresholds for fast combat movements
track_high_thresh: 0.2       # Catch fast punches/kicks
track_low_thresh: 0.05       # Recover from occlusions

# Higher threshold for new tracks (avoid false positives)
new_track_thresh: 0.35

# Longer buffer for clinch/grappling
track_buffer: 60             # ~2 seconds at 30 FPS

# Lower match threshold for overlapping fighters
match_thresh: 0.6

fuse_score: true
```

### Two-Fighter Constraint

```python
def filter_top_two_fighters(result, min_confidence: float = 0.3):
    """Keep only top 2 detections by confidence."""
    if result.boxes.id is None or len(result.boxes) < 2:
        return result

    # Sort by confidence
    confidences = result.boxes.conf.cpu().numpy()
    top_indices = confidences.argsort()[-2:][::-1]

    # Filter to top 2
    # Note: This is a simplified example; actual filtering requires more code
    return result
```

---

## References

- [ByteTrack Paper (arXiv)](https://arxiv.org/abs/2110.06864)
- [ByteTrack GitHub Repository](https://github.com/ifzhang/ByteTrack)
- [Ultralytics Tracking Documentation](https://docs.ultralytics.com/modes/track/)
- [ByteTrack YAML Config](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
- [BoT-SORT YAML Config](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [MOT17 Benchmark](https://motchallenge.net/data/MOT17/)
- [MOT20 Benchmark](https://motchallenge.net/data/MOT20/)
