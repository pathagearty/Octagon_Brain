# MediaPipe Documentation

> **Source:** [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe)
> **GitHub:** [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
> **Python Guide:** [MediaPipe Python Setup](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
> **PyPI:** [mediapipe on PyPI](https://pypi.org/project/mediapipe/)
> **Last Updated:** 2025-12-23

---

## Overview

MediaPipe is Google's framework for building cross-platform ML pipelines. It provides pre-built, optimized solutions for pose, hands, face, and more, running in real-time on mobile devices, web browsers, and desktop applications.

### Key Features ([Source](https://ai.google.dev/edge/mediapipe))

- **MediaPipe Tasks:** Cross-platform APIs for deploying AI solutions
- **MediaPipe Models:** Pre-trained, ready-to-run models
- **MediaPipe Model Maker:** Customize models with your data
- **MediaPipe Studio:** Browser-based visualization and benchmarking

**Use in OctagonBrain:** Mobile ML integration, especially for BlazePose and real-time camera processing in OctagonCoach.

---

## Installation

```bash
# Python package
pip install mediapipe

# Verify installation
python -c "import mediapipe as mp; print(mp.__version__)"
```

**Version:** 0.10.26 (as of December 2025)

**Requirements:** ([Source](https://ai.google.dev/edge/mediapipe/solutions/setup_python))
- Python 3.9 - 3.12
- PIP 20.3+
- NumPy, OpenCV (auto-installed)

**Supported Platforms:**
- Desktop: Windows, Mac, Linux
- Mobile: Android, iOS
- Web: JavaScript/TF.js
- IoT: Raspberry OS 64-bit

**License:** Apache-2.0

---

## Available Solutions

### Vision Tasks ([Source](https://ai.google.dev/edge/mediapipe))

| Solution | Description | Keypoints/Output |
|----------|-------------|------------------|
| **Pose Landmarker** | Full body pose | 33 landmarks |
| **Hand Landmarker** | Hand tracking | 21 landmarks per hand |
| **Face Landmarker** | Face mesh | 468 landmarks |
| **Holistic Landmarker** | Pose + hands + face | 540+ landmarks |
| **Object Detection** | Detect objects | Bounding boxes + labels |
| **Image Segmentation** | Pixel-level segmentation | Segmentation mask |
| **Gesture Recognition** | Hand gestures | Gesture class |
| **Face Detection** | Face bounding box | Bounding box + landmarks |

### Text Tasks

| Solution | Description |
|----------|-------------|
| Text Classification | Classify text |
| Text Embedding | Generate embeddings |
| Language Detection | Detect language |

### Audio Tasks

| Solution | Description |
|----------|-------------|
| Audio Classification | Classify audio |

---

## Pose Solution (BlazePose)

See [blazepose.md](blazepose.md) for detailed documentation.

### Quick Reference

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,           # 0=lite, 1=full, 2=heavy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Process frame
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Draw landmarks
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
    )

pose.close()
```

### 33 Pose Landmarks

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | nose | 17 | left_pinky |
| 1-6 | eyes | 18 | right_pinky |
| 7-8 | ears | 19-22 | fingers |
| 9-10 | mouth | 23-24 | hips |
| 11-12 | shoulders | 25-26 | knees |
| 13-14 | elbows | 27-28 | ankles |
| 15-16 | wrists | 29-32 | feet |

---

## Hands Solution

### 21 Hand Landmarks ([Source](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker))

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | WRIST | 11 | MIDDLE_FINGER_DIP |
| 1 | THUMB_CMC | 12 | MIDDLE_FINGER_TIP |
| 2 | THUMB_MCP | 13 | RING_FINGER_MCP |
| 3 | THUMB_IP | 14 | RING_FINGER_PIP |
| 4 | THUMB_TIP | 15 | RING_FINGER_DIP |
| 5 | INDEX_FINGER_MCP | 16 | RING_FINGER_TIP |
| 6 | INDEX_FINGER_PIP | 17 | PINKY_MCP |
| 7 | INDEX_FINGER_DIP | 18 | PINKY_PIP |
| 8 | INDEX_FINGER_TIP | 19 | PINKY_DIP |
| 9 | MIDDLE_FINGER_MCP | 20 | PINKY_TIP |
| 10 | MIDDLE_FINGER_PIP | | |

**Joint Abbreviations:**
- CMC = Carpometacarpal joint
- MCP = Metacarpophalangeal joint (knuckle)
- PIP = Proximal interphalangeal joint
- DIP = Distal interphalangeal joint
- IP = Interphalangeal joint (thumb only)
- TIP = Fingertip

### Hand Landmarker Usage

```python
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

results = hands.process(rgb_image)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Access specific landmarks
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        print(f"Wrist: ({wrist.x:.3f}, {wrist.y:.3f})")
        print(f"Thumb tip: ({thumb_tip.x:.3f}, {thumb_tip.y:.3f})")

hands.close()
```

### Hand Landmarker Configuration ([Source](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker))

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `running_mode` | IMAGE | IMAGE, VIDEO, LIVE_STREAM | Processing mode |
| `num_hands` | 1 | >0 | Maximum hands to detect |
| `min_hand_detection_confidence` | 0.5 | 0.0-1.0 | Detection threshold |
| `min_hand_presence_confidence` | 0.5 | 0.0-1.0 | Presence threshold |
| `min_tracking_confidence` | 0.5 | 0.0-1.0 | Tracking threshold |

### Hand Landmarker Performance

| Device | CPU Latency | GPU Latency |
|--------|-------------|-------------|
| Pixel 6 | 17.12ms | 12.27ms |

---

## Holistic Solution

Combines Pose (33) + Hands (21×2) + Face (468) = **540+ landmarks**

```python
import mediapipe as mp

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

results = holistic.process(rgb_image)

# Access all components
if results.pose_landmarks:
    # 33 pose landmarks
    pass

if results.left_hand_landmarks:
    # 21 left hand landmarks
    pass

if results.right_hand_landmarks:
    # 21 right hand landmarks
    pass

if results.face_landmarks:
    # 468 face landmarks
    pass

holistic.close()
```

### Holistic Pipeline Architecture ([Source](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/))

```
Input Frame
    │
    ▼
BlazePose Detector + Landmark Model (33 keypoints)
    │
    ├──► Derive Face ROI ──► Face Mesh Model (468 landmarks)
    │
    ├──► Derive Left Hand ROI ──► Hand Landmark Model (21 landmarks)
    │
    └──► Derive Right Hand ROI ──► Hand Landmark Model (21 landmarks)
    │
    ▼
Combined Output (540+ landmarks)
```

---

## Code Examples

### Video Processing with Pose

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('video.mp4')

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = pose.process(rgb)

        # Draw
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### Extract Landmarks as NumPy

```python
import numpy as np

def pose_landmarks_to_numpy(results) -> np.ndarray:
    """Convert pose landmarks to numpy array.

    Returns:
        np.ndarray: Shape (33, 4) with [x, y, z, visibility]
    """
    if results.pose_landmarks is None:
        return None

    coords = []
    for lm in results.pose_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(coords, dtype=np.float32)  # (33, 4)


def hand_landmarks_to_numpy(hand_landmarks) -> np.ndarray:
    """Convert hand landmarks to numpy array.

    Returns:
        np.ndarray: Shape (21, 3) with [x, y, z]
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)  # (21, 3)
```

### Calculate Joint Angles

```python
import numpy as np

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b given points a, b, c.

    Args:
        a, b, c: Points as (x, y) or (x, y, z) arrays

    Returns:
        Angle in degrees
    """
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1, 1))

    return np.degrees(angle)


# Example: Calculate elbow angle
def get_elbow_angle(pose_landmarks: np.ndarray, side: str = 'left') -> float:
    """Get elbow angle from pose landmarks.

    Args:
        pose_landmarks: (33, 4) array
        side: 'left' or 'right'

    Returns:
        Elbow angle in degrees
    """
    if side == 'left':
        shoulder = pose_landmarks[11, :2]
        elbow = pose_landmarks[13, :2]
        wrist = pose_landmarks[15, :2]
    else:
        shoulder = pose_landmarks[12, :2]
        elbow = pose_landmarks[14, :2]
        wrist = pose_landmarks[16, :2]

    return calculate_angle(shoulder, elbow, wrist)
```

---

## Mobile Integration

### React Native

```javascript
// Using react-native-vision-camera with ML Kit
import { usePoseDetection } from 'react-native-vision-camera-mlkit';

function PoseCamera() {
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    const poses = detectPose(frame);
    console.log(poses);
  }, []);

  return (
    <Camera
      frameProcessor={frameProcessor}
      frameProcessorFps={30}
    />
  );
}
```

### Android (Kotlin)

```kotlin
import com.google.mediapipe.solutions.pose.Pose

val options = PoseLandmarkerOptions.builder()
    .setBaseOptions(
        BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .build()
    )
    .setRunningMode(RunningMode.VIDEO)
    .setNumPoses(1)
    .setMinPoseDetectionConfidence(0.5f)
    .setMinTrackingConfidence(0.5f)
    .build()

val poseLandmarker = PoseLandmarker.createFromOptions(context, options)

// Process frame
val result = poseLandmarker.detectForVideo(mpImage, timestampMs)
val landmarks = result.landmarks()
```

### iOS (Swift)

```swift
import MediaPipeTasksVision

let options = PoseLandmarkerOptions()
options.baseOptions.modelAssetPath = "pose_landmarker_lite.task"
options.runningMode = .video
options.minPoseDetectionConfidence = 0.5
options.minTrackingConfidence = 0.5

let poseLandmarker = try PoseLandmarker(options: options)

// Process frame
let result = try poseLandmarker.detect(videoFrame: frame, timestampInMilliseconds: timestamp)
let landmarks = result.landmarks
```

---

## Export & Deployment

### Model Download Locations

Pre-trained models available at:
```
https://storage.googleapis.com/mediapipe-models/
```

Example paths:
- `pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`
- `pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
- `pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`
- `hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`

### TensorFlow Lite

```python
# MediaPipe models are TFLite-optimized by default
# Download .tflite or .task files from storage

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model from file
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)
```

### CoreML Conversion (iOS)

```python
import coremltools as ct

# Convert TFLite to CoreML
model = ct.convert('pose_landmarker.tflite')
model.save('PoseLandmarker.mlmodel')
```

---

## Performance Optimization

### Reduce Latency

```python
# Use lite model
pose = mp_pose.Pose(model_complexity=0)

# Reduce input resolution
small_frame = cv2.resize(frame, (320, 240))
results = pose.process(small_frame)

# Scale landmarks back to original size
scale_x = original_width / 320
scale_y = original_height / 240
```

### GPU Acceleration

```python
# MediaPipe uses GPU automatically when available
# For explicit GPU control, use MediaPipe Tasks API:

from mediapipe.tasks.python import BaseOptions

base_options = BaseOptions(
    model_asset_path='model.task',
    delegate=BaseOptions.Delegate.GPU  # or CPU
)
```

### Benchmark Comparison

| Model | Complexity | CPU (ms) | GPU (ms) | Keypoints |
|-------|------------|----------|----------|-----------|
| Pose Lite | 0 | 20 | 10 | 33 |
| Pose Full | 1 | 25 | 12 | 33 |
| Pose Heavy | 2 | 53 | 25 | 33 |
| Hand | - | 17 | 12 | 21 |
| Face Mesh | - | 15 | 8 | 468 |
| Holistic | 1 | 80+ | 40+ | 540+ |

---

## Common Issues & Solutions

### 1. BGR vs RGB Color Space

```python
# OpenCV reads BGR, MediaPipe needs RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)
```

### 2. No Detection

```python
# Lower confidence thresholds
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# Check for None results
if results.pose_landmarks is None:
    print("No pose detected")
```

### 3. Memory Leaks

```python
# Always close MediaPipe objects
pose.close()  # or use context manager

# Use context manager (recommended)
with mp_pose.Pose() as pose:
    results = pose.process(image)
```

### 4. Slow Performance

```python
# Use lite model
pose = mp_pose.Pose(model_complexity=0)

# Disable unused features
pose = mp_pose.Pose(
    enable_segmentation=False,
    smooth_landmarks=False,  # Faster but jittery
)
```

### 5. Coordinate Normalization

```python
# Landmarks are normalized [0, 1]
# Convert to pixel coordinates:
x_px = int(landmark.x * image_width)
y_px = int(landmark.y * image_height)
```

---

## Comparison: MediaPipe vs YOLO11-Pose

| Feature | MediaPipe (BlazePose) | YOLO11-Pose |
|---------|----------------------|-------------|
| Keypoints | 33 | 17 |
| 3D coords | Yes | No |
| Hand detail | Separate (21/hand) | No |
| Face detail | Separate (468) | No |
| Multi-person | Limited (1) | Yes (native) |
| Mobile optimized | Yes (primary) | Yes (via TFLite) |
| Speed (mobile) | 30+ FPS | 25+ FPS |
| Framework | TFLite/MediaPipe | PyTorch/ONNX |

**Recommendations:**
- **MediaPipe:** Single-person mobile apps (OctagonCoach training)
- **YOLO11-Pose:** Multi-person desktop analysis (OctagonAnalytics)

---

## Combat Sports Application

### Key Landmarks for Combat Analysis

| Feature | Solution | Landmarks |
|---------|----------|-----------|
| Guard position | Pose | 15, 16 (wrists) |
| Fist formation | Hands | 0, 4, 8, 12, 16, 20 (tips) |
| Foot placement | Pose | 27-32 (ankles, feet) |
| Hip rotation | Pose | 23, 24 (hips) |
| Head movement | Pose/Face | 0-10 (head landmarks) |
| Punch tracking | Hands | Full 21 landmarks |

### Example: Fist Detection

```python
def is_fist_closed(hand_landmarks: np.ndarray) -> bool:
    """Detect if hand is making a fist.

    Args:
        hand_landmarks: (21, 3) array from MediaPipe Hands

    Returns:
        True if fist is closed
    """
    # Get fingertip positions relative to knuckles
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
    finger_mcps = [2, 5, 9, 13, 17]   # Corresponding knuckles

    closed_count = 0
    for tip_idx, mcp_idx in zip(finger_tips[1:], finger_mcps[1:]):
        tip_y = hand_landmarks[tip_idx, 1]
        mcp_y = hand_landmarks[mcp_idx, 1]

        # Tip below knuckle = finger curled (in normalized coords)
        if tip_y > mcp_y:
            closed_count += 1

    # 4 fingers curled = fist
    return closed_count >= 3
```

---

## References

- [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe)
- [MediaPipe GitHub Repository](https://github.com/google-ai-edge/mediapipe)
- [MediaPipe Python Setup](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
- [Pose Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [Hand Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Face Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Holistic Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker)
- [MediaPipe Studio](https://mediapipe-studio.webapps.google.com/)
- [MediaPipe Research Blog](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/)
- [PyPI Package](https://pypi.org/project/mediapipe/)
