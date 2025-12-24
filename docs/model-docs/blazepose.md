# BlazePose Documentation

> **Source:** [MediaPipe Pose Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
> **GitHub:** [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
> **Research Blog:** [On-device Real-time Body Pose Tracking](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/)
> **Paper:** [BlazePose GHUM Holistic (arXiv:2206.11678)](https://arxiv.org/abs/2206.11678)
> **Last Updated:** 2025-12-23

---

## Overview

BlazePose is Google's mobile-optimized pose estimation model, part of MediaPipe Solutions. It provides **33 keypoints** including hands, feet, and face landmarks - significantly more than COCO's 17 keypoints. The model runs in real-time on mobile devices and browsers.

### Key Innovations ([Source](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/))

- **33 Keypoints:** Superset of COCO (17), BlazeFace, and BlazePalm topologies
- **3D Pose:** Real-world 3D coordinates in meters using GHUM model
- **Two-stage Pipeline:** Detector + Tracker for efficient real-time performance
- **Mobile-first:** Real-time on most modern phones (15+ FPS with 3D)
- **Segmentation Mask:** Optional body segmentation output

**Use in OctagonBrain:** Mobile pose estimation for OctagonCoach app (iOS/Android).

---

## Installation

```bash
# Python package
pip install mediapipe

# Verify installation
python -c "import mediapipe as mp; print(mp.__version__)"
```

**Version:** 0.10.26 (as of December 2025)

**Requirements:** ([Source](https://github.com/google-ai-edge/mediapipe))
- Python 3.10, 3.11, or 3.12
- NumPy, OpenCV (auto-installed)

**Supported Platforms:**
- Android, iOS
- Web (JavaScript/TF.js)
- Desktop (Windows, macOS, Linux)

**License:** Apache-2.0

---

## 33 Keypoint Format

BlazePose outputs 33 landmarks per detected person. ([Source](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker))

### Complete Landmark List

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | nose | 17 | left_pinky |
| 1 | left_eye_inner | 18 | right_pinky |
| 2 | left_eye | 19 | left_index |
| 3 | left_eye_outer | 20 | right_index |
| 4 | right_eye_inner | 21 | left_thumb |
| 5 | right_eye | 22 | right_thumb |
| 6 | right_eye_outer | 23 | left_hip |
| 7 | left_ear | 24 | right_hip |
| 8 | right_ear | 25 | left_knee |
| 9 | mouth_left | 26 | right_knee |
| 10 | mouth_right | 27 | left_ankle |
| 11 | left_shoulder | 28 | right_ankle |
| 12 | right_shoulder | 29 | left_heel |
| 13 | left_elbow | 30 | right_heel |
| 14 | right_elbow | 31 | left_foot_index |
| 15 | left_wrist | 32 | right_foot_index |
| 16 | right_wrist | | |

### Body Region Groups

| Region | Indices | Count |
|--------|---------|-------|
| Face | 0-10 | 11 |
| Upper Body | 11-22 | 12 |
| Lower Body | 23-32 | 10 |

### Key Advantage for Combat Sports

BlazePose provides **detailed hand landmarks (17-22)** and **foot landmarks (29-32)** useful for punch and kick analysis, unlike COCO's 17 keypoints.

---

## Model Variants

Three complexity levels are available: ([Source](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md))

| Variant | Complexity | Detector Input | Tracker Input | Mobile (ms) | Desktop (ms) |
|---------|------------|----------------|---------------|-------------|--------------|
| **Lite** | 0 | 128×128 | 256×256 | ~20 | ~25 |
| **Full** | 1 | 128×128 | 256×256 | ~25 | ~27 |
| **Heavy** | 2 | 128×128 | 256×256 | ~53 | ~38 |

**Recommendation:** Use `model_complexity=0` (Lite) for real-time mobile, `model_complexity=2` (Heavy) for accuracy.

---

## API Reference

### Python API ([Source](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md))

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
```

### Pose Class Constructor

```python
pose = mp_pose.Pose(
    static_image_mode=False,        # False for video, True for images
    model_complexity=1,             # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,          # Temporal smoothing (video only)
    enable_segmentation=False,      # Output segmentation mask
    smooth_segmentation=True,       # Smooth segmentation (video only)
    min_detection_confidence=0.5,   # Detection threshold [0.0, 1.0]
    min_tracking_confidence=0.5,    # Tracking threshold [0.0, 1.0]
)
```

### Configuration Options ([Source](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker))

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `static_image_mode` | bool | False | Runs detection every frame when True |
| `model_complexity` | int | 1 | 0=lite, 1=full, 2=heavy |
| `smooth_landmarks` | bool | True | Temporal smoothing for video |
| `enable_segmentation` | bool | False | Output body mask |
| `smooth_segmentation` | bool | True | Temporal smoothing for mask |
| `min_detection_confidence` | float | 0.5 | Person detection threshold |
| `min_tracking_confidence` | float | 0.5 | Landmark tracking threshold |

---

## Input/Output Format

### Input
- **Image:** RGB image (BGR from OpenCV must be converted)
- **Video Frame:** Decoded frame from cv2.VideoCapture
- **Live Stream:** Real-time camera feed

### Output Attributes

```python
results = pose.process(image_rgb)

# Pose landmarks (normalized 0-1)
results.pose_landmarks.landmark[i].x      # X coordinate (0-1, image width)
results.pose_landmarks.landmark[i].y      # Y coordinate (0-1, image height)
results.pose_landmarks.landmark[i].z      # Depth (relative to hip center)
results.pose_landmarks.landmark[i].visibility  # Confidence (0-1)

# World landmarks (real 3D in meters)
results.pose_world_landmarks.landmark[i].x    # X in meters
results.pose_world_landmarks.landmark[i].y    # Y in meters
results.pose_world_landmarks.landmark[i].z    # Z in meters

# Segmentation mask (if enabled)
results.segmentation_mask  # float32 array, values [0.0, 1.0]
```

### Output Shapes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `pose_landmarks` | (33,) NormalizedLandmark | Normalized image coordinates |
| `pose_world_landmarks` | (33,) Landmark | 3D coordinates in meters |
| `segmentation_mask` | (H, W) float32 | Body mask (if enabled) |

### Coordinate Systems

- **Normalized:** x, y in [0, 1] relative to image size; z is depth relative to hips
- **World:** Real 3D coordinates in meters, origin at hip center
- **Z-depth:** Negative = closer to camera, Positive = farther

---

## Code Examples

### Basic Usage

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize pose estimator
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Process image
image = cv2.imread('frame.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Extract landmarks
if results.pose_landmarks:
    for i, lm in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, vis={lm.visibility:.3f}")

pose.close()
```

### Video Processing

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

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### Extract Landmarks as NumPy Array

```python
import numpy as np

def landmarks_to_numpy(results) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array.

    Returns:
        np.ndarray: Shape (33, 4) with [x, y, z, visibility] per landmark
    """
    if results.pose_landmarks is None:
        return None

    coords = []
    for lm in results.pose_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(coords, dtype=np.float32)  # (33, 4)
```

### 3D World Coordinates

```python
def get_world_landmarks(results) -> np.ndarray:
    """Extract world coordinates in meters.

    Returns:
        np.ndarray: Shape (33, 3) with [x, y, z] in meters
    """
    if results.pose_world_landmarks is None:
        return None

    coords = []
    for lm in results.pose_world_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)  # (33, 3)
```

### Convert to Pixel Coordinates

```python
def landmarks_to_pixels(results, image_width: int, image_height: int) -> np.ndarray:
    """Convert normalized landmarks to pixel coordinates.

    Returns:
        np.ndarray: Shape (33, 2) with [x_px, y_px] per landmark
    """
    if results.pose_landmarks is None:
        return None

    pixels = []
    for lm in results.pose_landmarks.landmark:
        x_px = int(lm.x * image_width)
        y_px = int(lm.y * image_height)
        pixels.append([x_px, y_px])

    return np.array(pixels, dtype=np.int32)  # (33, 2)
```

---

## BlazePose to COCO Mapping

For compatibility with YOLO11-Pose (COCO 17 format):

```python
# BlazePose 33 -> COCO 17 mapping
BLAZEPOSE_TO_COCO = {
    0: 0,    # nose -> nose
    2: 1,    # left_eye -> left_eye
    5: 2,    # right_eye -> right_eye
    7: 3,    # left_ear -> left_ear
    8: 4,    # right_ear -> right_ear
    11: 5,   # left_shoulder -> left_shoulder
    12: 6,   # right_shoulder -> right_shoulder
    13: 7,   # left_elbow -> left_elbow
    14: 8,   # right_elbow -> right_elbow
    15: 9,   # left_wrist -> left_wrist
    16: 10,  # right_wrist -> right_wrist
    23: 11,  # left_hip -> left_hip
    24: 12,  # right_hip -> right_hip
    25: 13,  # left_knee -> left_knee
    26: 14,  # right_knee -> right_knee
    27: 15,  # left_ankle -> left_ankle
    28: 16,  # right_ankle -> right_ankle
}

def blazepose_to_coco(blazepose_landmarks: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 landmarks to COCO 17 format.

    Args:
        blazepose_landmarks: (33, 4) array [x, y, z, visibility]

    Returns:
        COCO landmarks: (17, 3) array [x, y, confidence]
    """
    coco_landmarks = np.zeros((17, 3), dtype=np.float32)

    for bp_idx, coco_idx in BLAZEPOSE_TO_COCO.items():
        coco_landmarks[coco_idx, 0] = blazepose_landmarks[bp_idx, 0]  # x
        coco_landmarks[coco_idx, 1] = blazepose_landmarks[bp_idx, 1]  # y
        coco_landmarks[coco_idx, 2] = blazepose_landmarks[bp_idx, 3]  # visibility -> conf

    return coco_landmarks
```

---

## Performance Benchmarks

### Inference Speed ([Source](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md))

| Platform | Lite (0) | Full (1) | Heavy (2) |
|----------|----------|----------|-----------|
| Pixel 3 (mobile) | ~20ms | ~25ms | ~53ms |
| MacBook Pro 2020 | ~25ms | ~27ms | ~38ms |

### FPS Estimates

| Model | Mobile | Desktop |
|-------|--------|---------|
| Lite | ~50 FPS | ~40 FPS |
| Full | ~40 FPS | ~37 FPS |
| Heavy | ~19 FPS | ~26 FPS |
| 3D GHUM | ~15 FPS | ~20 FPS |

### Accuracy ([Source](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/))

- **Metric:** PCK@0.2 (Percent Correct Keypoints with 20% tolerance)
- **Human Baseline:** 97.2% PCK@0.2
- BlazePose achieves near-human accuracy on yoga/fitness poses

---

## Mobile Integration

### React Native

```javascript
// Using react-native-mediapipe-posedetection
import { usePoseDetection } from 'react-native-mediapipe-posedetection';

function PoseCamera() {
  const { poses, isProcessing } = usePoseDetection(frame, {
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // poses[0].landmarks = array of 33 landmarks
  return <Camera frameProcessor={detectPose} />;
}
```

### iOS (Swift via Vision)

```swift
import Vision

// Note: Apple Vision uses 19 keypoints, not BlazePose 33
let request = VNDetectHumanBodyPoseRequest { request, error in
    guard let observations = request.results as? [VNHumanBodyPoseObservation] else { return }

    for observation in observations {
        if let points = try? observation.recognizedPoints(.all) {
            // Process 19 Vision keypoints
        }
    }
}
```

**Note:** Apple's Vision framework uses different keypoints (19) than BlazePose (33). For full BlazePose on iOS, use MediaPipe's iOS SDK.

### Android (Kotlin)

```kotlin
val options = PoseLandmarkerOptions.builder()
    .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
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

---

## Combat Sports Features

BlazePose's 33 keypoints are valuable for combat analysis:

| Feature | Keypoints Used | Indices |
|---------|----------------|---------|
| Fist position | left/right thumb, index, pinky | 17-22 |
| Foot position | heels, foot indices | 29-32 |
| Guard height | wrists relative to nose | 15, 16, 0 |
| Hip rotation | left/right hips | 23, 24 |
| Stance width | left/right ankles | 27, 28 |
| Head movement | nose, eyes, ears | 0-8 |

### Calculating Combat Metrics

```python
import numpy as np

def calculate_guard_height(landmarks: np.ndarray, image_height: int) -> float:
    """Calculate guard height relative to head.

    Args:
        landmarks: (33, 4) array from BlazePose

    Returns:
        Guard height ratio (0 = at waist, 1 = at head level)
    """
    nose_y = landmarks[0, 1]
    left_wrist_y = landmarks[15, 1]
    right_wrist_y = landmarks[16, 1]
    left_hip_y = landmarks[23, 1]

    # Average wrist height
    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2

    # Normalize: 0 = hip level, 1 = nose level
    height_range = nose_y - left_hip_y
    if abs(height_range) < 1e-6:
        return 0.5

    guard_height = (left_hip_y - avg_wrist_y) / (left_hip_y - nose_y)
    return np.clip(guard_height, 0, 1)


def calculate_stance_width(landmarks: np.ndarray, image_width: int) -> float:
    """Calculate stance width in normalized units.

    Returns:
        Stance width as fraction of image width
    """
    left_ankle_x = landmarks[27, 0]
    right_ankle_x = landmarks[28, 0]

    return abs(right_ankle_x - left_ankle_x)


def calculate_hip_rotation(world_landmarks: np.ndarray) -> float:
    """Calculate hip rotation angle in degrees.

    Args:
        world_landmarks: (33, 3) array from pose_world_landmarks

    Returns:
        Rotation angle in degrees (0 = square, positive = left hip forward)
    """
    left_hip = world_landmarks[23]
    right_hip = world_landmarks[24]

    # Hip vector
    hip_vec = right_hip - left_hip

    # Angle from frontal plane
    angle_rad = np.arctan2(hip_vec[2], hip_vec[0])
    return np.degrees(angle_rad)
```

---

## Common Issues & Solutions

### 1. Performance on Mobile

```python
# Use lite model for real-time
pose = mp_pose.Pose(model_complexity=0)

# Reduce input resolution
small_frame = cv2.resize(frame, (320, 240))
results = pose.process(small_frame)

# Scale landmarks back
scale_x = original_width / 320
scale_y = original_height / 240
```

### 2. Color Space Conversion

```python
# OpenCV reads BGR, MediaPipe needs RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)
```

### 3. No Pose Detected

```python
# Lower confidence thresholds
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# Check for None
if results.pose_landmarks is not None:
    # Process landmarks
    pass
else:
    print("No pose detected")
```

### 4. Jittery Landmarks

```python
# Enable smoothing (default is True)
pose = mp_pose.Pose(smooth_landmarks=True)

# Or apply manual smoothing
from collections import deque

class LandmarkSmoother:
    def __init__(self, window_size: int = 5):
        self.history = deque(maxlen=window_size)

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        self.history.append(landmarks)
        return np.mean(self.history, axis=0)
```

### 5. Z-Depth Interpretation

```python
# z is relative to hip center
# Negative z = closer to camera
# Positive z = farther from camera

hip_center_z = (landmarks[23, 2] + landmarks[24, 2]) / 2  # Should be ~0
wrist_z = landmarks[15, 2]

if wrist_z < hip_center_z:
    print("Wrist is in front of body (toward camera)")
else:
    print("Wrist is behind body")
```

---

## Comparison: BlazePose vs YOLO11-Pose

| Feature | BlazePose | YOLO11-Pose |
|---------|-----------|-------------|
| Keypoints | 33 | 17 |
| 3D coords | Yes (world landmarks) | No |
| Hand detail | Yes (thumb, index, pinky) | No |
| Foot detail | Yes (heel, foot index) | No |
| Multi-person | Limited (one at a time) | Yes (native) |
| Mobile optimized | Yes (primary target) | Yes (via TFLite) |
| Speed (mobile) | 30+ FPS | 25+ FPS |
| Accuracy | Higher for single person | Higher for multi-person |

**Recommendation:**
- Use BlazePose for **single-person mobile** (OctagonCoach training mode)
- Use YOLO11-Pose for **multi-person desktop** (OctagonAnalytics fight analysis)

---

## References

- [MediaPipe Pose Landmarker Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe GitHub Repository](https://github.com/google-ai-edge/mediapipe)
- [BlazePose Research Blog](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/)
- [BlazePose GHUM Paper (arXiv)](https://arxiv.org/abs/2206.11678)
- [TensorFlow.js BlazePose](https://blog.tensorflow.org/2021/05/high-fidelity-pose-tracking-with-mediapipe-blazepose-and-tfjs.html)
- [3D Pose with BlazePose GHUM](https://blog.tensorflow.org/2021/08/3d-pose-detection-with-mediapipe-blazepose-ghum-tfjs.html)
- [MediaPipe Python Guide](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)
- [React Native MediaPipe](https://github.com/EndLess728/react-native-mediapipe-posedetection)
