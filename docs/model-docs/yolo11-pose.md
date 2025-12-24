# YOLO11-Pose Documentation

> **Source:** [Ultralytics Pose Estimation Docs](https://docs.ultralytics.com/tasks/pose/)
> **GitHub:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
> **Model Page:** [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
> **Last Updated:** 2025-12-23

---

## Overview

YOLO11-Pose is the latest pose estimation model from Ultralytics, released on **2024-09-30** (version 11.0.0). It performs person detection and keypoint estimation in a single forward pass, making it ideal for real-time applications.

### Key Innovations ([Source](https://docs.ultralytics.com/models/yolo11/))

- **Improved backbone and neck architecture** for enhanced feature extraction
- **22% fewer parameters** than YOLOv8m while achieving higher mAP
- **Multi-format export support** for edge devices, cloud platforms, and NVIDIA GPUs
- **Single-pass detection** of both bounding boxes and 17 keypoints

**Use in OctagonBrain:** Primary model for real-time pose estimation in OctagonCoach.

---

## Installation

```bash
# Install ultralytics (Python>=3.8, PyTorch>=1.8 required)
pip install ultralytics

# Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```

**Requirements:** ([Source](https://github.com/ultralytics/ultralytics))
- Python >= 3.8
- PyTorch >= 1.8
- CUDA (optional, for GPU acceleration)

**License:** AGPL-3.0 (open source) or Enterprise License (commercial)

---

## Model Variants

All models are trained on the [COCO Keypoints](https://docs.ultralytics.com/datasets/pose/coco/) dataset with 640×640 input resolution.

| Model | mAP^pose 50-95 | mAP^pose 50 | CPU Speed (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|----------------|-------------|----------------|------------------|------------|-----------|
| [yolo11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | 50.0 | 81.0 | 52.4 ± 0.5 | 1.7 ± 0.0 | 2.9 | 7.4 |
| [yolo11s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) | 58.9 | 86.3 | 90.5 ± 0.6 | 2.6 ± 0.0 | 9.9 | 23.1 |
| **[yolo11m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt)** | **64.9** | **89.4** | 187.3 ± 0.8 | 4.9 ± 0.1 | 20.9 | 71.4 |
| [yolo11l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt) | 66.1 | 89.9 | 247.7 ± 1.1 | 6.4 ± 0.1 | 26.1 | 90.3 |
| [yolo11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | 69.5 | 91.1 | 488.0 ± 13.9 | 12.1 ± 0.2 | 58.8 | 202.8 |

**Source:** [Ultralytics Pose Docs - Models Table](https://docs.ultralytics.com/tasks/pose/#models)

**Recommendation:** Use `yolo11m-pose` for balance of speed (4.9ms on T4) and accuracy (64.9 mAP).

---

## Keypoint Format (COCO 17)

The model outputs 17 keypoints per detected person following the COCO format:

```
Index  Name              Index  Name
-----  ----              -----  ----
0      nose              9      left_wrist
1      left_eye          10     right_wrist
2      right_eye         11     left_hip
3      left_ear          12     right_hip
4      right_ear         13     left_knee
5      left_shoulder     14     right_knee
6      right_shoulder    15     left_ankle
7      left_elbow        16     right_ankle
8      right_elbow
```

**Skeleton Connections:** ([Source](https://docs.ultralytics.com/datasets/pose/coco/))
```
Head:  0-1, 0-2, 1-3, 2-4
Arms:  5-7, 7-9, 6-8, 8-10
Torso: 5-6, 5-11, 6-12, 11-12
Legs:  11-13, 13-15, 12-14, 14-16
```

**Flip Indexing (for augmentation):** `[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`

---

## API Reference

### Results Class ([Source](https://docs.ultralytics.com/reference/engine/results/))

```python
results = model('image.jpg')
result = results[0]  # First image result

# Key attributes
result.orig_img        # np.ndarray - Original image
result.orig_shape      # tuple - (height, width)
result.boxes           # Boxes object - Bounding boxes
result.keypoints       # Keypoints object - Pose keypoints
result.names           # dict - Class name mapping
result.speed           # dict - Processing times (preprocess, inference, postprocess)

# Methods
result.plot()          # Returns annotated image as np.ndarray
result.show()          # Display annotated image
result.save(filename)  # Save annotated image
result.summary()       # Returns results as list of dicts
```

### Boxes Class ([Source](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes))

```python
boxes = result.boxes

# Bounding box coordinates
boxes.xyxy             # torch.Tensor (N, 4) - [x1, y1, x2, y2] in pixels
boxes.xywh             # torch.Tensor (N, 4) - [x_center, y_center, width, height]
boxes.xyxyn            # torch.Tensor (N, 4) - Normalized [0-1] coordinates
boxes.xywhn            # torch.Tensor (N, 4) - Normalized center format

# Detection info
boxes.conf             # torch.Tensor (N,) - Confidence scores
boxes.cls              # torch.Tensor (N,) - Class IDs (0 = person)
boxes.id               # torch.Tensor (N,) - Track IDs (if tracking enabled)
boxes.is_track         # bool - Whether tracking IDs are present
```

### Keypoints Class ([Source](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints))

```python
keypoints = result.keypoints

# Coordinates
keypoints.xy           # torch.Tensor (N, 17, 2) - [x, y] in pixels
keypoints.xyn          # torch.Tensor (N, 17, 2) - Normalized [0-1] coordinates
keypoints.conf         # torch.Tensor (N, 17) - Confidence per keypoint
keypoints.data         # torch.Tensor (N, 17, 3) - [x, y, confidence]
keypoints.has_visible  # bool - Whether confidence data is available

# Conversion methods
keypoints.cpu()        # Returns CPU tensor copy
keypoints.cuda()       # Returns GPU tensor copy
keypoints.numpy()      # Returns numpy array
keypoints.to(device)   # Transfer to specified device
```

---

## Input/Output Format

### Input
- **Image:** Any format readable by PIL/OpenCV (JPEG, PNG, etc.)
- **Video:** MP4, AVI, MOV, or any FFmpeg-supported format
- **Tensor:** `(B, C, H, W)` with values in [0, 255] or [0, 1]
- **Supported sources:** Files, URLs, directories, webcams, RTSP/RTMP streams, YouTube

### Output Tensor Shapes
| Attribute | Shape | Description |
|-----------|-------|-------------|
| `keypoints.xy` | `(N, 17, 2)` | Pixel coordinates [x, y] |
| `keypoints.xyn` | `(N, 17, 2)` | Normalized coordinates [0-1] |
| `keypoints.conf` | `(N, 17)` | Keypoint confidence scores |
| `keypoints.data` | `(N, 17, 3)` | Combined [x, y, conf] |
| `boxes.xyxy` | `(N, 4)` | Bounding boxes in pixels |
| `boxes.conf` | `(N,)` | Detection confidence |

Where `N` = number of detected persons.

---

## Inference Options

Full reference: [Ultralytics Predict Mode](https://docs.ultralytics.com/modes/predict/)

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')

results = model(
    source='image.jpg',       # Image, video, URL, directory, webcam (0)
    conf=0.25,                # Confidence threshold (default: 0.25)
    iou=0.7,                  # NMS IoU threshold (default: 0.7)
    imgsz=640,                # Inference size (default: 640)
    device='cuda',            # 'cuda', 'cpu', or device ID (0, 1, etc.)
    max_det=300,              # Maximum detections per image
    half=False,               # FP16 half-precision inference
    batch=1,                  # Batch size
    vid_stride=1,             # Video frame stride (skip frames)
    stream=False,             # Memory-efficient generator for videos
    verbose=False,            # Suppress output logging

    # Visualization options
    show=False,               # Display results
    save=False,               # Save annotated output
    show_labels=True,         # Show class labels
    show_conf=True,           # Show confidence scores
    line_width=2,             # Bounding box line width
)
```

### Video Streaming (Memory Efficient)

```python
# Use stream=True for long videos to avoid memory issues
for result in model('long_video.mp4', stream=True):
    keypoints = result.keypoints.xy.cpu().numpy()
    # Process each frame
```

---

## Code Examples

### Basic Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11m-pose.pt')  # Download automatically if not present

# Run inference on image
results = model('image.jpg')
result = results[0]

# Access keypoints
keypoints_xy = result.keypoints.xy        # (N, 17, 2) - pixel coords
keypoints_conf = result.keypoints.conf    # (N, 17) - confidence

# Access bounding boxes
boxes_xyxy = result.boxes.xyxy            # (N, 4) - [x1, y1, x2, y2]
boxes_conf = result.boxes.conf            # (N,) - confidence

# Visualize and save
annotated_frame = result.plot()
result.save('output.jpg')
```

### Multi-Person Processing

```python
results = model('group_photo.jpg')
result = results[0]

n_people = len(result.keypoints)  # Number of detected people

for i in range(n_people):
    person_keypoints = result.keypoints.xy[i]   # (17, 2)
    person_conf = result.keypoints.conf[i]      # (17,)
    person_box = result.boxes.xyxy[i]           # (4,)
    detection_conf = result.boxes.conf[i]       # scalar

    print(f"Person {i}: {person_keypoints.shape}, conf={detection_conf:.2f}")
```

### Video Processing Pipeline

```python
from ultralytics import YOLO
import cv2
import numpy as np

def process_video(video_path: str, model_path: str = 'yolo11m-pose.pt'):
    """Process video and extract pose data for each frame."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    all_poses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, verbose=False)
        result = results[0]

        # Extract keypoints for all detected people
        if len(result.keypoints) > 0:
            frame_poses = {
                'keypoints': result.keypoints.xy.cpu().numpy(),      # (N, 17, 2)
                'confidence': result.keypoints.conf.cpu().numpy(),   # (N, 17)
                'boxes': result.boxes.xyxy.cpu().numpy(),            # (N, 4)
                'scores': result.boxes.conf.cpu().numpy(),           # (N,)
            }
            all_poses.append(frame_poses)
        else:
            all_poses.append(None)

    cap.release()
    return all_poses
```

### Integration with Tracking

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')

# Track with ByteTrack (maintains consistent IDs across frames)
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',  # or 'botsort.yaml' for ReID
    stream=True,
    persist=True,              # Maintain tracks across frames
)

for result in results:
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        keypoints = result.keypoints.xy.cpu().numpy()

        for tid, kpts in zip(track_ids, keypoints):
            print(f"Track {tid}: {kpts.shape}")
```

### Pose Normalization (for Action Recognition)

```python
import numpy as np

def normalize_pose(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize pose to be scale and position invariant.

    Args:
        keypoints: (17, 2) array of x, y coordinates

    Returns:
        Normalized (17, 2) array centered on hip midpoint, scaled by torso
    """
    # Hip midpoint (center)
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    hip_center = (left_hip + right_hip) / 2

    # Center on hip
    centered = keypoints - hip_center

    # Scale by torso length (shoulder to hip distance)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    torso_length = np.linalg.norm(shoulder_center - hip_center)

    if torso_length > 0:
        normalized = centered / torso_length
    else:
        normalized = centered

    return normalized
```

---

## Export for Deployment

YOLO11-Pose supports 16+ export formats ([Source](https://docs.ultralytics.com/modes/export/)):

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')

# ONNX (cross-platform)
model.export(format='onnx', dynamic=True)

# TensorRT (NVIDIA GPU - fastest)
model.export(format='engine', device=0, half=True)

# CoreML (iOS/macOS)
model.export(format='coreml', nms=True)

# TFLite (Android/Edge)
model.export(format='tflite')

# OpenVINO (Intel hardware)
model.export(format='openvino')

# NCNN (mobile deployment)
model.export(format='ncnn')
```

**Supported formats:** PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TF SavedModel, TF GraphDef, TF Lite, TF Edge TPU, TF.js, PaddlePaddle, MNN, NCNN, IMX500, RKNN, ExecuTorch

---

## Training

### Dataset Format

YOLO pose datasets use the following annotation format per image:

```
<class-index> <x> <y> <width> <height> <px1> <py1> <pv1> <px2> <py2> <pv2> ... <px17> <py17> <pv17>
```

Where `<pxN> <pyN> <pvN>` are the normalized x, y, and visibility for each keypoint.

### Training Example

```python
from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt')  # Load pretrained model

# Train on custom dataset
results = model.train(
    data='coco-pose.yaml',    # Dataset config
    epochs=100,               # Training epochs
    imgsz=640,                # Image size
    batch=16,                 # Batch size
    device=0,                 # GPU device
    workers=8,                # Data loader workers
    patience=50,              # Early stopping patience
    lr0=0.01,                 # Initial learning rate
    lrf=0.01,                 # Final learning rate (lr0 * lrf)
)

# Validate
metrics = model.val()
print(f"mAP50-95: {metrics.pose.map}")
print(f"mAP50: {metrics.pose.map50}")
```

### COCO-Pose Dataset ([Source](https://docs.ultralytics.com/datasets/pose/coco/))

| Split | Images | Purpose |
|-------|--------|---------|
| train2017 | 56,599 | Model training |
| val2017 | 2,346 | Validation |
| test2017 | 20,288 | Benchmarking |

**Evaluation Metric:** Object Keypoint Similarity (OKS)

---

## Performance Benchmarks

### COCO Keypoints Validation ([Source](https://docs.ultralytics.com/tasks/pose/#models))

| Model | mAP^pose 50-95 | mAP^pose 50 | CPU (ms) | T4 TensorRT (ms) |
|-------|----------------|-------------|----------|------------------|
| yolo11n-pose | 50.0 | 81.0 | 52.4 | 1.7 |
| yolo11s-pose | 58.9 | 86.3 | 90.5 | 2.6 |
| yolo11m-pose | 64.9 | 89.4 | 187.3 | 4.9 |
| yolo11l-pose | 66.1 | 89.9 | 247.7 | 6.4 |
| yolo11x-pose | 69.5 | 91.1 | 488.0 | 12.1 |

### Estimated Real-World Performance

| Device | Model | Input Size | FPS | Latency |
|--------|-------|------------|-----|---------|
| RTX 3080 | yolo11m-pose | 640×640 | ~95 | ~10.5ms |
| RTX 4090 | yolo11m-pose | 640×640 | ~180 | ~5.5ms |
| T4 (cloud) | yolo11m-pose | 640×640 | ~200 | ~5ms |
| CPU (i7) | yolo11m-pose | 640×640 | ~5 | ~187ms |

---

## Common Issues & Solutions

### 1. CUDA Out of Memory

```python
# Reduce image size
results = model('image.jpg', imgsz=480)  # Smaller than default 640

# Use smaller model variant
model = YOLO('yolo11s-pose.pt')

# Enable half-precision
results = model('image.jpg', half=True)

# Process in batches
for frame in frames:
    result = model(frame, verbose=False)
    # Clear CUDA cache periodically
    torch.cuda.empty_cache()
```

### 2. Low Confidence Detections

```python
# Raise confidence threshold
results = model('image.jpg', conf=0.5)  # Default is 0.25

# Filter by keypoint visibility
valid_mask = result.keypoints.conf > 0.5  # (N, 17) boolean mask
```

### 3. Missing Keypoints (Occlusion)

```python
def handle_missing_keypoints(keypoints, conf, threshold=0.3):
    """Replace low-confidence keypoints with NaN or interpolate."""
    mask = conf > threshold
    keypoints_clean = keypoints.copy()
    keypoints_clean[~mask] = np.nan  # Mark as missing
    return keypoints_clean, mask
```

### 4. Slow Inference on CPU

```python
# Use nano variant
model = YOLO('yolo11n-pose.pt')

# Fuse layers for faster inference
model.fuse()

# Export to ONNX for optimized CPU inference
model.export(format='onnx')
```

### 5. Video Memory Leak

```python
# Always use stream=True for videos
for result in model('video.mp4', stream=True):
    # Process each frame
    pass  # Memory is freed after each iteration
```

---

## Combat Sports Application Notes

### Recommended Configuration for OctagonBrain

```python
# Training/analysis (GPU, high accuracy)
model = YOLO('yolo11l-pose.pt')

# Mobile real-time (speed priority)
model = YOLO('yolo11s-pose.pt')

# Inference settings for combat sports
results = model(
    frame,
    conf=0.3,      # Lower threshold to catch fast movements
    iou=0.5,       # Lower IOU for overlapping fighters
    imgsz=640,     # Standard resolution
)
```

### Key Keypoints for Combat Analysis

| Metric | Keypoints |
|--------|-----------|
| Guard position | 9 (left_wrist), 10 (right_wrist) |
| Hip rotation | 11 (left_hip), 12 (right_hip) |
| Stance width | 15 (left_ankle), 16 (right_ankle) |
| Head movement | 0 (nose), 3-4 (ears) |
| Punch extension | 5-6 (shoulders), 7-8 (elbows), 9-10 (wrists) |
| Kick height | 13-14 (knees), 15-16 (ankles) |

---

## References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Pose Estimation Task Guide](https://docs.ultralytics.com/tasks/pose/)
- [COCO-Pose Dataset](https://docs.ultralytics.com/datasets/pose/coco/)
- [Prediction Mode Reference](https://docs.ultralytics.com/modes/predict/)
- [Results API Reference](https://docs.ultralytics.com/reference/engine/results/)
- [Export Formats](https://docs.ultralytics.com/modes/export/)
- [GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Ultralytics Discord Community](https://ultralytics.com/discord)
