# System Architecture

> **Last Updated:** 2025-12-23

---

## High-Level Overview

OctagonBrain is a combat sports analysis platform with two products sharing a common ML core.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                           │
│                     Video / Camera Stream                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED ML CORE (~70-80%)                             │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Pose Estimation │  │ Action          │  │ Metrics Engine              │  │
│  │ ─────────────── │  │ Recognition     │  │ ───────────────             │  │
│  │ • YOLO11-Pose   │  │ ─────────────── │  │ • Center of Mass            │  │
│  │ • BlazePose     │  │ • SkateFormer   │  │ • Joint Angles              │  │
│  │ • ViTPose       │  │ • Rule-based    │  │ • Velocity/Acceleration     │  │
│  │   (Teacher)     │  │   fallback      │  │ • Weight Distribution       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Data Pipeline                                      │  │
│  │  Video → Frames → Pose Detection → Skeleton Normalization → Sequence  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│       OCTAGON COACH           │       │      OCTAGON ANALYTICS        │
│       (Training Mode)         │       │      (Fight Analysis)         │
├───────────────────────────────┤       ├───────────────────────────────┤
│ • Single-person focus         │       │ • Multi-person tracking       │
│ • Real-time mobile inference  │       │ • ByteTrack/BoT-SORT          │
│ • Form analysis & feedback    │       │ • Fighter identification      │
│ • Progress tracking           │       │ • Strike stats per fighter    │
│ • Drill mode                  │       │ • Video upload processing     │
│ • React Native app            │       │ • Web dashboard               │
└───────────────────────────────┘       └───────────────────────────────┘
              │                                           │
              ▼                                           ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│      Mobile App Output        │       │      Web Dashboard Output     │
│ • Real-time feedback overlay  │       │ • Fight statistics            │
│ • Session summaries           │       │ • Round-by-round breakdown    │
│ • Technique counts            │       │ • Fighter comparisons         │
│ • Form scores                 │       │ • Video annotations           │
└───────────────────────────────┘       └───────────────────────────────┘
```

---

## Data Flow

### 1. Input Processing
```
Video File/Stream
    │
    ├─→ Frame Extraction (30 FPS typical)
    │       │
    │       ▼
    ├─→ Preprocessing
    │       • Resize to model input size
    │       • Normalize pixel values
    │       • Convert color space if needed
    │
    ▼
Preprocessed Frames
```

### 2. Pose Estimation Pipeline
```
Frame
    │
    ├─→ Person Detection (YOLO11-detect)
    │       │
    │       ▼
    ├─→ Pose Estimation (YOLO11-pose / BlazePose)
    │       │
    │       ▼
    ├─→ Skeleton Normalization
    │       • Center on hip midpoint
    │       • Scale by torso length
    │       • Handle missing keypoints
    │
    ▼
Normalized Skeleton (17-33 keypoints)
```

### 3. Action Recognition Pipeline
```
Skeleton Sequence (N frames)
    │
    ├─→ Temporal Window (25-64 frames)
    │       │
    │       ▼
    ├─→ Feature Extraction
    │       • Joint velocities
    │       • Bone angles
    │       • Body part trajectories
    │       │
    │       ▼
    ├─→ Action Classification (SkateFormer)
    │       │
    │       ▼
    ├─→ Post-processing
    │       • Temporal smoothing
    │       • Confidence thresholding
    │
    ▼
Action Labels + Timestamps
```

---

## Component Details

### Pose Estimation

| Model | Use Case | Keypoints | Speed | Accuracy |
|-------|----------|-----------|-------|----------|
| YOLO11-Pose-M | Real-time desktop | 17 | 30+ FPS | High |
| YOLO11-Pose-L | Fight analysis | 17 | 15+ FPS | Higher |
| BlazePose | Mobile | 33 | 30+ FPS | Good |
| ViTPose-Huge | Auto-labeling | 17 | 5 FPS | SOTA |

**Keypoint Format (COCO 17):**
```
0: nose           5: left_shoulder   11: left_hip
1: left_eye       6: right_shoulder  12: right_hip
2: right_eye      7: left_elbow      13: left_knee
3: left_ear       8: right_elbow     14: right_knee
4: right_ear      9: left_wrist      15: left_ankle
                 10: right_wrist     16: right_ankle
```

### Action Recognition

**Target Classes (20 total):**

| Category | Classes |
|----------|---------|
| Punches | Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut, Overhand, Body Shot |
| Kicks | Roundhouse (L/R), Front Kick (L/R), Side Kick, Back Kick, Axe Kick, Low Kick |
| Defense | Slip, Bob/Weave, Check, Footwork |

### Metrics Engine

**Static Metrics (per frame):**
- Stance width (feet distance / shoulder width)
- Guard height (wrist position relative to chin)
- Weight distribution (CoM projection)
- Hip/shoulder rotation angles

**Dynamic Metrics (per sequence):**
- Strike velocity (wrist/ankle displacement over time)
- Acceleration profiles
- Movement trajectories
- Technique timing

### Tracking (Fight Analysis)

```
Frame N
    │
    ├─→ Person Detection
    │       │
    │       ▼
    ├─→ ByteTrack Association
    │       • Match detections to existing tracks
    │       • Handle occlusions
    │       • Assign consistent IDs
    │       │
    │       ▼
    ├─→ Fighter Assignment
    │       • Red corner / Blue corner
    │       • Maintain identity through fight
    │
    ▼
Tracked Fighters with IDs
```

---

## Technology Stack

### Backend (Python)
```
Framework:      FastAPI + Uvicorn
ML:             PyTorch
Pose:           Ultralytics (YOLO11), MMPose
Action:         SkateFormer
Video:          OpenCV, FFmpeg
Data:           NumPy, Pandas
Testing:        pytest, mypy
```

### Mobile (React Native)
```
Framework:      React Native
Camera:         react-native-vision-camera
ML iOS:         CoreML
ML Android:     TensorFlow Lite
State:          Zustand or Redux
UI:             React Native Paper
```

### Infrastructure
```
Training:       Lambda Labs / RunPod (cloud GPU)
Inference:      TensorRT (server), CoreML/TFLite (mobile)
Storage:        AWS S3
API Hosting:    AWS EC2 / Fargate
CDN:            CloudFront
Database:       PostgreSQL
Cache:          Redis
```

---

## API Contracts

### Pose Estimation Output
```python
@dataclass
class PoseResult:
    keypoints: np.ndarray      # (N, 17, 3) - x, y, confidence
    boxes: np.ndarray          # (N, 4) - bounding boxes
    scores: np.ndarray         # (N,) - detection confidence
    frame_id: int
    timestamp: float
```

### Action Recognition Output
```python
@dataclass
class ActionResult:
    action: str                # e.g., "jab", "roundhouse"
    confidence: float          # 0.0 - 1.0
    start_frame: int
    end_frame: int
    fighter_id: Optional[int]  # For multi-person
    hand_or_leg: str           # "left" or "right"
```

### Metrics Output
```python
@dataclass
class MetricsResult:
    stance_width: float
    guard_height: float
    weight_distribution: Dict[str, float]  # {"front": 0.6, "back": 0.4}
    center_of_mass: Tuple[float, float, float]
    hip_rotation: float
    shoulder_rotation: float
```

---

## File Structure

```
src/octagon/
├── __init__.py
│
├── models/                    # Model wrappers
│   ├── __init__.py
│   ├── base.py               # Abstract base classes
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── yolo_pose.py      # YOLO11-Pose wrapper
│   │   ├── blazepose.py      # BlazePose wrapper
│   │   └── vitpose.py        # ViTPose wrapper
│   ├── action/
│   │   ├── __init__.py
│   │   ├── skateformer.py    # SkateFormer wrapper
│   │   └── rule_based.py     # Rule-based baseline
│   └── tracking/
│       ├── __init__.py
│       └── byte_track.py     # ByteTrack wrapper
│
├── metrics/                   # Metrics calculation
│   ├── __init__.py
│   ├── static.py             # Single-frame metrics
│   ├── dynamic.py            # Sequence metrics
│   └── biomechanics.py       # CoM, weight distribution
│
├── analysis/                  # High-level analysis
│   ├── __init__.py
│   ├── form_analyzer.py      # Form comparison
│   ├── fight_analyzer.py     # Fight statistics
│   └── feedback.py           # Feedback generation
│
├── pipelines/                 # End-to-end pipelines
│   ├── __init__.py
│   ├── training_mode.py      # Single-person (Coach)
│   └── fight_mode.py         # Two-person (Analytics)
│
├── data/                      # Data handling
│   ├── __init__.py
│   ├── video.py              # Video I/O
│   ├── skeleton.py           # Skeleton utilities
│   └── augmentation.py       # Data augmentation
│
└── api/                       # FastAPI backend
    ├── __init__.py
    ├── main.py               # FastAPI app
    └── routes/
        ├── __init__.py
        ├── inference.py      # Inference endpoints
        └── health.py         # Health check
```

---

## Deployment Architecture

### Mobile (OctagonCoach)
```
┌─────────────────────────────────────────┐
│           Mobile Device                  │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │ Camera  │→ │ On-device│→ │ UI/UX  │ │
│  │         │  │ ML Model │  │        │ │
│  │         │  │(CoreML/  │  │        │ │
│  │         │  │ TFLite)  │  │        │ │
│  └─────────┘  └──────────┘  └────────┘ │
│                    │                     │
│                    ▼                     │
│           ┌──────────────┐              │
│           │ Session Data │──────────────┼──→ Cloud Sync
│           │   (Local)    │              │
│           └──────────────┘              │
└─────────────────────────────────────────┘
```

### Web (OctagonAnalytics)
```
┌──────────────┐        ┌─────────────────────────────────────┐
│    Browser   │  API   │            Cloud                     │
│  ┌────────┐  │───────→│  ┌───────────┐  ┌─────────────────┐ │
│  │ React  │  │        │  │  FastAPI  │→ │ GPU Inference   │ │
│  │  App   │  │←───────│  │  Backend  │  │ (TensorRT)      │ │
│  └────────┘  │        │  └───────────┘  └─────────────────┘ │
└──────────────┘        │        │                │            │
                        │        ▼                ▼            │
                        │  ┌───────────┐  ┌─────────────────┐ │
                        │  │PostgreSQL │  │  S3 (Videos)    │ │
                        │  │ (Results) │  │                 │ │
                        │  └───────────┘  └─────────────────┘ │
                        └─────────────────────────────────────┘
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Mobile inference | >25 FPS | iPhone 12+ |
| Desktop inference | >30 FPS | RTX 3080+ |
| Cloud video processing | <500ms/sec | Per second of video |
| Pose accuracy | >90% PCK@0.5 | On test set |
| Action recognition | >85% mAP | Punches |
| Action recognition | >80% mAP | Kicks |
