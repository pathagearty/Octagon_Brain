# BoxingVI Dataset Documentation

> **Paper:** [arXiv:2511.16524](https://arxiv.org/abs/2511.16524)
> **GitHub:** [Bikudebug/BoxingVI](https://github.com/Bikudebug/BoxingVI)
> **Authors:** Rahul Kumar, Vipul Baghel, Sudhanshu Singh, Bikash Kumar Badatya, Ravi Hegde (IIT Gandhinagar), Shivam Yadav (Dr. A.P.J. Abdul Kalam Technical University), Babji Srinivasan (IIT Madras)
> **Last Updated:** 2025-12-23

---

## Overview

BoxingVI is a multi-modal benchmark for boxing action recognition and temporal localization. It provides real-world boxing data with temporal annotations, per-clip class labels, and 2D pose trajectories - making it ideal for skeleton-based action recognition research.

### Key Features

- **6,915 punch clips** extracted from 20 YouTube sparring videos
- **18 athletes** (11 male, 7 female)
- **6 punch types** with temporal boundaries
- **2D pose data** (AlphaPose, COCO format)
- **Real-world conditions** - monocular RGB, varying lighting/angles

**Use in OctagonBrain:** Primary training dataset for boxing action recognition in both OctagonCoach and OctagonAnalytics.

---

## Dataset Statistics

| Split | Clips | Subjects |
|-------|-------|----------|
| Training | 5,513 | S1-S15 |
| Validation | 1,402 | S16-S20 |
| **Total** | **6,915** | **18** |

---

## Action Classes (6 Punch Types)

| Class | Description |
|-------|-------------|
| **Jab** | Quick straight punch with lead hand |
| **Cross** | Powerful straight punch with rear hand |
| **Lead Hook** | Curved punch with lead hand |
| **Rear Hook** | Curved punch with rear hand |
| **Lead Uppercut** | Upward punch with lead hand |
| **Rear Uppercut** | Upward punch with rear hand |

---

## Data Organization

```
BoxingVI/
├── Annotation_files/          # Excel files with temporal boundaries
│   └── *.xlsx                 # (start_frame, end_frame, punch_class)
├── RGB_videos/                # Raw video files
│   └── *.mp4                  # 20 YouTube sparring videos
└── Skeleton_data/             # Pre-extracted 2D pose
    └── *.npy                  # AlphaPose keypoints (COCO format)
```

---

## Annotation Format

Annotations are stored in Excel (.xlsx) files with the following structure:

| Column | Description |
|--------|-------------|
| `start_frame` | First frame of punch |
| `end_frame` | Last frame of punch |
| `punch_class` | One of 6 punch types |

Each row represents a single punch clip with temporal boundaries.

---

## Pose Data Format

Pose estimations are extracted using **AlphaPose** in **COCO format** (17 keypoints).

### Pre-processing Applied

1. **Scaling:** Keypoints scaled to video dimensions
2. **Padding:** Sequences padded to maximum 25 frames (at 30 fps)
3. **Tracking:** Center-of-mass following to identify primary boxer

### COCO 17 Keypoints

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

---

## Comparison with Other Datasets

| Dataset | Videos | Clips | 2D Pose | Temporal Annotations | Public |
|---------|--------|-------|---------|---------------------|--------|
| **BoxingVI** | 20 | 6,915 | Yes | Yes | Yes |
| 3DCG Boxing | - | 6,900 | No | No | ? |
| BoxMAC | 15 | 2,314 | No | No | Withdrawn |
| MDPI Boxing | - | 312K frames | No | Punch/no-punch only | Yes |

**BoxingVI Advantages:**
- Only dataset with RGB videos + temporal segmentation + 2D pose
- Real-world data (not synthetic)
- Per-clip class labels for supervised learning

---

## Usage Examples

### Loading Annotations

```python
import pandas as pd
import numpy as np

# Load annotations
annotations = pd.read_excel('Annotation_files/video_01.xlsx')

# Each row is a punch clip
for idx, row in annotations.iterrows():
    start_frame = row['start_frame']
    end_frame = row['end_frame']
    punch_class = row['punch_class']

    print(f"Punch {idx}: frames {start_frame}-{end_frame}, class: {punch_class}")
```

### Loading Skeleton Data

```python
import numpy as np

# Load pre-extracted pose data
skeleton_data = np.load('Skeleton_data/video_01.npy')

# Shape: (num_clips, max_frames=25, num_keypoints=17, 2)
print(f"Skeleton shape: {skeleton_data.shape}")

# Extract single clip
clip_skeleton = skeleton_data[0]  # (25, 17, 2)
```

### Preparing for SkateFormer

```python
import numpy as np

def prepare_for_skateformer(skeleton_data: np.ndarray) -> np.ndarray:
    """Convert BoxingVI skeleton to SkateFormer format.

    BoxingVI format: (N, T, V, C) = (clips, 25, 17, 2)
    SkateFormer format: (N, C, T, V, M) = (clips, 3, T, V, 1)

    Args:
        skeleton_data: BoxingVI skeleton array

    Returns:
        SkateFormer-compatible array
    """
    N, T, V, C = skeleton_data.shape

    # Add z-coordinate as zeros (2D -> 3D)
    z_coords = np.zeros((N, T, V, 1))
    skeleton_3d = np.concatenate([skeleton_data, z_coords], axis=-1)  # (N, T, V, 3)

    # Transpose to (N, C, T, V)
    skeleton_3d = skeleton_3d.transpose(0, 3, 1, 2)  # (N, 3, T, V)

    # Add M dimension (single person)
    skeleton_3d = skeleton_3d[:, :, :, :, np.newaxis]  # (N, 3, T, V, 1)

    return skeleton_3d

# Usage
skateformer_input = prepare_for_skateformer(skeleton_data)
print(f"SkateFormer input shape: {skateformer_input.shape}")  # (N, 3, 25, 17, 1)
```

### COCO to NTU Keypoint Mapping

If using SkateFormer pre-trained on NTU RGB+D (25 keypoints), map COCO 17 to NTU:

```python
# COCO 17 to NTU 25 mapping (approximate)
COCO_TO_NTU = {
    0: 3,    # nose -> head
    5: 4,    # left_shoulder
    6: 8,    # right_shoulder
    7: 5,    # left_elbow
    8: 9,    # right_elbow
    9: 6,    # left_wrist
    10: 10,  # right_wrist
    11: 12,  # left_hip
    12: 16,  # right_hip
    13: 13,  # left_knee
    14: 17,  # right_knee
    15: 14,  # left_ankle
    16: 18,  # right_ankle
}

def map_coco_to_ntu(coco_skeleton: np.ndarray) -> np.ndarray:
    """Map COCO 17 keypoints to NTU 25 format.

    Args:
        coco_skeleton: (T, 17, C) array

    Returns:
        (T, 25, C) array with unmapped joints as zeros
    """
    T, _, C = coco_skeleton.shape
    ntu_skeleton = np.zeros((T, 25, C))

    for coco_idx, ntu_idx in COCO_TO_NTU.items():
        ntu_skeleton[:, ntu_idx, :] = coco_skeleton[:, coco_idx, :]

    return ntu_skeleton
```

---

## Training Configuration

### Class Weights (for imbalanced data)

```python
# If classes are imbalanced, compute weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
```

### Data Augmentation

```python
import random

def augment_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Apply augmentations to skeleton sequence.

    Args:
        skeleton: (T, V, C) array

    Returns:
        Augmented skeleton
    """
    # Random rotation (2D)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        skeleton = skeleton @ rotation.T

    # Random scaling
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        skeleton = skeleton * scale

    # Random horizontal flip (mirror)
    if random.random() > 0.5:
        skeleton[:, :, 0] = -skeleton[:, :, 0]
        # Swap left/right keypoints
        skeleton = swap_left_right(skeleton)

    return skeleton
```

---

## Download & Access

**Current Status:** Dataset download link will be provided after publication.

**Repository:** https://github.com/Bikudebug/BoxingVI

Contains:
- Video links (YouTube)
- Temporal annotations
- Punch category labels
- Pre-extracted skeleton data

---

## Integration with OctagonBrain

### Directory Structure

```
data/
└── boxingvi/
    ├── raw/
    │   ├── videos/           # Downloaded videos
    │   └── annotations/      # Excel files
    ├── processed/
    │   ├── skeletons/        # Converted to our format
    │   └── clips/            # Extracted video clips
    └── splits/
        ├── train.txt         # Training file list
        └── val.txt           # Validation file list
```

### Processing Pipeline

```python
# scripts/process_boxingvi.py

def process_boxingvi_dataset(raw_dir: str, output_dir: str):
    """Process BoxingVI dataset for OctagonBrain training.

    1. Load annotations
    2. Extract video clips
    3. Convert skeletons to standard format
    4. Create train/val splits
    """
    # Implementation details...
    pass
```

---

## Common Issues

### 1. Video Download Issues

```python
# Use yt-dlp to download videos
# pip install yt-dlp

import subprocess

def download_video(url: str, output_path: str):
    subprocess.run([
        'yt-dlp',
        '-f', 'best[height<=720]',  # Limit resolution
        '-o', output_path,
        url
    ])
```

### 2. Frame Rate Mismatch

```python
# BoxingVI uses 30 fps
# Ensure consistent frame rate when extracting

import cv2

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps != 30:
    print(f"Warning: Video FPS is {fps}, expected 30")
```

### 3. Missing Keypoints

```python
# Handle missing/low-confidence keypoints
def interpolate_missing(skeleton: np.ndarray, confidence_threshold: float = 0.3):
    """Interpolate missing keypoints."""
    # Implementation...
    pass
```

---

## References

- [BoxingVI Paper (arXiv)](https://arxiv.org/abs/2511.16524)
- [BoxingVI GitHub Repository](https://github.com/Bikudebug/BoxingVI)
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) - Pose extraction used
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- [SkateFormer Documentation](skateformer.md) - Action recognition model
