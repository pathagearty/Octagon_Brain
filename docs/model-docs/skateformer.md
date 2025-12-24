# SkateFormer Documentation

> **Source:** [KAIST VIC Lab Project Page](https://kaist-viclab.github.io/SkateFormer_site/)
> **GitHub:** [KAIST-VICLab/SkateFormer](https://github.com/KAIST-VICLab/SkateFormer)
> **Paper:** [arXiv:2403.09508](https://arxiv.org/abs/2403.09508)
> **Published:** ECCV 2024, pages 401-420
> **Last Updated:** 2025-12-23

---

## Overview

SkateFormer is a **Skeletal-Temporal Transformer** for skeleton-based human action recognition, published at ECCV 2024. It introduces a partition-specific attention strategy (Skate-MSA) that categorizes skeletal-temporal relations into four distinct types to efficiently capture meaningful patterns.

### Key Innovations ([Source](https://arxiv.org/abs/2403.09508))

- **Skate-Type Partitions:** Categorizes joint-frame relations into 4 types based on skeletal and temporal proximity
- **Skate-MSA:** Multi-scale skeletal-temporal self-attention within partitioned regions
- **Skate-Embedding:** Novel positional encoding combining skeletal and temporal features via outer product
- **Computational Efficiency:** 2.03M parameters, 3.62G FLOPs (vs. 2.71M/9.64G for Hyperformer)
- **State-of-the-art:** Outperforms previous methods on NTU RGB+D 60, 120, and NW-UCLA

**Authors:** Jeonghyeok Do and Munchurl Kim (KAIST, South Korea)

**Use in OctagonBrain:** Primary model for classifying punches, kicks, and defensive movements from skeleton sequences.

---

## Installation

```bash
# Clone repository
git clone https://github.com/KAIST-VICLab/SkateFormer.git
cd SkateFormer

# Create conda environment (recommended)
conda env create -f requirements.yaml
conda activate skateformer

# Install torchlight (required)
pip install -e torchlight
```

**Requirements:** ([Source](https://github.com/KAIST-VICLab/SkateFormer))
- Python >= 3.9.16
- PyTorch >= 1.12.1
- CUDA 11.6
- Ubuntu 22.04 (tested platform)

**License:** MIT (source code) / Research-only for pretrained weights. Commercial use requires permission from Prof. Munchurl Kim (mkimee@kaist.ac.kr).

---

## Pretrained Models

Download pretrained weights from [Google Drive](https://drive.google.com/file/d/16dBg4nq91dUYqVqB4W0d8r4TzOMe0U2u/view?usp=sharing).

Models are available for:
- NTU RGB+D 60 (X-Sub, X-View)
- NTU RGB+D 120 (X-Sub, X-Set)
- NTU-Inter (X-Sub, X-View)
- NTU-Inter 120 (X-Sub, X-Set)
- NW-UCLA

---

## Key Concepts

### Skeletal-Temporal Relation Types (Skate-Types) ([Source](https://arxiv.org/html/2403.09508v1))

SkateFormer categorizes joint-frame relations into 4 types:

| Type | Skeletal Relation | Temporal Relation | Description |
|------|-------------------|-------------------|-------------|
| Type 1 | Neighboring | Neighboring | Adjacent joints in adjacent frames (local motion) |
| Type 2 | Neighboring | Distant | Adjacent joints in distant frames (global motion patterns) |
| Type 3 | Distant | Neighboring | Distant joints in adjacent frames (cross-body coordination) |
| Type 4 | Distant | Distant | Distant joints in distant frames (long-range dependencies) |

### Partition Strategy

Instead of computing attention across all joints×frames (expensive), SkateFormer:
1. Partitions joints based on physical proximity (neighboring vs distant)
2. Partitions frames based on temporal distance (neighboring vs distant)
3. Computes efficient attention within each partition type
4. Enables selective focus on key joints and frames in an action-adaptive manner

---

## Input/Output Format

### Input Tensor Shape ([Source](https://arxiv.org/html/2403.09508v1))

```python
# Expected input shape
x = torch.randn(B, C, T, V, M)

# Where:
# B = batch size
# C = input channels (3 for x, y, z coordinates)
# T = temporal frames (64 for NTU datasets)
# V = number of joints (25 for NTU, 20 for NW-UCLA)
# M = number of persons (2 for NTU, 1 for single-person)
```

### Internal Dimensions

| Dataset | V (joints) | T (frames) | C_in | C (embed) |
|---------|------------|------------|------|-----------|
| NTU RGB+D | 48* | 64 | 3 | 96 |
| NW-UCLA | 20 | 64 | 3 | 96 |

*Note: NTU uses 25 joints × 2 persons = 50 → processed as 48 after padding/selection.

### Modalities

SkateFormer supports 4 input modalities:

| Modality | Code | Description |
|----------|------|-------------|
| Joint (J) | `j` | Raw 3D joint coordinates |
| Bone (B) | `b` | Bone vectors between connected joints |
| Joint Motion (JM) | `jm` | Temporal difference of joint positions |
| Bone Motion (BM) | `bm` | Temporal difference of bone vectors |

### Output

```python
output = model(x)  # (B, num_classes)
pred_class = output.argmax(dim=1)
```

---

## Model Architecture ([Source](https://arxiv.org/html/2403.09508v1))

```
Input: (B, 3, T, V, M)
    │
    ▼
Skate-Embedding
    │ - Skeletal features (learnable per joint)
    │ - Temporal features (fixed positional encoding)
    │ - Combined via outer product
    │
    ▼
SkateFormer Blocks (×8)
    │
    ├── Skate-MSA (partition-specific attention)
    │   ├── Type 1: neighboring × neighboring
    │   ├── Type 2: neighboring × distant
    │   ├── Type 3: distant × neighboring
    │   └── Type 4: distant × distant
    │
    ├── T-Conv (temporal convolution, k=7)
    │
    ├── FFN (feed-forward network, expansion=4)
    │
    └── LayerNorm + Residual
    │
    │ [Temporal downsampling after every 2 blocks]
    │
    ▼
Global Average Pooling
    │
    ▼
Classification Head (Linear)
    │
    ▼
Output: (B, num_classes)
```

### Architecture Hyperparameters ([Source](https://arxiv.org/html/2403.09508v1))

| Parameter | Value |
|-----------|-------|
| SkateFormer Blocks (R) | 8 |
| Attention Heads (H) | 32 |
| T-Conv Kernel Size (k) | 7 |
| FFN Expansion Ratio | 4 (NTU), 1 (NW-UCLA) |
| Temporal Downsampling | After every 2 blocks |

---

## Performance Benchmarks

### NTU RGB+D 60 ([Source](https://arxiv.org/html/2403.09508v1))

| Method | Ensemble | X-Sub (%) | X-View (%) |
|--------|----------|-----------|------------|
| ST-GCN | - | 81.5 | 88.3 |
| CTR-GCN | E4 | 92.4 | 96.8 |
| InfoGCN | E4 | 93.0 | 97.1 |
| HD-GCN | E4 | 93.4 | 97.2 |
| **SkateFormer** | E1 | **92.6** | **97.0** |
| **SkateFormer** | E2 | **93.0** | **97.4** |
| **SkateFormer** | E4 | **93.5** | **97.8** |

### NTU RGB+D 120 ([Source](https://arxiv.org/html/2403.09508v1))

| Method | Ensemble | X-Sub (%) | X-Set (%) |
|--------|----------|-----------|-----------|
| CTR-GCN | E4 | 88.9 | 90.6 |
| InfoGCN | E4 | 89.8 | 91.2 |
| HD-GCN | E4 | 89.8 | 91.2 |
| **SkateFormer** | E1 | **87.7** | **89.3** |
| **SkateFormer** | E2 | **89.4** | **89.8** |
| **SkateFormer** | E4 | **89.8** | **91.2** |

### NW-UCLA ([Source](https://arxiv.org/html/2403.09508v1))

| Method | Accuracy (%) |
|--------|--------------|
| CTR-GCN | 96.5 |
| InfoGCN | 97.0 |
| HD-GCN | 97.2 |
| **SkateFormer** | **98.3** |

### Computational Efficiency ([Source](https://arxiv.org/html/2403.09508v1))

| Model | Params (M) | FLOPs (G) | Inference (ms) |
|-------|------------|-----------|----------------|
| Hyperformer | 2.71 | 9.64 | 18.07 |
| **SkateFormer** | **2.03** | **3.62** | **11.46** |
| Improvement | 25% fewer | 62% fewer | 37% faster |

### Per-Modality Results (NTU RGB+D 60 X-Sub) ([Source](https://arxiv.org/html/2403.09508v1))

| Modality | Accuracy (%) |
|----------|--------------|
| Joint (J) | 92.6 |
| Bone (B) | 92.1 |
| Joint Motion (JM) | 89.8 |
| Bone Motion (BM) | 89.0 |

---

## NTU RGB+D 25 Joint Format

For NTU RGB+D datasets, the skeleton has 25 joints per person:

```
Index  Joint                Index  Joint
-----  -----                -----  -----
0      base of spine        13     left wrist
1      middle of spine      14     right wrist
2      neck                 15     left hand
3      head                 16     right hand
4      left shoulder        17     left hip
5      left elbow           18     right hip
6      left wrist           19     left knee
7      left hand            20     right knee
8      right shoulder       21     left ankle
9      right elbow          22     right ankle
10     right wrist          23     left foot
11     right hand           24     right foot
12     left hip
```

---

## Code Examples

### Loading Pretrained Model

```python
import torch
from model import SkateFormer

# Initialize model for NTU RGB+D 60
model = SkateFormer(
    num_class=60,           # NTU RGB+D 60 classes
    num_point=25,           # 25 joints per person
    num_person=2,           # Max 2 persons
    in_channels=3,          # x, y, z coordinates
    num_frames=64,          # Temporal window
)

# Load pretrained weights
checkpoint = torch.load('checkpoints/ntu60_xsub_joint.pt')
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()
```

### Inference

```python
# Prepare input: (batch, channels, frames, joints, persons)
skeleton_sequence = torch.randn(1, 3, 64, 25, 2).cuda()

# Run inference
with torch.no_grad():
    output = model(skeleton_sequence)  # (1, 60)
    pred_class = output.argmax(dim=1)  # (1,)
    confidence = torch.softmax(output, dim=1).max()

print(f"Predicted class: {pred_class.item()}, Confidence: {confidence:.3f}")
```

### Training

```bash
# NTU RGB+D 60 X-Sub (joint modality)
python main.py --config ./config/train/ntu_cs/SkateFormer_j.yaml

# NTU RGB+D 60 X-Sub (bone modality)
python main.py --config ./config/train/ntu_cs/SkateFormer_b.yaml

# NTU RGB+D 120 X-Sub
python main.py --config ./config/train/ntu120_csub/SkateFormer_j.yaml

# NW-UCLA
python main.py --config ./config/train/nw_ucla/SkateFormer_j.yaml
```

### Testing

```bash
# NTU RGB+D 60 X-View
python main.py --config ./config/test/ntu_cv/SkateFormer_j.yaml

# NTU RGB+D 120 X-Sub
python main.py --config ./config/test/ntu120_csub/SkateFormer_j.yaml

# NW-UCLA
python main.py --config ./config/test/nw_ucla/SkateFormer_b.yaml
```

---

## Training Configuration

### Example YAML Config

```yaml
# config/train/ntu_cs/SkateFormer_j.yaml
model:
  num_class: 60            # NTU RGB+D 60 classes
  num_point: 25            # Joints per skeleton
  num_person: 2            # Max persons
  in_channels: 3           # x, y, z
  num_frames: 64           # Temporal length

training:
  batch_size: 64
  epochs: 100
  base_lr: 0.1
  weight_decay: 0.0004
  optimizer: SGD
  momentum: 0.9
  nesterov: True

scheduler:
  type: MultiStepLR
  milestones: [35, 55]
  gamma: 0.1
  warmup_epoch: 5
```

### Training Tips

1. **Multi-modality ensemble:** Train separate models for J, B, JM, BM and ensemble predictions
2. **Learning rate:** Start with 0.1 for SGD, use warmup for 5 epochs
3. **Data augmentation:** Random rotation, random crop, random flip
4. **Temporal sampling:** Uniform sampling to 64 frames

---

## Data Augmentation

```python
def augment_skeleton(skeleton: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Augment skeleton sequence.

    Args:
        skeleton: (C, T, V, M) tensor
        p: probability of each augmentation
    """
    C, T, V, M = skeleton.shape

    # 1. Random rotation around y-axis (vertical)
    if random.random() < p:
        angle = random.uniform(-15, 15) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=skeleton.dtype)
        skeleton = torch.einsum('ij,jTVM->iTVM', rotation, skeleton)

    # 2. Random scale
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        skeleton = skeleton * scale

    # 3. Random temporal crop
    if random.random() < p:
        crop_len = random.randint(int(T * 0.8), T)
        start = random.randint(0, T - crop_len)
        skeleton = skeleton[:, start:start+crop_len]
        skeleton = F.interpolate(skeleton.unsqueeze(0), size=(T, V, M)).squeeze(0)

    # 4. Random joint dropout
    if random.random() < p:
        drop_mask = torch.rand(V) > 0.1
        skeleton[:, :, ~drop_mask, :] = 0

    # 5. Horizontal flip (swap left/right joints)
    if random.random() < p:
        skeleton = mirror_skeleton_ntu(skeleton)

    return skeleton
```

---

## COCO to NTU Keypoint Mapping

For using COCO 17-keypoint poses with SkateFormer (NTU 25 format):

```python
# COCO 17 -> NTU 25 mapping (approximate)
COCO_TO_NTU = {
    0: 3,    # nose -> head
    5: 4,    # left_shoulder -> left_shoulder
    6: 8,    # right_shoulder -> right_shoulder
    7: 5,    # left_elbow -> left_elbow
    8: 9,    # right_elbow -> right_elbow
    9: 6,    # left_wrist -> left_wrist
    10: 10,  # right_wrist -> right_wrist
    11: 12,  # left_hip -> left_hip
    12: 17,  # right_hip -> right_hip
    13: 13,  # left_knee -> left_knee
    14: 18,  # right_knee -> right_knee
    15: 14,  # left_ankle -> left_ankle
    16: 19,  # right_ankle -> right_ankle
}

def coco_to_ntu(coco_skeleton: np.ndarray, T: int = 64) -> torch.Tensor:
    """
    Convert COCO 17 keypoints to NTU 25 format.

    Args:
        coco_skeleton: (T, 17, 3) array of x, y, z coordinates
        T: target temporal length

    Returns:
        NTU skeleton: (3, T, 25, 1) tensor
    """
    ntu_skeleton = np.zeros((T, 25, 3))

    for coco_idx, ntu_idx in COCO_TO_NTU.items():
        ntu_skeleton[:, ntu_idx, :] = coco_skeleton[:, coco_idx, :]

    # Estimate missing joints (spine, hands, feet)
    # Base of spine: midpoint of hips
    ntu_skeleton[:, 0, :] = (ntu_skeleton[:, 12, :] + ntu_skeleton[:, 17, :]) / 2
    # Middle of spine: between base and neck
    ntu_skeleton[:, 1, :] = (ntu_skeleton[:, 0, :] + ntu_skeleton[:, 2, :]) / 2
    # Neck: midpoint of shoulders
    ntu_skeleton[:, 2, :] = (ntu_skeleton[:, 4, :] + ntu_skeleton[:, 8, :]) / 2

    # Convert to (C, T, V, M) format
    ntu_tensor = torch.from_numpy(ntu_skeleton).permute(2, 0, 1).unsqueeze(-1)
    return ntu_tensor.float()
```

---

## Combat Sports Adaptation

### Custom Action Classes for OctagonBrain

```python
COMBAT_CLASSES = {
    # Punches
    0: 'jab',
    1: 'cross',
    2: 'lead_hook',
    3: 'rear_hook',
    4: 'lead_uppercut',
    5: 'rear_uppercut',
    6: 'overhand',
    7: 'body_shot',

    # Kicks
    8: 'lead_roundhouse',
    9: 'rear_roundhouse',
    10: 'lead_teep',
    11: 'rear_teep',
    12: 'side_kick',
    13: 'back_kick',
    14: 'axe_kick',
    15: 'low_kick',

    # Defense
    16: 'slip',
    17: 'bob_weave',
    18: 'check',
    19: 'stance_change',
}
```

### Fine-tuning Strategy

```python
# 1. Load NTU-pretrained weights
model = SkateFormer(num_class=60, num_point=25, num_person=2)
checkpoint = torch.load('skateformer_ntu60_xsub.pth')
model.load_state_dict(checkpoint['model'])

# 2. Replace classification head for combat sports
model.fc = torch.nn.Linear(model.fc.in_features, 20)  # 20 combat classes

# 3. Freeze backbone initially
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# 4. Train head only (5-10 epochs)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# ... training loop ...

# 5. Unfreeze and fine-tune entire model (lower LR)
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# ... continue training ...
```

---

## Common Issues & Solutions

### 1. Input Shape Mismatch

```python
# Error: Expected 25 joints, got 17
# Solution: Map COCO keypoints to NTU format
ntu_skeleton = coco_to_ntu(coco_skeleton)
```

### 2. Temporal Length Mismatch

```python
def adjust_temporal_length(skeleton: torch.Tensor, target_length: int = 64) -> torch.Tensor:
    """Pad or sample skeleton to fixed temporal length."""
    C, T, V, M = skeleton.shape

    if T < target_length:
        # Pad with last frame
        padding = skeleton[:, -1:].repeat(1, target_length - T, 1, 1)
        return torch.cat([skeleton, padding], dim=1)
    elif T > target_length:
        # Uniform temporal sampling
        indices = torch.linspace(0, T - 1, target_length).long()
        return skeleton[:, indices]
    return skeleton
```

### 3. Memory Issues

```python
# Reduce batch size
batch_size = 32  # Default is 64

# Use gradient checkpointing
model.use_checkpoint = True

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(skeleton)
    loss = criterion(output, label)
```

### 4. Single Person Input

```python
# If only 1 person, pad to 2-person format
if skeleton.shape[-1] == 1:
    skeleton = torch.cat([skeleton, torch.zeros_like(skeleton)], dim=-1)
```

---

## Ensemble Strategies

### E2 Ensemble (Joint + Bone)

```python
def ensemble_e2(model_j, model_b, skeleton):
    """Two-modality ensemble (Joint + Bone)."""
    # Joint prediction
    logits_j = model_j(skeleton)

    # Compute bone representation
    skeleton_bone = compute_bone(skeleton)
    logits_b = model_b(skeleton_bone)

    # Average logits
    logits_ensemble = (logits_j + logits_b) / 2
    return logits_ensemble.argmax(dim=1)
```

### E4 Ensemble (All Modalities)

```python
def ensemble_e4(models, skeleton):
    """Four-modality ensemble (J, B, JM, BM)."""
    skeleton_j = skeleton
    skeleton_b = compute_bone(skeleton)
    skeleton_jm = compute_motion(skeleton)
    skeleton_bm = compute_motion(skeleton_b)

    logits = []
    for modality, data in zip(['j', 'b', 'jm', 'bm'],
                               [skeleton_j, skeleton_b, skeleton_jm, skeleton_bm]):
        logits.append(models[modality](data))

    # Weighted average (can tune weights)
    logits_ensemble = sum(logits) / len(logits)
    return logits_ensemble.argmax(dim=1)
```

---

## References

- [SkateFormer Paper (arXiv)](https://arxiv.org/abs/2403.09508)
- [ECCV 2024 Paper](https://link.springer.com/chapter/10.1007/978-3-031-72940-9_23)
- [Official GitHub Repository](https://github.com/KAIST-VICLab/SkateFormer)
- [Project Page](https://kaist-viclab.github.io/SkateFormer_site/)
- [Pretrained Weights (Google Drive)](https://drive.google.com/file/d/16dBg4nq91dUYqVqB4W0d8r4TzOMe0U2u/view?usp=sharing)
- [NTU RGB+D Dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [Papers with Code Leaderboard](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd)
