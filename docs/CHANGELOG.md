# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Metrics engine (CoM, angles, velocity)
- Multi-person tracking (ByteTrack)
- OctagonCoach mobile app
- API backend (FastAPI)

---

## [0.2.0] - 2025-12-23

### Added

#### Core ML Pipeline (Phase 1 Complete)
- **src/octagon/models/pose/base.py** - PoseResult dataclass, BasePoseEstimator ABC, COCO 17 keypoint constants
- **src/octagon/models/pose/yolo_pose.py** - YOLO11Pose wrapper with detect, detect_batch, detect_video, detect_stream
- **src/octagon/models/action/base.py** - ActionResult dataclass, BoxingAction enum (6 classes), BaseActionRecognizer ABC
- **src/octagon/models/action/skateformer.py** - SkateFormer model adapted for COCO 17 keypoints (444K params), SkateFormerWrapper for inference

#### Data Pipeline
- **src/octagon/data/skeleton.py** - Skeleton utilities:
  - SkeletonSequence dataclass
  - normalize_skeleton() - Center on hips, scale by torso
  - interpolate_temporal() - Linear/nearest interpolation
  - create_skeleton_windows() - Sliding window extraction
  - augment_skeleton() - Rotation, scale, flip, noise
  - skeleton_to_tensor() - Convert to model input format
  - compute_bone_features() - Bone vectors between joints
  - compute_motion_features() - Temporal velocity

- **src/octagon/data/datasets/base.py** - BaseSkeletonDataset ABC, SkeletonSample dataclass
- **src/octagon/data/datasets/boxingvi.py** - BoxingVIDataset for 6-class boxing punch recognition:
  - Train split: Subjects S1-S15 (5,513 clips)
  - Val split: Subjects S16-S20 (1,402 clips)
  - Automatic normalization and augmentation
  - Class weight computation for imbalanced data

#### Training Infrastructure
- **scripts/train_action.py** - Full training script with:
  - YAML config loading
  - Mixed precision (AMP) for memory efficiency
  - Gradient accumulation (effective batch 32 from batch 8)
  - TensorBoard logging
  - Checkpoint saving (best + latest)
  - Early stopping support
  - Class-weighted loss for imbalanced data

- **configs/training/skateformer_boxing.yaml** - Training config optimized for GTX 1650 (4GB VRAM)

#### Unit Tests (70 tests)
- **tests/models/test_pose.py** - PoseResult, COCO constants (10 tests)
- **tests/models/test_action.py** - ActionResult, BoxingAction, SkateFormer (16 tests)
- **tests/data/test_skeleton.py** - Skeleton utilities (29 tests)
- **tests/data/test_boxingvi.py** - Dataset loader (15 tests)

#### Documentation
- **docs/PHASE1_IMPLEMENTATION.md** - Comprehensive implementation tracker for agent handoff
- Updated PROJECT_STATUS.md with Phase 1 completion

### Changed
- Added pandas>=2.0.0 and openpyxl>=3.1.0 to pyproject.toml for Excel annotation loading

---

## [0.1.1] - 2025-12-23

### Added

#### Model Documentation (docs/model-docs/)
- **yolo11-pose.md** - Comprehensive YOLO11-Pose documentation with:
  - All 5 model variants with benchmarks (nano to x-large)
  - Complete COCO 17 keypoint format reference
  - API reference for Results, Boxes, and Keypoints classes
  - Inference options and video streaming examples
  - Export formats for deployment (ONNX, TensorRT, CoreML, TFLite)
  - Training configuration and dataset format
  - Combat sports-specific usage examples

- **skateformer.md** - SkateFormer action recognition documentation with:
  - ECCV 2024 paper details and benchmarks
  - Skate-Type partition strategy explanation
  - NTU RGB+D 60/120 performance (92.6%/87.7% X-Sub)
  - Input format (B, C, T, V, M) and modality support (J, B, JM, BM)
  - COCO to NTU keypoint mapping code
  - Fine-tuning strategy for combat sports
  - Ensemble strategies (E1, E2, E4)

- **blazepose.md** - MediaPipe BlazePose documentation with:
  - Complete 33 landmark reference table
  - 3 model variants (Lite, Full, Heavy) with latency
  - BlazePose to COCO keypoint mapping
  - Mobile integration (React Native, iOS, Android)
  - Combat sports metrics (guard height, stance width)
  - Performance benchmarks (20-53ms on mobile)

- **bytetrack.md** - ByteTrack tracker documentation with:
  - Two-stage association algorithm explanation
  - Complete bytetrack.yaml configuration reference
  - Ultralytics YOLO integration examples
  - Fighter assignment and ID switch handling code
  - ByteTrack vs BoT-SORT comparison
  - Combat sports-specific configuration
  - MOT17/MOT20 benchmark results

- **mediapipe.md** - MediaPipe framework documentation with:
  - All available solutions (Vision, Text, Audio)
  - 21 hand landmark reference table
  - Holistic solution (540+ landmarks)
  - Mobile SDK integration (Android, iOS, React Native)
  - Model download locations and deployment options
  - Performance optimization techniques

### Documentation Quality
- All docs include source links for verification
- Working code examples that can be copy-pasted
- Exact tensor shapes and data types
- Combat sports-specific sections for OctagonBrain
- Cross-references between related models

---

## [0.1.0] - 2025-12-23

### Added

#### Documentation
- Project README with overview and quick start
- CLAUDE.md - AI agent contribution guide
- docs/PROJECT_STATUS.md - Progress tracking
- docs/ARCHITECTURE.md - System design documentation
- docs/DECISIONS.md - Architecture Decision Records
- docs/CHANGELOG.md - This changelog
- docs/RESEARCH.md - Technology research and analysis

#### Repository Structure
- Created directory structure for source code (src/octagon/)
- Created docs/ directory with subdirectories:
  - docs/components/ - Per-component documentation
  - docs/guides/ - How-to guides
  - docs/references/ - External references
  - docs/model-docs/ - Local model documentation
- Created mobile/ directory for React Native app
- Created scripts/ directory for utility scripts
- Created notebooks/ directory for Jupyter experiments
- Created tests/ directory for test suite
- Created configs/ directory for configuration files
- Created data/ and weights/ directories (gitignored)

#### Configuration
- .gitignore for Python, ML, mobile, and IDE files

### Research Completed
- Evaluated pose estimation models (YOLO11, DETRPose, ViTPose, BlazePose, RTMPose)
- Evaluated action recognition models (SkateFormer, VideoMAEv2, ST-GCN)
- Evaluated tracking solutions (ByteTrack, BoT-SORT)
- Identified key datasets (BoxingVI, Olympic Boxing, NTU RGB+D)
- Defined technology stack and architecture
- Created 36-week development timeline

### Decisions Made
- ADR-001: YOLO11-Pose for real-time pose estimation
- ADR-002: SkateFormer for action recognition
- ADR-003: Teacher-Student architecture for data efficiency
- ADR-004: React Native for cross-platform mobile
- ADR-005: Shared ML core between OctagonCoach and OctagonAnalytics
- ADR-006: FastAPI for backend API
- ADR-007: ByteTrack for multi-person tracking
- ADR-008: Full striking scope (boxing + kicks) for MVP

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.2.0 | 2025-12-23 | Phase 1 Complete: Core ML pipeline (pose, skeleton, dataset, SkateFormer, training) |
| 0.1.1 | 2025-12-23 | Comprehensive model documentation (YOLO11-Pose, SkateFormer, BlazePose, ByteTrack, MediaPipe) |
| 0.1.0 | 2025-12-23 | Initial repository setup and documentation |

---

## How to Update This Changelog

When completing work, add an entry under `[Unreleased]`:

```markdown
### Added
- New feature description

### Changed
- Updated feature description

### Fixed
- Bug fix description

### Removed
- Removed feature description
```

When releasing a version, move unreleased items to a new version section:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
...
```
