# Architecture Decision Records (ADRs)

This document tracks important architectural decisions made during the project.

---

## Template

```markdown
### ADR-XXX: [Title]

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded
**Superseded by:** ADR-XXX (if applicable)

#### Context
[Why this decision was needed]

#### Decision
[What was decided]

#### Consequences
[Tradeoffs and implications]

#### Alternatives Considered
[Other options that were evaluated]
```

---

## Decisions

### ADR-001: Use YOLO11-Pose for Real-Time Pose Estimation

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Need a pose estimation model that can run in real-time for the training app while maintaining good accuracy. Must support both desktop and mobile deployment.

#### Decision
Use YOLO11-Pose-M as the primary model for real-time inference, with BlazePose as the mobile alternative.

#### Consequences
- **Pros:**
  - YOLO11 is production-ready with excellent documentation
  - Single-stage detection + pose in one forward pass
  - Multiple size variants (nano to extra-large)
  - Easy export to ONNX/TensorRT
- **Cons:**
  - Only 17 keypoints (COCO format) - no detailed hand/foot
  - May need BlazePose for finer-grained analysis

#### Alternatives Considered
- **ViTPose:** Higher accuracy (81.1 AP) but slower, better suited for offline processing
- **RTMPose:** Similar speed but less mature ecosystem
- **DETRPose:** Newer transformer-based, good accuracy but less tested in production

---

### ADR-002: Use SkateFormer for Action Recognition

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Need a skeleton-based action recognition model that can classify combat sports techniques (punches, kicks) from pose sequences.

#### Decision
Use SkateFormer as the primary action recognition model, with a rule-based fallback for simple cases.

#### Consequences
- **Pros:**
  - ECCV 2024 SOTA for skeleton-based action recognition
  - Efficient partition-specific attention reduces computation
  - Pre-trained models available
- **Cons:**
  - Trained on general action datasets (NTU RGB+D), will need fine-tuning
  - Requires substantial labeled combat sports data

#### Alternatives Considered
- **ST-GCN:** Proven but older, lower accuracy
- **VideoMAEv2:** Video-based (not skeleton), heavier compute
- **Rule-based only:** Fast but limited to ~70-80% accuracy

---

### ADR-003: Teacher-Student Architecture for Data Efficiency

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Limited labeled training data for combat sports techniques. Need to maximize data efficiency while maintaining high accuracy.

#### Decision
Use a Teacher-Student architecture:
- **Teacher (Offline):** ViTPose-Huge + manual review for auto-labeling
- **Student (Real-time):** YOLO11-Pose trained on teacher labels

#### Consequences
- **Pros:**
  - Reduces manual annotation burden significantly
  - Teacher can be run offline on cloud GPU
  - Student benefits from teacher's higher accuracy
- **Cons:**
  - Need cloud GPU for teacher inference
  - Some label noise from teacher errors
  - Two-stage pipeline complexity

---

### ADR-004: React Native for Cross-Platform Mobile

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Need to build mobile app for both iOS and Android as a solo developer. Must integrate with on-device ML inference.

#### Decision
Use React Native with react-native-vision-camera for camera access and ML Kit for cross-platform ML inference.

#### Consequences
- **Pros:**
  - Single codebase for iOS and Android
  - Large ecosystem and community
  - react-native-vision-camera has good ML integration
- **Cons:**
  - Performance slightly lower than native
  - Some native code may still be needed for optimization
  - Complex ML model integration

#### Alternatives Considered
- **Flutter:** Good alternative, but less mature ML ecosystem
- **Native (Swift/Kotlin):** Best performance but 2x development time
- **Expo:** Easier setup but limited native module support

---

### ADR-005: Two Products Sharing Core ML Infrastructure

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Building two products: OctagonCoach (training mode) and OctagonAnalytics (fight analysis). Need to decide if they should share code or be separate.

#### Decision
Share ~70-80% of ML infrastructure between products:
- **Shared:** Pose estimation, action recognition, metrics engine, data pipeline
- **Product-specific:** Multi-person tracking (Analytics), form feedback (Coach)

#### Consequences
- **Pros:**
  - Reduced code duplication
  - Improvements benefit both products
  - Single model training pipeline
- **Cons:**
  - Shared code must be more general/flexible
  - Changes can affect both products
  - Testing complexity increases

---

### ADR-006: FastAPI for Backend API

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Need a backend API for video upload, processing, and results retrieval. Must integrate well with Python ML stack.

#### Decision
Use FastAPI with Uvicorn for the backend API.

#### Consequences
- **Pros:**
  - Async support for handling video uploads
  - Native Python integration with ML code
  - Automatic OpenAPI documentation
  - High performance with Uvicorn
- **Cons:**
  - Python GIL can limit true parallelism
  - May need Celery for heavy video processing tasks

#### Alternatives Considered
- **Flask:** Simpler but synchronous, less performant
- **Django:** Full-featured but overkill for API-only
- **Node.js:** Faster but would require Python microservice for ML

---

### ADR-007: ByteTrack for Multi-Person Tracking

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Fight analysis requires tracking two fighters consistently throughout a video, including handling occlusions during clinches and exchanges.

#### Decision
Use ByteTrack for multi-person tracking, with option to upgrade to BoT-SORT if ReID is needed.

#### Consequences
- **Pros:**
  - Excellent occlusion handling (uses low-confidence detections)
  - Already integrated in Ultralytics YOLO
  - Real-time performance
- **Cons:**
  - No ReID by default (BoT-SORT adds this)
  - May struggle with very long occlusions

---

### ADR-008: Full Striking Scope for MVP

**Date:** 2025-12-23
**Status:** Accepted

#### Context
Initially considered boxing-only for faster MVP, but user requested full striking (boxing + kicks) support.

#### Decision
Include both punches and kicks in MVP, resulting in ~20 action classes instead of ~6.

#### Consequences
- **Pros:**
  - More comprehensive product from launch
  - Larger target market (MMA, Kickboxing, Muay Thai)
- **Cons:**
  - ~3x more training data needed
  - Longer development timeline
  - Kick detection is harder (faster motion, more occlusion)

---

## Pending Decisions

### ADR-009: Cloud GPU Provider

**Status:** Proposed

Options under consideration:
- Lambda Labs
- RunPod
- AWS SageMaker
- Google Cloud TPUs

Decision criteria: Cost, availability, ease of use, PyTorch support.

---

### ADR-010: Annotation Tool

**Status:** Proposed

Options under consideration:
- CVAT (Intel, specialized for video)
- Label Studio (HumanSignal, flexible)
- Custom solution

Decision criteria: Video support, pose annotation, team features, cost.
