# Combat Sports Analysis App - Research & Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for building a combat sports training and analysis application similar to DeepStrike by Jabbr.ai, but designed for a solo developer. The app will use computer vision to analyze body movements, track limbs, estimate weight distribution, and provide training feedback for MMA, Boxing, and Kickboxing.

---

## Part 1: Research Findings & Technology Landscape (December 2025)

### 1.1 Pose Estimation Models - Current State of the Art

| Model | Release | Accuracy | Speed | Keypoints | Best For |
|-------|---------|----------|-------|-----------|----------|
| **YOLO11-Pose** | Late 2024 | Strong | 30+ FPS (GPU) | 17 (COCO) | Production real-time |
| **DETRPose** | June 2025 | 75.1 AP | Real-time | 17 | Multi-person, transformer |
| **RTMPose/RTMO** | 2024 | 75.8% AP | 90+ FPS (CPU) | 17/133 | High throughput |
| **ViTPose-Huge** | 2022+ | 81.1 AP (SOTA) | Slower | 17 | Maximum accuracy (Teacher) |
| **BlazePose** | 2020+ | Good | 30+ FPS mobile | 33 (3D) | Mobile, 3D estimation |

**Key 2025 Discovery - DETRPose:**
- First real-time transformer-based multi-person pose estimation model
- Trains 10-12x faster than competitors (48 epochs vs 700 for RTMO)
- Outperforms YOLO11-X on COCO test-dev
- GitHub: https://github.com/SebastianJanampa/DETRPose

**Recommendation:** Use a **Teacher-Student architecture**:
- **Teacher (Offline):** ViTPose-Huge for auto-labeling and training data generation
- **Student (Real-time):** YOLO11-Pose-M or DETRPose for production inference

### 1.2 Action Recognition Models

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **SkateFormer** (ECCV 2024) | Skeleton Transformer | SOTA accuracy, efficient | Punch/kick classification |
| **VideoMAEv2** | Self-supervised | Strong foundation model | Pre-training on fight videos |
| **SA-TDGFormer** (Feb 2025) | GCN + Transformer hybrid | Local + global features | Complex actions |
| **ST-GCN variants** | Graph neural network | Proven, well-documented | Baseline model |

**SkateFormer Key Features:**
- Partitions joints and frames based on skeletal-temporal relations
- 4 distinct relation types combining skeletal and temporal patterns
- Pre-trained models available for NTU RGB+D datasets
- GitHub: https://github.com/KAIST-VICLab/SkateFormer

### 1.3 Multi-Object Tracking

| Tracker | Performance | Notes |
|---------|-------------|-------|
| **BoT-SORT** | Best overall | ReID + camera motion compensation |
| **ByteTrack** | Excellent | Fast, handles occlusion well |
| **DeepOCSort** | Strong | Deep learning enhanced |

**For Combat Sports:** BoT-SORT or ByteTrack are recommended. Both are integrated in Ultralytics YOLO.

### 1.4 Detection & Segmentation

| Model | Use Case |
|-------|----------|
| **Grounded-SAM-2** (Aug 2024) | Text-prompted detection + segmentation |
| **YOLO11-detect** | Fast fighter detection |
| **Florence-2 + SAM 2** | Open-vocabulary segmentation |

### 1.5 Center of Mass & Biomechanics

Recent 2025 research (MDPI Sensors) presents methods for whole-body 3D pose estimation integrating body mass distribution and center of gravity constraints:
- Uses anatomical mass ratios (head: 8%, trunk: 50%, etc.)
- Achieves 44.49mm MPJPE (60.4% improvement over baselines)
- Critical for weight distribution analysis in striking

### 1.6 Available Datasets

| Dataset | Size | Content | Link |
|---------|------|---------|------|
| **BoxingVI** (Nov 2025) | 6,915 clips | 18 athletes, 6 punch types, 2D pose + temporal annotations | [arXiv](https://arxiv.org/abs/2511.16524) / [GitHub](https://github.com/Bikudebug/BoxingVI) |
| **Olympic Boxing (Kaggle)** | Videos | Real fights, referee labeled | Kaggle |
| **MDPI Boxing Detection** | 312K frames | Punch/no-punch labeled | MDPI |
| **NTU RGB+D 120** | 114K clips | 120 action classes | Standard benchmark |

---

## Part 2: Verified Technology Stack Recommendations

### 2.1 Core ML Pipeline

```
RECOMMENDED STACK (Solo Developer Optimized):

Pose Estimation:
├── Mobile/Training Mode: MediaPipe BlazePose (33 keypoints, 3D, easy integration)
├── Desktop Real-time: YOLO11-Pose-M or RTMPose-M (balanced)
├── Fight Analysis: DETRPose or YOLO11-Pose-L (multi-person)
└── Teacher/Labeling: ViTPose-Huge (maximum accuracy)

Action Recognition:
├── Primary: SkateFormer (SOTA skeleton-based)
├── Alternative: Fine-tuned ST-GCN (simpler, proven)
└── Video Foundation: VideoMAEv2 (optional, for pre-training)

Tracking:
└── ByteTrack or BoT-SORT (both integrated in Ultralytics)

Framework:
├── ML Framework: PyTorch
├── Pose Toolbox: MMPose or Ultralytics
├── Inference: ONNX + TensorRT (NVIDIA) / CoreML (iOS) / TFLite (Android)
└── Backend: FastAPI + Uvicorn
```

### 2.2 Mobile Development

| Platform | Framework | ML Runtime |
|----------|-----------|------------|
| iOS | Swift/SwiftUI or React Native | CoreML |
| Android | Kotlin or React Native | TensorFlow Lite |
| Cross-platform | React Native + ML Kit | Google ML Kit |
| Cross-platform | Flutter | TFLite/google_ml_kit |

**Recommendation:** React Native with react-native-vision-camera + ML Kit for rapid cross-platform development.

### 2.3 Annotation & Data Pipeline

| Tool | Best For | Notes |
|------|----------|-------|
| **CVAT** | Video/pose annotation | Intel-maintained, specialized for CV |
| **Label Studio** | Multi-modal, cloud-native | More flexible integrations |

**Recommendation:** CVAT for pose and action annotation (better video support, interpolation, automatic tracking).

### 2.4 Deployment & Infrastructure

```
Local Development:
├── GPU: RTX 3080/4080+ or cloud GPU (Lambda Labs, RunPod)
├── Storage: 2TB+ SSD for video data
└── RAM: 32GB+

Production:
├── API: FastAPI + Uvicorn (async, fast)
├── GPU Inference: NVIDIA TensorRT (2-10x speedup)
├── Model Serving: ONNX Runtime or Triton Inference Server
├── Cloud: AWS (EC2, S3, ECR, Fargate) or GCP
└── Mobile: On-device inference preferred
```

---

## Part 3: Development Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal:** Set up development environment and basic pose estimation pipeline

1. **Environment Setup**
   - Python 3.10+, PyTorch, CUDA, Ultralytics
   - Project structure (see repository structure below)
   - GPU development environment

2. **Basic Pose Estimation**
   - Implement YOLO11-Pose wrapper
   - Implement BlazePose wrapper for mobile testing
   - Frame extraction and pose normalization utilities

3. **Initial Data Collection**
   - Record yourself performing basic techniques (50-100 reps each)
   - Set up CVAT for annotation
   - Create initial labeled dataset (jab, cross, hook, uppercut)

### Phase 2: Metrics Engine (Weeks 4-8)
**Goal:** Calculate combat-sports-specific metrics from pose data

1. **Static Metrics (single frame)**
   - Stance width calculation
   - Guard position analysis
   - Weight distribution estimation (using CoM)
   - Hip/shoulder rotation angles

2. **Dynamic Metrics (pose sequences)**
   - Punch/kick detection via velocity thresholds
   - Movement trajectory analysis
   - Speed and power estimation

3. **Biomechanics Module**
   - Center of mass estimation using segment mass percentages
   - Balance analysis
   - Weight transfer detection

### Phase 3: Action Recognition (Weeks 8-14)
**Goal:** Classify combat techniques from pose sequences

1. **Rule-Based Baseline**
   - Implement heuristic punch classification (70-80% accuracy target)
   - Use velocity, trajectory, and arm extension features
   - Test on collected data

2. **Classical ML Layer**
   - Train SVM/Random Forest on hand-crafted features
   - Target 85-90% accuracy with ~1000 labeled examples

3. **Deep Learning (Optional)**
   - Fine-tune SkateFormer on combat-specific data
   - Requires 2000+ labeled examples per class
   - Target 95%+ accuracy

### Phase 4: Mobile App MVP (Weeks 14-20)
**Goal:** Build cross-platform training mode app

1. **Mobile Framework**
   - React Native + react-native-vision-camera
   - ML Kit / CoreML / TFLite integration
   - Basic UI/UX for training mode

2. **On-Device Inference**
   - Convert models to CoreML/TFLite
   - Optimize for mobile (quantization, pruning)
   - Target 30+ FPS on modern phones

3. **Core Features**
   - Real-time punch counting
   - Basic form feedback
   - Session statistics

### Phase 5: Form Analysis (Weeks 20-26)
**Goal:** Add detailed technique analysis and feedback

1. **Reference Pose Database**
   - Capture expert form for each technique
   - Multiple angles and phases (stance, extension, recovery)

2. **Comparison Engine**
   - Pose similarity scoring
   - Joint-by-joint analysis
   - Automated feedback generation

3. **Training Drills**
   - Guided practice sessions
   - Progressive difficulty
   - Gamification elements

### Phase 6: Fight Analysis (Weeks 26-32)
**Goal:** Analyze multi-person combat footage (post-MVP)

1. **Multi-Person Pipeline**
   - Upgrade to DETRPose or YOLO11-L for multi-person
   - ByteTrack/BoT-SORT integration
   - Fighter assignment and tracking

2. **Fight Statistics**
   - Strike counting per fighter
   - Landed vs missed classification
   - Distance and positioning analysis

3. **Video Upload & Processing**
   - Cloud processing pipeline
   - Async video analysis
   - Results visualization

---

## Part 4: Repository Structure

```
octagon-brain/
├── README.md
├── pyproject.toml                    # Dependencies (use uv or poetry)
├── .env.example
│
├── configs/                          # Configuration files
│   ├── models/
│   │   ├── pose.yaml
│   │   └── action.yaml
│   ├── training/
│   └── inference/
│
├── src/octagon/
│   ├── __init__.py
│   │
│   ├── data/                         # Data handling
│   │   ├── datasets/
│   │   ├── preprocessing/
│   │   └── augmentation/
│   │
│   ├── models/                       # Model wrappers
│   │   ├── pose/
│   │   │   ├── yolo_pose.py
│   │   │   ├── blazepose.py
│   │   │   └── vitpose.py
│   │   ├── action/
│   │   │   ├── skateformer.py
│   │   │   └── rule_based.py
│   │   └── tracking/
│   │       └── byte_track.py
│   │
│   ├── metrics/                      # Combat metrics
│   │   ├── static.py                 # Single-frame metrics
│   │   ├── dynamic.py                # Sequence metrics
│   │   └── biomechanics.py           # CoM, weight distribution
│   │
│   ├── analysis/                     # High-level analysis
│   │   ├── form_analyzer.py
│   │   ├── fight_analyzer.py
│   │   └── feedback.py
│   │
│   ├── pipelines/                    # End-to-end pipelines
│   │   ├── training_mode.py          # Single-person training
│   │   └── fight_mode.py             # Two-person analysis
│   │
│   └── api/                          # Backend API
│       ├── main.py                   # FastAPI app
│       └── routes/
│
├── mobile/                           # Mobile app (React Native)
│   ├── src/
│   └── ios/android
│
├── scripts/                          # Utility scripts
│   ├── download_models.py
│   ├── train_action.py
│   └── export_mobile.py
│
├── notebooks/                        # Experiments
│
├── data/                             # Data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
└── weights/                          # Model weights (gitignored)
```

---

## Part 5: Key Technical Decisions

### 5.1 Why This Stack?

1. **YOLO11-Pose + BlazePose combo:**
   - YOLO11 for accuracy and multi-person capability
   - BlazePose for mobile with 33 keypoints (including hands/feet detail)
   - Both are production-ready and well-documented

2. **SkateFormer for action recognition:**
   - ECCV 2024 SOTA, specifically designed for skeleton-based action
   - Efficient enough for near-real-time
   - Pre-trained checkpoints available

3. **Teacher-Student for data efficiency:**
   - Use ViTPose (heavy) to auto-label your training data
   - Train lightweight models on the labels
   - Reduces manual annotation burden significantly

4. **React Native for mobile:**
   - Single codebase for iOS/Android
   - Good ML Kit integration via community libraries
   - Faster iteration as a solo developer

### 5.2 Realistic Scope for Solo Developer

**Start with Training Mode (not Fight Analysis):**
- No multi-person tracking complexity
- No occlusion handling
- Controlled environment (user wants accurate tracking)
- Can request specific camera angles

**Advantages over Jabbr/DeepStrike:**
- They need multi-camera, multi-person, real-time fight analysis
- You can focus on personal training, which is actually a larger market
- Training mode builds value immediately while you develop harder features

### 5.3 Data Strategy

1. **Self-recording:** 2-3 hours of your own training (500-1000 examples)
2. **YouTube scraping:** Tutorial videos for technique variety (for training only)
3. **BoxingVI dataset:** 6,915 clips, 6 punch types (Jab, Cross, Lead/Rear Hook, Lead/Rear Uppercut), includes 2D pose
4. **Active learning:** Deploy basic model, collect user corrections

---

## Part 6: Sources & References

### Pose Estimation
- [YOLO11 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [DETRPose GitHub](https://github.com/SebastianJanampa/DETRPose)
- [RTMPose Paper](https://arxiv.org/abs/2303.07399)
- [ViTPose Supervisely Guide](https://supervisely.com/blog/vitpose-state-of-the-art-pose-estimation-model-in-supervisely/)
- [MMPose GitHub](https://github.com/open-mmlab/mmpose)

### Action Recognition
- [SkateFormer GitHub](https://github.com/KAIST-VICLab/SkateFormer)
- [SkateFormer Paper](https://arxiv.org/abs/2403.09508)
- [VideoMAEv2 GitHub](https://github.com/OpenGVLab/VideoMAEv2)
- [3D Skeleton Action Recognition Survey](https://arxiv.org/html/2506.00915v1)
- [ST-GCN GitHub](https://github.com/yysijie/st-gcn)

### Combat Sports AI
- [Jabbr.ai](https://jabbr.ai/)
- [Combat IQ](https://www.combatiq.io/press)
- [BoxingVI Dataset Paper](https://arxiv.org/abs/2511.16524)
- [BoxingVI GitHub Repository](https://github.com/Bikudebug/BoxingVI)
- [MDPI Boxing Punch Detection](https://www.mdpi.com/1099-4300/26/8/617)

### Deployment & Mobile
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [React Native ML Kit](https://github.com/a7medev/react-native-ml-kit)
- [FastAPI ML Deployment](https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/)
- [CVAT Annotation Tool](https://www.cvat.ai)

### Biomechanics
- [Whole-Body 3D Pose with CoM Constraints](https://www.mdpi.com/1424-8220/25/13/3944)
- [Center of Mass Estimation Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7830449/)

### Tracking
- [ByteTrack GitHub](https://github.com/FoundationVision/ByteTrack)
- [Ultralytics Tracking Documentation](https://docs.ultralytics.com/modes/track/)

---

## Part 7: Validation of Previous Research

The user-provided research from other LLMs was largely accurate. Here are key validations and updates:

**Validated:**
- YOLO11-Pose is indeed the production standard for 2025
- BlazePose with 33 keypoints is excellent for mobile training mode
- Teacher-Student architecture is the right approach for data efficiency
- ByteTrack remains the go-to tracker for sports video
- Starting with training mode (single-person) is strategically wise
- Center of mass estimation using segment mass percentages is standard biomechanics

**Updates/Corrections:**
- **DETRPose (June 2025)** is a major new option not mentioned - first real-time transformer pose model
- **SkateFormer (ECCV 2024)** is the current SOTA for skeleton-based action recognition
- **RTMPose/RTMO** from MMPose offers excellent speed-accuracy tradeoffs
- **BoxingVI dataset (Nov 2025)** provides 6,915 boxing clips with temporal annotations, 2D pose, and 6 punch types
- **Grounded-SAM-2** enables powerful text-prompted segmentation for fighter detection

**Cautions:**
- The 6-month timeline is aggressive for a solo developer - plan for 9-12 months
- Data annotation is the biggest time sink - invest in auto-labeling early
- Fight analysis (multi-person) should be Phase 2, not Phase 1

---

## Part 8: Refined Scope Based on User Requirements

### 8.1 Product Strategy: Two Products, Shared Core

Based on clarification, we're building **two products**:

1. **OctagonCoach** - Training/form analysis app for gyms and personal use
2. **OctagonAnalytics** - Fight analysis platform (like DeepStrike)

**Key Decision: Shared Model Architecture - YES**

The core ML models can be shared between both products (~70-80% shared code):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SHARED ML CORE                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │ Pose          │  │ Action        │  │ Metrics Engine            │   │
│  │ Estimation    │  │ Recognition   │  │ (CoM, angles, velocity)   │   │
│  │ (YOLO11/DETR) │  │ (SkateFormer) │  │                           │   │
│  └───────────────┘  └───────────────┘  └───────────────────────────┘   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Data Pipeline: Video → Frames → Poses → Normalized Skeletons     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│   OCTAGON COACH         │          │   OCTAGON ANALYTICS     │
│   (Training Mode)       │          │   (Fight Analysis)      │
├─────────────────────────┤          ├─────────────────────────┤
│ • Single-person focus   │          │ • Multi-person tracking │
│ • Real-time feedback    │          │ • Fighter identification│
│ • Form comparison       │          │ • Strike stats per      │
│ • Progress tracking     │          │   fighter               │
│ • Drill suggestions     │          │ • Video upload/process  │
│ • Mobile-first          │          │ • Web dashboard         │
└─────────────────────────┘          └─────────────────────────┘
```

### 8.2 Technique Scope: Full Striking

Target action classes for MVP:

**Punches (8 classes):**
1. Jab (lead hand straight)
2. Cross (rear hand straight)
3. Lead Hook
4. Rear Hook
5. Lead Uppercut
6. Rear Uppercut
7. Overhand
8. Body shots (variant modifier)

**Kicks (8 classes):**
1. Roundhouse (lead leg)
2. Roundhouse (rear leg)
3. Front Kick / Teep (lead)
4. Front Kick / Teep (rear)
5. Side Kick
6. Back Kick
7. Axe Kick
8. Low Kick / Leg Kick

**Defensive/Movement (4 classes):**
1. Slip
2. Bob/Weave
3. Check (kick defense)
4. Footwork/Stance change

**Total: ~20 action classes** (vs 4-6 for boxing-only)

### 8.3 Platform: React Native + Web (Simultaneous)

```
Technology Stack:
├── Mobile: React Native + react-native-vision-camera
│   ├── iOS: CoreML for on-device inference
│   └── Android: TensorFlow Lite
│
├── Web: React (shared components with React Native Web)
│
├── Backend: FastAPI + PostgreSQL + Redis
│   ├── Video processing service
│   ├── ML inference service (TensorRT)
│   └── User/session management
│
└── Cloud: AWS
    ├── EC2/Lambda for inference
    ├── S3 for video storage
    └── CloudFront for CDN
```

### 8.4 Infrastructure: Cloud GPU Training

Available resources:
- **Training:** Lambda Labs / RunPod / AWS SageMaker
- **Inference:** TensorRT on cloud GPU + CoreML/TFLite on mobile
- **Storage:** AWS S3 for videos and model weights

---

## Part 9: Revised Development Phases

### Phase 1: Core ML Infrastructure (Weeks 1-6)
**Goal:** Build the shared ML core that both products will use

**Week 1-2: Environment & Pose Pipeline**
- Set up Python development environment (PyTorch, Ultralytics, MMPose)
- Implement YOLO11-Pose wrapper with pose normalization
- Implement BlazePose wrapper for mobile testing
- Set up cloud GPU training environment (Lambda Labs or RunPod)

**Week 3-4: Action Recognition Foundation**
- Download and process BoxingVI dataset (boxing)
- Set up SkateFormer with pre-trained weights
- Implement rule-based baseline for comparison
- Create data pipeline for skeleton sequences

**Week 5-6: Metrics Engine**
- Implement static metrics (stance, guard, angles)
- Implement dynamic metrics (velocity, acceleration)
- Center of mass estimation module
- Weight distribution calculation

### Phase 2: Data Collection & Training (Weeks 6-12)
**Goal:** Build training dataset for full striking

**Week 6-8: Kick Data Collection**
- Record kick techniques (self + partners)
- Set up CVAT annotation workspace
- Begin manual annotation of kick dataset
- Auto-label using ViTPose teacher model

**Week 9-10: Model Training**
- Fine-tune SkateFormer on combined punch/kick data
- Train rule-based classifier as fast fallback
- Evaluate on held-out test set
- Iterate on problem classes

**Week 11-12: Model Optimization**
- Export to ONNX format
- Optimize with TensorRT for cloud inference
- Convert to CoreML for iOS
- Convert to TFLite for Android
- Benchmark inference speeds

### Phase 3: OctagonCoach MVP (Weeks 12-20)
**Goal:** Launch training mode app first (faster to market)

**Week 12-14: Mobile App Foundation**
- React Native project setup
- Camera integration (react-native-vision-camera)
- ML Kit / CoreML / TFLite integration
- Basic pose visualization

**Week 15-17: Core Training Features**
- Real-time technique counting
- Basic form feedback (guard position, stance width)
- Session recording and summary
- Strike velocity display

**Week 18-20: Polish & Beta**
- Progress tracking over time
- Drill mode with prompts
- User authentication
- Beta testing with real users

### Phase 4: OctagonAnalytics Foundation (Weeks 20-28)
**Goal:** Build fight analysis capabilities

**Week 20-22: Multi-Person Pipeline**
- Implement ByteTrack/BoT-SORT integration
- Fighter detection and assignment
- Handle occlusion scenarios
- Two-person pose tracking

**Week 23-25: Fight Statistics**
- Strike counting per fighter
- Landed vs missed classification
- Distance and positioning analysis
- Round-by-round breakdown

**Week 26-28: Web Dashboard**
- Video upload interface
- Async processing pipeline
- Results visualization
- Export and sharing features

### Phase 5: Integration & Advanced Features (Weeks 28-36)
**Goal:** Polish both products, add advanced features

**Week 28-30: Form Analysis (Coach)**
- Reference pose database from experts
- Pose comparison scoring
- Joint-by-joint feedback
- Technique improvement suggestions

**Week 31-33: Advanced Fight Analysis**
- Power estimation from kinetic chain
- Combo detection and patterns
- Fighter style classification
- Comparison with historical data

**Week 34-36: Production Hardening**
- Load testing and optimization
- Error handling and recovery
- Analytics and monitoring
- App store submission preparation

---

## Part 10: Risk Assessment & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Kick data insufficient | High | Medium | Partner with gyms, use synthetic augmentation |
| Multi-person tracking failures | High | Medium | Start with clear footage, add occlusion handling iteratively |
| Mobile inference too slow | Medium | Low | Use lighter models (YOLO11-n), aggressive quantization |
| Action recognition accuracy <90% | Medium | Medium | Ensemble methods, rule-based fallback |
| Solo developer burnout | High | Medium | Phase releases, focus on Coach MVP first |

### Recommended MVP Order

1. **OctagonCoach (Training Mode)** - Launch first
   - Simpler (single person)
   - Faster to revenue
   - Generates training data from real users

2. **OctagonAnalytics (Fight Analysis)** - Launch second
   - More complex (multi-person)
   - Can use learnings from Coach
   - Harder to differentiate from Jabbr initially

---

## Part 11: Success Metrics

### Technical Metrics
- Pose estimation: >90% PCK@0.5 on test set
- Action recognition: >85% mAP for punches, >80% for kicks
- Mobile inference: >25 FPS on iPhone 12+
- Cloud inference: <500ms per video second

### Product Metrics (OctagonCoach MVP)
- Real-time technique detection accuracy: >85% user satisfaction
- Form feedback relevance: >70% user agreement with suggestions
- Session completion rate: >60%
- Weekly active retention: >40% after 4 weeks

---

## Final Notes

This plan is designed for a solo developer with:
- Full striking scope (boxing + kicks)
- Two products sharing core ML infrastructure
- React Native for simultaneous mobile/web
- Cloud GPU budget for training
- Realistic 9-month timeline to both products

The key insight is that **OctagonCoach should launch first** - it's simpler technically, generates revenue faster, and the user data helps improve models for fight analysis later.

---

**Ready to begin implementation.** The first step would be setting up the development environment and core pose estimation pipeline.
