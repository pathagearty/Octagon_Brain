# OctagonBrain

> Combat Sports Computer Vision Analysis Platform

## Overview

OctagonBrain is an AI-powered platform for analyzing combat sports (MMA, Boxing, Kickboxing) using computer vision. The platform consists of two products:

- **OctagonCoach**: Mobile-first training and form analysis app
- **OctagonAnalytics**: Web-based fight analysis platform (similar to DeepStrike)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/octagon-brain-cv.git
cd octagon-brain-cv

# Install dependencies
pip install -e .

# Run tests
pytest
```

## Project Status

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for current development status.

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | AI Agent guide for contributing |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Current project state and progress |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and component relationships |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Technology research and decisions |
| [docs/model-docs/](docs/model-docs/) | Local model documentation (YOLO11, SkateFormer, etc.) |

## Features

### OctagonCoach (Training Mode)
- Real-time pose estimation
- Technique detection (punches, kicks)
- Form analysis and feedback
- Training session tracking
- Progress over time

### OctagonAnalytics (Fight Analysis)
- Multi-person tracking
- Fighter identification
- Strike counting per fighter
- Fight statistics
- Video upload and processing

## Technology Stack

- **ML Framework**: PyTorch
- **Pose Estimation**: YOLO11-Pose, BlazePose (mobile)
- **Action Recognition**: SkateFormer
- **Tracking**: ByteTrack
- **Backend**: FastAPI
- **Mobile**: React Native
- **Inference**: TensorRT, CoreML, TFLite

## Project Structure

```
octagon-brain-cv/
├── src/octagon/           # Python source code
│   ├── models/            # Model wrappers (pose, action, tracking)
│   ├── metrics/           # Combat metrics calculations
│   ├── analysis/          # Form and fight analysis
│   ├── pipelines/         # End-to-end pipelines
│   └── api/               # FastAPI backend
├── mobile/                # React Native app
├── docs/                  # Documentation
│   ├── components/        # Per-component docs
│   ├── guides/            # How-to guides
│   ├── model-docs/        # Local model documentation
│   └── references/        # External references
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter experiments
├── tests/                 # Test suite
└── configs/               # Configuration files
```

## Contributing

1. Read [CLAUDE.md](CLAUDE.md) for contribution guidelines
2. Check [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for current tasks
3. Follow existing code patterns
4. Update documentation when completing work

## License

[License TBD]

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer)
- [MediaPipe](https://github.com/google/mediapipe)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
