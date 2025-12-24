# OctagonBrain - AI Agent Guide

This document provides instructions for AI agents (Claude, GPT, etc.) working on this project.

## Quick Start

1. **Read** `docs/PROJECT_STATUS.md` for current state and active tasks
2. **Check** `docs/ARCHITECTURE.md` for system design
3. **Review** `docs/model-docs/` for model-specific documentation
4. **Reference** `docs/components/` for module-specific details

---

## Project Overview

**OctagonBrain** is a combat sports computer vision platform with two products:

| Product | Description | Priority |
|---------|-------------|----------|
| **OctagonCoach** | Mobile training/form analysis app | MVP - Build First |
| **OctagonAnalytics** | Web fight analysis platform | Phase 2 |

### Shared ML Core
Both products share ~70-80% of the ML infrastructure:
- Pose Estimation (YOLO11-Pose, BlazePose)
- Action Recognition (SkateFormer)
- Metrics Engine (CoM, angles, velocity)
- Data Pipeline

---

## Current Phase

**Always check `docs/PROJECT_STATUS.md` for real-time status.**

The project follows a 36-week timeline:
- Phase 1 (Weeks 1-6): Core ML Infrastructure
- Phase 2 (Weeks 6-12): Data Collection & Training
- Phase 3 (Weeks 12-20): OctagonCoach MVP
- Phase 4 (Weeks 20-28): OctagonAnalytics Foundation
- Phase 5 (Weeks 28-36): Advanced Features

---

## Coding Conventions

### Python
```python
# Type hints required
def process_frame(frame: np.ndarray) -> PoseResult:
    """Docstrings for public functions."""
    pass

# Use dataclasses for data structures
@dataclass
class PoseResult:
    keypoints: np.ndarray  # (17, 3) - x, y, confidence
    bbox: np.ndarray       # (4,) - x1, y1, x2, y2
```

### Style
- Python 3.10+
- Type hints on all public functions
- Docstrings on public functions/classes
- Black formatting (line length 88)
- isort for imports
- Use `pathlib.Path` over `os.path`

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Testing
- pytest for all tests
- Tests in `tests/` mirroring `src/` structure
- Run `pytest` before committing
- Run `mypy src/` for type checking

---

## Key Files for Context

| File | Purpose |
|------|---------|
| `docs/PROJECT_STATUS.md` | Current state, active tasks, blockers |
| `docs/ARCHITECTURE.md` | System design, component relationships |
| `docs/RESEARCH.md` | Technology choices, model comparisons |
| `docs/DECISIONS.md` | Architecture Decision Records (ADRs) |
| `docs/model-docs/*.md` | Local model documentation |
| `docs/components/*.md` | Per-module status and TODOs |

---

## Model Documentation

**Before web searching for model info, check `docs/model-docs/`:**

| Model | File | Use Case |
|-------|------|----------|
| YOLO11-Pose | `yolo11-pose.md` | Real-time pose estimation |
| SkateFormer | `skateformer.md` | Skeleton action recognition |
| BlazePose | `blazepose.md` | Mobile pose (33 keypoints) |
| ByteTrack | `bytetrack.md` | Multi-person tracking |
| MediaPipe | `mediapipe.md` | Mobile ML integration |

---

## After Completing Work

### 1. Update PROJECT_STATUS.md
```markdown
| Component | Status | Progress | Last Updated |
|-----------|--------|----------|--------------|
| Pose Estimation | 🟡 In Progress | 60% | 2025-01-15 |
```

### 2. Add CHANGELOG Entry
```markdown
## [2025-01-15]
### Added
- YOLO11-Pose wrapper implementation
- Pose normalization utilities
```

### 3. Update Component Doc
Update the relevant file in `docs/components/`:
```markdown
### Completed
- [x] YOLO11-Pose wrapper
- [x] Pose normalization

### In Progress
- [ ] BlazePose integration
```

### 4. Mark TODOs Complete
If working from a TODO list, mark items as done.

---

## Commit Message Format

```
<type>: <short description>

[optional body]
[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or fixing tests
- `chore`: Build process, dependencies, etc.

**Examples:**
```
feat: add YOLO11-Pose wrapper

Implements pose estimation using Ultralytics YOLO11.
Supports batch processing and GPU acceleration.

Closes #12
```

---

## Common Tasks

### Adding a New Model Wrapper
1. Create file in `src/octagon/models/{category}/`
2. Implement base class interface
3. Add tests in `tests/models/`
4. Add documentation in `docs/model-docs/`
5. Update `docs/components/` status

### Adding a New Metric
1. Create function in `src/octagon/metrics/`
2. Add unit tests
3. Document in relevant component doc

### Running the Pipeline
```python
from octagon.pipelines import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.process_video("path/to/video.mp4")
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use smaller model variant |
| Model not found | Run `scripts/download_models.py` |
| Import errors | Ensure package installed: `pip install -e .` |

### Getting Help
- Check existing component docs
- Review `docs/DECISIONS.md` for context on choices
- Check test files for usage examples

---

## Important Notes

1. **Don't skip documentation updates** - Future agents rely on accurate docs
2. **Check model-docs first** - Avoid unnecessary web searches
3. **Update progress immediately** - Keep PROJECT_STATUS.md current
4. **Follow existing patterns** - Check similar code before implementing new features
5. **Test before completing** - Run `pytest` on affected modules
