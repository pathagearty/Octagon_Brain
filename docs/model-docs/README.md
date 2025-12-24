# Model Documentation

This directory contains local documentation for the ML models used in OctagonBrain.

**Purpose:** Provide quick reference for AI agents without requiring web searches.

---

## Available Documentation

| Model/Dataset | File | Purpose | Last Updated |
|---------------|------|---------|--------------|
| YOLO11-Pose | [yolo11-pose.md](yolo11-pose.md) | Real-time pose estimation | 2025-12-23 |
| SkateFormer | [skateformer.md](skateformer.md) | Skeleton action recognition | 2025-12-23 |
| BlazePose | [blazepose.md](blazepose.md) | Mobile pose estimation (33 kpts) | 2025-12-23 |
| ByteTrack | [bytetrack.md](bytetrack.md) | Multi-person tracking | 2025-12-23 |
| MediaPipe | [mediapipe.md](mediapipe.md) | Mobile ML integration | 2025-12-23 |
| **BoxingVI** | [boxingvi.md](boxingvi.md) | Boxing action recognition dataset | 2025-12-23 |

---

## How to Use

1. **Check here first** before web searching for model information
2. **Each doc contains:**
   - Overview and purpose
   - Installation
   - API reference (key functions/classes)
   - Input/output formats
   - Code examples
   - Common issues

---

## Adding New Model Documentation

When adding a new model:

1. Create `{model-name}.md` in this directory
2. Use the template below
3. Update this README's table

### Template

```markdown
# [Model Name] Documentation

> Source: [Official documentation URL]
> Last Updated: YYYY-MM-DD

## Overview
[What the model does and when to use it]

## Installation
[pip/conda install commands]

## Basic Usage
[Code example]

## API Reference
[Key functions/classes and their parameters]

## Input/Output Format
[Data shapes, types, and meanings]

## Common Issues
[Known problems and solutions]
```

---

## External Resources

For the most up-to-date information, refer to:

| Model/Dataset | Official Docs | GitHub |
|---------------|---------------|--------|
| YOLO11 | [Ultralytics Docs](https://docs.ultralytics.com/) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| SkateFormer | [Paper](https://arxiv.org/abs/2403.09508) | [KAIST-VICLab/SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) |
| BlazePose | [MediaPipe Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | [google/mediapipe](https://github.com/google/mediapipe) |
| ByteTrack | [Paper](https://arxiv.org/abs/2110.06864) | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) |
| BoxingVI | [Paper](https://arxiv.org/abs/2511.16524) | [Bikudebug/BoxingVI](https://github.com/Bikudebug/BoxingVI) |
