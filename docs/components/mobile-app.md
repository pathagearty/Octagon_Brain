# Mobile App Component (OctagonCoach)

> **Status:** 🔴 Not Started | **Progress:** 0%

---

## Overview

OctagonCoach is the mobile training app built with React Native. It provides real-time pose estimation, technique detection, and form feedback.

---

## Files

```
mobile/
├── src/
│   ├── App.tsx
│   ├── screens/
│   │   ├── HomeScreen.tsx
│   │   ├── TrainingScreen.tsx
│   │   ├── SessionReviewScreen.tsx
│   │   └── ProgressScreen.tsx
│   ├── components/
│   │   ├── Camera/
│   │   ├── PoseOverlay/
│   │   └── FeedbackDisplay/
│   ├── ml/
│   │   ├── poseDetection.ts
│   │   └── actionRecognition.ts
│   ├── hooks/
│   └── utils/
├── ios/
├── android/
└── package.json
```

---

## Tech Stack

- **Framework:** React Native
- **Camera:** react-native-vision-camera
- **ML (iOS):** CoreML
- **ML (Android):** TensorFlow Lite
- **State:** Zustand or Redux
- **UI:** React Native Paper

---

## Features

### MVP
- [ ] Camera with pose overlay
- [ ] Real-time technique counting
- [ ] Basic form feedback
- [ ] Session summary

### Post-MVP
- [ ] Progress tracking
- [ ] Drill mode
- [ ] Social features
- [ ] Gamification

---

## Implementation Status

### TODO
- [ ] Initialize React Native project
- [ ] Set up react-native-vision-camera
- [ ] Integrate ML Kit for pose detection
- [ ] Create pose visualization overlay
- [ ] Implement technique detection
- [ ] Build training session flow
- [ ] Create progress tracking

---

## References

- [docs/model-docs/blazepose.md](../model-docs/blazepose.md)
- [docs/model-docs/mediapipe.md](../model-docs/mediapipe.md)
