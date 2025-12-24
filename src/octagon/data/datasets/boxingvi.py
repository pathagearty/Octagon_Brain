"""BoxingVI dataset loader for boxing action recognition.

BoxingVI is a multi-modal benchmark for boxing action recognition containing
6,915 punch clips from 20 YouTube sparring videos with 6 punch types.

Paper: https://arxiv.org/abs/2511.16524
GitHub: https://github.com/Bikudebug/BoxingVI
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseSkeletonDataset


# BoxingVI class mapping
BOXINGVI_CLASSES = [
    "jab",
    "cross",
    "lead_hook",
    "rear_hook",
    "lead_uppercut",
    "rear_uppercut",
]

# Subject splits as defined in the paper
# Note: Dataset files may be named by subject (S1, S2) or by video (V1, V2)
# The paper describes S1-S15 train, S16-S20 val across 20 videos
TRAIN_SUBJECTS = list(range(1, 16))  # S1-S15 or V1-V15
VAL_SUBJECTS = list(range(16, 21))  # S16-S20 or V16-V20


class BoxingVIDataset(BaseSkeletonDataset):
    """PyTorch Dataset for BoxingVI skeleton data.

    Expected directory structure:
        root_dir/
        ├── Annotation_files/     # Excel files with temporal boundaries
        │   └── *.xlsx
        ├── RGB_videos/           # Video files (optional)
        │   └── *.mp4
        └── Skeleton_data/        # Pre-extracted pose data
            └── *.npy             # (N, 25, 17, 2) arrays

    The skeleton .npy files contain pre-extracted clips where:
        - N: number of clips in the video
        - 25: max frames per clip (at 30 fps)
        - 17: COCO keypoints
        - 2: x, y coordinates

    Attributes:
        samples: List of (skeleton_path, clip_idx, label, label_name) tuples.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        target_frames: int = 64,
        normalize: bool = True,
        augment: bool = False,
        preload: bool = False,
        val_ratio: float = 0.2,
    ) -> None:
        """Initialize BoxingVI dataset.

        Args:
            root_dir: Path to BoxingVI dataset root.
            split: 'train' or 'val'.
            target_frames: Number of frames to interpolate to (default 64).
            normalize: Whether to normalize skeleton positions.
            augment: Whether to apply data augmentation.
            preload: Whether to preload all skeletons into memory.
            val_ratio: Ratio of data for validation if standard split fails (default 0.2).
        """
        self.preload = preload
        self.val_ratio = val_ratio
        self.samples: list[Tuple[Path, int, int, str]] = []
        self._skeleton_cache: dict[str, np.ndarray] = {}

        super().__init__(
            root_dir=root_dir,
            split=split,
            target_frames=target_frames,
            normalize=normalize,
            augment=augment,
        )

        if self.preload:
            self._preload_skeletons()

    @property
    def class_names(self) -> list[str]:
        """Return list of class names."""
        return BOXINGVI_CLASSES

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(BOXINGVI_CLASSES)

    def _load_annotations(self) -> None:
        """Load BoxingVI annotations from Excel files and skeleton data.

        Uses standard V1-V15/V16-V20 split if possible. Falls back to percentage-based
        split if the dataset doesn't contain validation videos (V16+).
        """
        skeleton_dir = self.root_dir / "Skeleton_data"
        annotation_dir = self.root_dir / "Annotation_files"

        if not skeleton_dir.exists():
            raise FileNotFoundError(f"Skeleton directory not found: {skeleton_dir}")

        # Find all skeleton files
        skeleton_files = sorted(skeleton_dir.glob("*.npy"))

        if not skeleton_files:
            raise FileNotFoundError(f"No .npy files found in {skeleton_dir}")

        # Check which video IDs we have
        video_ids = []
        for f in skeleton_files:
            vid = self._extract_subject_id(f.stem)
            if vid is not None:
                video_ids.append(vid)

        # Determine if we need fallback split
        has_train_videos = any(v in TRAIN_SUBJECTS for v in video_ids)
        has_val_videos = any(v in VAL_SUBJECTS for v in video_ids)
        use_fallback = not has_val_videos and has_train_videos

        if use_fallback:
            print(f"Warning: No validation videos (V16-V20) found. Using {self.val_ratio*100:.0f}% of videos for validation.")
            # Sort video IDs and split by percentage
            sorted_ids = sorted(set(video_ids))
            num_val = max(1, int(len(sorted_ids) * self.val_ratio))
            val_videos = set(sorted_ids[-num_val:])  # Last N videos for val
            train_videos = set(sorted_ids[:-num_val])  # Rest for train

            if self.split == "train":
                valid_videos = train_videos
            else:
                valid_videos = val_videos
            print(f"  Train videos: {sorted(train_videos)}")
            print(f"  Val videos: {sorted(val_videos)}")
        else:
            # Use standard split
            if self.split == "train":
                valid_videos = set(TRAIN_SUBJECTS)
            else:
                valid_videos = set(VAL_SUBJECTS)

        for skeleton_path in skeleton_files:
            # Try to extract subject ID from filename
            subject_id = self._extract_subject_id(skeleton_path.stem)

            # If we can extract subject ID, use it for splitting
            if subject_id is not None:
                if subject_id not in valid_videos:
                    continue

            # Load skeleton to get number of clips
            skeleton_data = np.load(skeleton_path)

            # Expected shape: (N, T, V, C) = (num_clips, 25, 17, 2)
            if skeleton_data.ndim != 4:
                print(f"Warning: Unexpected shape {skeleton_data.shape} in {skeleton_path}")
                continue

            num_clips = skeleton_data.shape[0]

            # Try to load corresponding annotation file
            annotation_path = annotation_dir / f"{skeleton_path.stem}.xlsx"
            labels = self._load_clip_labels(annotation_path, num_clips)

            # Add samples
            for clip_idx in range(num_clips):
                label_id, label_name = labels[clip_idx]
                self.samples.append((skeleton_path, clip_idx, label_id, label_name))

        if not self.samples:
            raise ValueError(f"No samples found for split '{self.split}'")

        print(f"BoxingVI {self.split}: Loaded {len(self.samples)} samples")

    def _extract_subject_id(self, filename: str) -> Optional[int]:
        """Extract subject/video ID from filename.

        Handles patterns like:
        - 'S01', 'S1', 'video_S15' (subject-based)
        - 'V1', 'V01', 'video_1' (video-based)
        - 'subject_1', 'subj1' (subject-based)

        Args:
            filename: Skeleton filename without extension.

        Returns:
            Subject/video ID as integer, or None if not found.
        """
        import re

        # Try pattern V## or V# (video-based naming - most common in BoxingVI)
        match = re.search(r"^[Vv](\d+)$", filename)
        if match:
            return int(match.group(1))

        # Try pattern S## or S# (subject-based naming)
        match = re.search(r"[Ss](\d+)", filename)
        if match:
            return int(match.group(1))

        # Try pattern subject_# or subj#
        match = re.search(r"subj(?:ect)?[_-]?(\d+)", filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try pattern video_# or vid#
        match = re.search(r"vid(?:eo)?[_-]?(\d+)", filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _load_clip_labels(
        self, annotation_path: Path, num_clips: int
    ) -> list[Tuple[int, str]]:
        """Load labels for clips from annotation file.

        Args:
            annotation_path: Path to Excel annotation file.
            num_clips: Number of clips in the skeleton file.

        Returns:
            List of (label_id, label_name) tuples, one per clip.
        """
        labels = []

        if annotation_path.exists():
            try:
                df = pd.read_excel(annotation_path)

                # Expected columns: start_frame, end_frame, punch_class
                for i in range(min(len(df), num_clips)):
                    punch_class = str(df.iloc[i].get("punch_class", "")).lower().strip()
                    punch_class = punch_class.replace(" ", "_")

                    if punch_class in BOXINGVI_CLASSES:
                        label_id = BOXINGVI_CLASSES.index(punch_class)
                        labels.append((label_id, punch_class))
                    else:
                        # Try fuzzy matching
                        label_id, label_name = self._fuzzy_match_class(punch_class)
                        labels.append((label_id, label_name))

            except Exception as e:
                print(f"Warning: Could not load annotations from {annotation_path}: {e}")

        # Fill remaining with unknown (use first class as default)
        while len(labels) < num_clips:
            labels.append((0, BOXINGVI_CLASSES[0]))

        return labels

    def _fuzzy_match_class(self, class_name: str) -> Tuple[int, str]:
        """Attempt to match a class name to known classes.

        Args:
            class_name: Input class name to match.

        Returns:
            Tuple of (label_id, label_name).
        """
        class_name = class_name.lower().replace(" ", "_").replace("-", "_")

        # Direct matches with common variations
        mappings = {
            "jab": 0,
            "cross": 1,
            "straight": 1,  # Alternative name for cross
            "lead_hook": 2,
            "leadhook": 2,
            "left_hook": 2,
            "rear_hook": 3,
            "rearhook": 3,
            "right_hook": 3,
            "lead_uppercut": 4,
            "leaduppercut": 4,
            "left_uppercut": 4,
            "rear_uppercut": 5,
            "rearuppercut": 5,
            "right_uppercut": 5,
        }

        if class_name in mappings:
            label_id = mappings[class_name]
            return label_id, BOXINGVI_CLASSES[label_id]

        # Check for partial matches
        for name, label_id in mappings.items():
            if name in class_name or class_name in name:
                return label_id, BOXINGVI_CLASSES[label_id]

        # Default to jab if no match
        return 0, BOXINGVI_CLASSES[0]

    def _preload_skeletons(self) -> None:
        """Preload all skeleton files into memory."""
        skeleton_paths = set(sample[0] for sample in self.samples)

        for path in skeleton_paths:
            if str(path) not in self._skeleton_cache:
                self._skeleton_cache[str(path)] = np.load(path)

        print(f"Preloaded {len(self._skeleton_cache)} skeleton files")

    def _load_skeleton(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """Load skeleton data for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (skeleton array, label id, label name).
        """
        skeleton_path, clip_idx, label_id, label_name = self.samples[idx]

        # Load from cache or file
        if str(skeleton_path) in self._skeleton_cache:
            skeleton_data = self._skeleton_cache[str(skeleton_path)]
        else:
            skeleton_data = np.load(skeleton_path)

        # Extract specific clip: (T, V, C) = (25, 17, 2)
        skeleton = skeleton_data[clip_idx]

        return skeleton, label_id, label_name

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def get_class_weights(self) -> np.ndarray:
        """Compute inverse class frequency weights for balanced training.

        Returns:
            Array of class weights, shape (num_classes,).
        """
        counts = np.zeros(self.num_classes)

        for _, _, label_id, _ in self.samples:
            counts[label_id] += 1

        # Inverse frequency weighting
        total = counts.sum()
        weights = total / (self.num_classes * counts + 1e-6)

        return weights.astype(np.float32)

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sample metadata.
        """
        skeleton_path, clip_idx, label_id, label_name = self.samples[idx]

        return {
            "skeleton_path": str(skeleton_path),
            "clip_idx": clip_idx,
            "label_id": label_id,
            "label_name": label_name,
            "video_name": skeleton_path.stem,
        }
