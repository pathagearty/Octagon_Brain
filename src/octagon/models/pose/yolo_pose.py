"""YOLO11-Pose wrapper for pose estimation.

This module provides a wrapper around the Ultralytics YOLO11-Pose model
for consistent pose estimation interface in OctagonBrain.
"""
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch

from .base import BasePoseEstimator, PoseResult


class YOLO11Pose(BasePoseEstimator):
    """YOLO11-Pose wrapper for pose estimation.

    Wraps the Ultralytics YOLO11-Pose model for consistent interface.
    Supports multiple model sizes and automatic GPU/CPU device selection.

    Attributes:
        model: The loaded YOLO model.
        device: Device for inference ('cuda', 'cpu', or device index).
        confidence_threshold: Minimum detection confidence.
        variant: Model size variant (n, s, m, l, x).

    Example:
        >>> estimator = YOLO11Pose(variant="m", device="cuda")
        >>> result = estimator.detect(frame)
        >>> print(f"Detected {result.num_detections} persons")

        >>> # Process video
        >>> results = estimator.detect_video("path/to/video.mp4")
        >>> for result in results:
        ...     if result.num_detections > 0:
        ...         keypoints = result.get_keypoint_xy(0)
    """

    VARIANTS = ("n", "s", "m", "l", "x")

    def __init__(
        self,
        variant: Literal["n", "s", "m", "l", "x"] = "m",
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ):
        """Initialize YOLO11-Pose model.

        Args:
            variant: Model size variant ('n', 's', 'm', 'l', 'x').
                - n: Nano (fastest, least accurate)
                - s: Small
                - m: Medium (recommended balance)
                - l: Large
                - x: Extra large (slowest, most accurate)
            model_path: Optional path to custom model weights.
                If None, uses official pretrained weights.
            device: Device for inference. If None, auto-selects GPU if available.
            confidence_threshold: Minimum detection confidence (0-1).
            iou_threshold: NMS IoU threshold (0-1).

        Raises:
            ImportError: If ultralytics package is not installed.
            ValueError: If variant is not valid.
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got '{variant}'")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        self.variant = variant
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        if model_path is not None:
            self.model = YOLO(str(model_path))
        else:
            model_name = f"yolo11{variant}-pose.pt"
            self.model = YOLO(model_name)

        # Move to device
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> PoseResult:
        """Detect poses in a single frame.

        Args:
            frame: Input image as numpy array (H, W, 3) in BGR format.

        Returns:
            PoseResult containing detected poses.

        Example:
            >>> import cv2
            >>> frame = cv2.imread("image.jpg")
            >>> result = estimator.detect(frame)
            >>> print(f"Found {result.num_detections} people")
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        return self._parse_result(results[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[PoseResult]:
        """Detect poses in multiple frames.

        Args:
            frames: List of input images.

        Returns:
            List of PoseResult, one per frame.

        Example:
            >>> frames = [frame1, frame2, frame3]
            >>> results = estimator.detect_batch(frames)
            >>> for i, result in enumerate(results):
            ...     print(f"Frame {i}: {result.num_detections} people")
        """
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        return [self._parse_result(r) for r in results]

    def _parse_result(self, result: "ultralytics.engine.results.Results") -> PoseResult:
        """Parse YOLO result into PoseResult.

        Args:
            result: Single result from YOLO inference.

        Returns:
            Parsed PoseResult.
        """
        # Handle empty detections
        if result.keypoints is None or len(result.keypoints) == 0:
            return PoseResult(
                keypoints=np.zeros((0, 17, 3), dtype=np.float32),
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )

        # Extract keypoints: (N, 17, 3) with x, y, confidence
        keypoints_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

        # Extract boxes: (N, 4)
        boxes = result.boxes.xyxy.cpu().numpy()

        # Extract detection scores: (N,)
        scores = result.boxes.conf.cpu().numpy()

        return PoseResult(
            keypoints=keypoints_data.astype(np.float32),
            boxes=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
        )

    def detect_video(
        self,
        video_path: Union[str, Path],
        stride: int = 1,
        max_frames: Optional[int] = None,
        return_frames: bool = False,
    ) -> Union[list[PoseResult], tuple[list[PoseResult], list[np.ndarray]]]:
        """Detect poses in a video file.

        Args:
            video_path: Path to video file.
            stride: Process every Nth frame.
            max_frames: Optional limit on number of frames to process.
            return_frames: If True, also return the video frames.

        Returns:
            List of PoseResult, one per processed frame.
            If return_frames=True, returns (results, frames) tuple.

        Example:
            >>> results = estimator.detect_video("video.mp4", stride=2)
            >>> print(f"Processed {len(results)} frames")

            >>> # With frame return
            >>> results, frames = estimator.detect_video("video.mp4", return_frames=True)
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        results = []
        frames_list = [] if return_frames else None
        frame_count = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % stride == 0:
                    result = self.detect(frame)
                    result.frame_id = frame_count
                    # Calculate timestamp based on video FPS
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        result.timestamp = frame_count / fps
                    results.append(result)

                    if return_frames:
                        frames_list.append(frame.copy())

                    processed_count += 1
                    if max_frames and processed_count >= max_frames:
                        break

                frame_count += 1
        finally:
            cap.release()

        if return_frames:
            return results, frames_list
        return results

    def detect_stream(
        self,
        source: Union[str, int] = 0,
        display: bool = False,
    ):
        """Generator for detecting poses from a video stream.

        Args:
            source: Video source (file path, URL, or camera index).
            display: If True, display annotated frames (requires GUI).

        Yields:
            Tuple of (frame, PoseResult) for each frame.

        Example:
            >>> for frame, result in estimator.detect_stream(0):
            ...     if result.num_detections > 0:
            ...         print(f"Detected {result.num_detections} people")
            ...     if cv2.waitKey(1) == ord('q'):
            ...         break
        """
        import cv2

        cap = cv2.VideoCapture(source)
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect(frame)
                result.frame_id = frame_count

                if display:
                    # Get annotated frame from YOLO
                    results = self.model(frame, verbose=False)
                    annotated = results[0].plot()
                    cv2.imshow("YOLO11-Pose", annotated)
                    if cv2.waitKey(1) == ord("q"):
                        break

                yield frame, result
                frame_count += 1
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    def normalize_poses(
        self,
        result: PoseResult,
        person_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Normalize poses from a PoseResult.

        Centers on hip midpoint and scales by torso length.

        Args:
            result: PoseResult from detection.
            person_idx: If specified, normalize only that person.
                If None, normalize all detected persons.

        Returns:
            Normalized keypoints array.
            Shape (17, 3) if person_idx specified, else (N, 17, 3).
        """
        if person_idx is not None:
            keypoints = result.keypoints[person_idx]
            return self.normalize_keypoints(keypoints)
        else:
            return np.array([
                self.normalize_keypoints(kpts)
                for kpts in result.keypoints
            ])

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        return {
            "variant": self.variant,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "num_keypoints": 17,
            "keypoint_format": "COCO",
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"YOLO11Pose(variant='{self.variant}', "
            f"device='{self.device}', "
            f"conf={self.confidence_threshold})"
        )
