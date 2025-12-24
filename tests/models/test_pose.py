"""Tests for pose estimation models."""
import numpy as np
import pytest

from octagon.models.pose import (
    COCO_FLIP_INDICES,
    COCO_KEYPOINT_NAMES,
    COCO_KEYPOINTS,
    COCO_SKELETON,
    BasePoseEstimator,
    PoseResult,
)


class TestPoseResult:
    """Tests for PoseResult dataclass."""

    def test_creation(self):
        """Test PoseResult creation with valid data."""
        keypoints = np.random.rand(2, 17, 3).astype(np.float32)
        boxes = np.random.rand(2, 4).astype(np.float32)
        scores = np.random.rand(2).astype(np.float32)

        result = PoseResult(keypoints=keypoints, boxes=boxes, scores=scores)

        assert result.keypoints.shape == (2, 17, 3)
        assert result.boxes.shape == (2, 4)
        assert result.scores.shape == (2,)

    def test_num_detections(self):
        """Test num_detections property."""
        keypoints = np.random.rand(5, 17, 3)
        boxes = np.random.rand(5, 4)
        scores = np.random.rand(5)

        result = PoseResult(keypoints=keypoints, boxes=boxes, scores=scores)

        assert result.num_detections == 5

    def test_empty_result(self):
        """Test creation with no detections."""
        keypoints = np.zeros((0, 17, 3))
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))

        result = PoseResult(keypoints=keypoints, boxes=boxes, scores=scores)

        assert result.num_detections == 0
        assert result.is_empty

    def test_filter_by_score(self):
        """Test filtering detections by confidence score."""
        keypoints = np.random.rand(5, 17, 3)
        boxes = np.random.rand(5, 4)
        scores = np.array([0.9, 0.3, 0.8, 0.2, 0.7])

        result = PoseResult(keypoints=keypoints, boxes=boxes, scores=scores)
        filtered = result.filter_by_score(0.5)

        assert filtered.num_detections == 3
        assert all(s >= 0.5 for s in filtered.scores)

    def test_optional_fields(self):
        """Test optional frame_id and timestamp fields."""
        keypoints = np.random.rand(1, 17, 3)
        boxes = np.random.rand(1, 4)
        scores = np.random.rand(1)

        result = PoseResult(
            keypoints=keypoints,
            boxes=boxes,
            scores=scores,
            frame_id=42,
            timestamp=1.5,
        )

        assert result.frame_id == 42
        assert result.timestamp == 1.5


class TestCOCOConstants:
    """Tests for COCO keypoint constants."""

    def test_coco_keypoints(self):
        """Test COCO_KEYPOINTS is valid."""
        assert len(COCO_KEYPOINTS) == 17
        assert COCO_KEYPOINTS[0] == "nose"
        assert COCO_KEYPOINTS[5] == "left_shoulder"
        assert COCO_KEYPOINTS[16] == "right_ankle"

    def test_coco_keypoint_names(self):
        """Test COCO_KEYPOINT_NAMES mapping."""
        assert len(COCO_KEYPOINT_NAMES) == 17
        assert COCO_KEYPOINT_NAMES["nose"] == 0
        assert COCO_KEYPOINT_NAMES["left_wrist"] == 9
        assert COCO_KEYPOINT_NAMES["right_hip"] == 12

    def test_coco_skeleton(self):
        """Test COCO_SKELETON connections."""
        assert len(COCO_SKELETON) > 0
        # Each connection should be a pair of valid indices
        for start, end in COCO_SKELETON:
            assert 0 <= start < 17
            assert 0 <= end < 17

    def test_coco_flip_indices(self):
        """Test COCO_FLIP_INDICES for horizontal flipping."""
        assert len(COCO_FLIP_INDICES) == 17

        # Symmetric joints should map to themselves
        assert COCO_FLIP_INDICES[0] == 0  # nose

        # Left-right pairs should be swapped
        # Left shoulder (5) <-> Right shoulder (6)
        assert COCO_FLIP_INDICES[5] == 6
        assert COCO_FLIP_INDICES[6] == 5

        # Left hip (11) <-> Right hip (12)
        assert COCO_FLIP_INDICES[11] == 12
        assert COCO_FLIP_INDICES[12] == 11


class TestBasePoseEstimator:
    """Tests for BasePoseEstimator abstract class."""

    def test_cannot_instantiate(self):
        """Test that BasePoseEstimator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePoseEstimator()
