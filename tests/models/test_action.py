"""Tests for action recognition models."""
import numpy as np
import pytest
import torch

from octagon.models.action import (
    BOXING_CLASS_NAMES,
    NUM_BOXING_CLASSES,
    ActionResult,
    BaseActionRecognizer,
    BoxingAction,
    SkateFormer,
    SkateFormerWrapper,
)


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_creation(self):
        """Test ActionResult creation with valid data."""
        result = ActionResult(
            action="jab",
            action_id=0,
            confidence=0.95,
            probabilities=np.array([0.95, 0.03, 0.01, 0.005, 0.003, 0.002]),
            start_frame=0,
            end_frame=64,
        )

        assert result.action == "jab"
        assert result.action_id == 0
        assert result.confidence == 0.95
        assert len(result.probabilities) == 6
        assert result.start_frame == 0
        assert result.end_frame == 64

    def test_probabilities_sum(self):
        """Test that probabilities can sum to approximately 1."""
        probs = np.array([0.4, 0.3, 0.15, 0.1, 0.03, 0.02])
        result = ActionResult(
            action="cross",
            action_id=1,
            confidence=0.4,
            probabilities=probs,
            start_frame=10,
            end_frame=74,
        )

        assert np.isclose(result.probabilities.sum(), 1.0)


class TestBoxingAction:
    """Tests for BoxingAction enum."""

    def test_all_classes(self):
        """Test all boxing action classes are defined."""
        assert BoxingAction.JAB == 0
        assert BoxingAction.CROSS == 1
        assert BoxingAction.LEAD_HOOK == 2
        assert BoxingAction.REAR_HOOK == 3
        assert BoxingAction.LEAD_UPPERCUT == 4
        assert BoxingAction.REAR_UPPERCUT == 5

    def test_num_classes(self):
        """Test number of boxing classes."""
        assert len(BoxingAction) == NUM_BOXING_CLASSES
        assert NUM_BOXING_CLASSES == 6

    def test_class_names(self):
        """Test BOXING_CLASS_NAMES mapping."""
        assert len(BOXING_CLASS_NAMES) == 6
        assert BOXING_CLASS_NAMES[0] == "Jab"
        assert BOXING_CLASS_NAMES[1] == "Cross"
        assert BOXING_CLASS_NAMES[5] == "Rear Uppercut"


class TestSkateFormer:
    """Tests for SkateFormer model."""

    def test_creation(self):
        """Test SkateFormer creation with default parameters."""
        model = SkateFormer()

        assert model.num_classes == 6
        assert model.num_joints == 17
        assert model.num_frames == 64
        assert model.embed_dim == 64
        assert model.num_blocks == 8

    def test_custom_parameters(self):
        """Test SkateFormer with custom parameters."""
        model = SkateFormer(
            num_classes=10,
            num_joints=17,
            num_frames=32,
            embed_dim=128,
            num_blocks=4,
        )

        assert model.num_classes == 10
        assert model.num_frames == 32
        assert model.embed_dim == 128
        assert model.num_blocks == 4

    def test_forward_pass(self):
        """Test SkateFormer forward pass."""
        model = SkateFormer(num_classes=6, num_joints=17, num_frames=64)
        model.eval()

        # Input: (batch, channels, frames, joints, persons)
        x = torch.randn(2, 2, 64, 17, 1)
        output = model(x)

        assert output.shape == (2, 6)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        model = SkateFormer()
        model.eval()

        x = torch.randn(1, 2, 64, 17, 1)
        output = model(x)

        assert output.shape == (1, 6)

    def test_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        model = SkateFormer()
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 2, 64, 17, 1)
            output = model(x)
            assert output.shape == (batch_size, 6)

    def test_parameter_count(self):
        """Test model has reasonable parameter count."""
        model = SkateFormer()
        num_params = model.get_num_params()

        # Should be less than 1M parameters for this compact model
        assert num_params < 1_000_000
        assert num_params > 100_000  # But more than 100K

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SkateFormer()
        model.train()

        x = torch.randn(2, 2, 64, 17, 1, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSkateFormerWrapper:
    """Tests for SkateFormerWrapper inference class."""

    def test_creation(self):
        """Test SkateFormerWrapper creation."""
        wrapper = SkateFormerWrapper(num_classes=6, device="cpu")

        assert wrapper.num_classes == 6
        assert wrapper.num_joints == 17
        assert wrapper.num_frames == 64

    def test_predict_3d_input(self):
        """Test prediction with (T, V, C) input format."""
        wrapper = SkateFormerWrapper(device="cpu")

        # (frames, joints, channels)
        skeleton = np.random.rand(64, 17, 2).astype(np.float32)
        result = wrapper.predict(skeleton)

        assert isinstance(result, ActionResult)
        # Action name is lowercase with underscores
        valid_actions = ["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"]
        assert result.action in valid_actions
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 6

    def test_predict_4d_input(self):
        """Test prediction with (C, T, V, M) input format."""
        wrapper = SkateFormerWrapper(device="cpu")

        # (channels, frames, joints, persons)
        skeleton = np.random.rand(2, 64, 17, 1).astype(np.float32)
        result = wrapper.predict(skeleton)

        assert isinstance(result, ActionResult)
        assert 0 <= result.action_id < 6

    def test_predict_batch(self):
        """Test batch prediction."""
        wrapper = SkateFormerWrapper(device="cpu")

        skeletons = [np.random.rand(64, 17, 2).astype(np.float32) for _ in range(3)]
        results = wrapper.predict_batch(skeletons)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ActionResult)


class TestBaseActionRecognizer:
    """Tests for BaseActionRecognizer abstract class."""

    def test_cannot_instantiate(self):
        """Test that BaseActionRecognizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseActionRecognizer()
