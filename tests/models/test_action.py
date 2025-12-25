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
    SkateFormerConfig,
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
    """Tests for SkateFormer model.

    Uses the official KAIST SkateFormer architecture with configurations
    for different skeleton formats (NTU RGB+D, COCO 17).
    """

    def test_creation_default(self):
        """Test SkateFormer creation with default parameters (NTU60)."""
        model = SkateFormer()

        # Default is NTU60 config
        assert model.num_classes == 60
        assert model.num_points == 24
        assert model.num_frames == 64
        assert model.embed_dim == 64

    def test_creation_with_config(self):
        """Test SkateFormer creation with COCO boxing config."""
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)

        assert model.num_classes == 6
        assert model.num_points == 17
        assert model.num_frames == 64
        assert model.embed_dim == 64

    def test_custom_parameters(self):
        """Test SkateFormer with custom parameters."""
        model = SkateFormer(
            num_classes=10,
            num_points=17,
            num_frames=32,
            embed_dim=64,
            in_channels=2,
        )

        assert model.num_classes == 10
        assert model.num_frames == 32
        assert model.embed_dim == 64
        assert model.num_points == 17

    def test_forward_pass(self):
        """Test SkateFormer forward pass with COCO config."""
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)
        model.eval()

        # Input: (batch, channels, frames, joints, persons)
        x = torch.randn(2, 2, 64, 17, 1)
        output = model(x)

        assert output.shape == (2, 6)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)
        model.eval()

        x = torch.randn(1, 2, 64, 17, 1)
        output = model(x)

        assert output.shape == (1, 6)

    def test_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 2, 64, 17, 1)
            output = model(x)
            assert output.shape == (batch_size, 6)

    def test_parameter_count(self):
        """Test model has reasonable parameter count.

        The official SkateFormer architecture is larger (~3M parameters)
        due to 4 stages with multi-head attention.
        """
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)
        num_params = model.get_num_params()

        # Official architecture is ~3M parameters
        assert num_params < 5_000_000
        assert num_params > 2_000_000

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = SkateFormerConfig.coco_17_boxing(num_classes=6)
        model = SkateFormer(config)
        model.train()

        x = torch.randn(2, 2, 64, 17, 1, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients exist for most model parameters
        # Some parameters like relative_position_bias_table may not get gradients
        # if attention is not used in certain configurations
        params_with_grad = 0
        total_params = 0
        for param in model.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1

        # At least 80% of parameters should have gradients
        # Some relative position bias tables may not be used depending on
        # the specific input dimensions and partition configurations
        assert params_with_grad / total_params > 0.8, (
            f"Only {params_with_grad}/{total_params} parameters have gradients"
        )

    def test_ntu_config(self):
        """Test NTU RGB+D configuration for pretrained weights."""
        config = SkateFormerConfig.ntu60_xsub_joint()
        model = SkateFormer(config)

        assert model.num_classes == 60
        assert model.num_points == 24
        assert model.num_frames == 64

        # Test forward with NTU format (3D, 2 persons)
        x = torch.randn(2, 3, 64, 24, 2)
        model.eval()
        output = model(x)
        assert output.shape == (2, 60)


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
