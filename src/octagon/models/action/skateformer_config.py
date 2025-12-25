"""SkateFormer configuration dataclass.

Provides configuration presets for different skeleton formats and tasks.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SkateFormerConfig:
    """Configuration for SkateFormer model.

    This configuration matches the official KAIST SkateFormer implementation
    to enable pretrained weight loading.

    Attributes:
        num_classes: Number of output action classes.
        num_points: Number of skeleton keypoints (24 for NTU, 17 for COCO).
        num_people: Maximum number of people per frame (2 for NTU, 1 for COCO).
        num_frames: Number of temporal frames.
        in_channels: Input coordinate channels (3 for xyz, 2 for xy).
        embed_dim: Initial embedding dimension.
        depths: Number of blocks per stage (4 stages total).
        channels: Output channels per stage.
        num_heads: Number of attention heads.
        kernel_size: Temporal convolution kernel size.
        mlp_ratio: MLP expansion ratio.
        type_1_size: Partition size for type 1 attention (temporal, spatial).
        type_2_size: Partition size for type 2 attention.
        type_3_size: Partition size for type 3 attention.
        type_4_size: Partition size for type 4 attention.
        attn_drop: Attention dropout rate.
        drop: General dropout rate.
        drop_path: Stochastic depth drop rate.
        head_drop: Classifier head dropout.
        index_t: Use temporal index embedding (vs full temporal embedding).
        rel: Use relative position bias in attention.
        global_pool: Pooling type ('avg' or 'max').
    """

    # Dataset parameters
    num_classes: int = 60
    num_points: int = 24
    num_people: int = 2
    num_frames: int = 64
    in_channels: int = 3

    # Architecture parameters
    embed_dim: int = 64
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    channels: Tuple[int, ...] = (96, 192, 192, 192)
    num_heads: int = 32
    kernel_size: int = 7
    mlp_ratio: float = 4.0

    # Partition sizes for 4 skating attention types
    type_1_size: Tuple[int, int] = (8, 8)
    type_2_size: Tuple[int, int] = (8, 12)
    type_3_size: Tuple[int, int] = (8, 8)
    type_4_size: Tuple[int, int] = (8, 12)

    # Dropout rates
    attn_drop: float = 0.5
    drop: float = 0.0
    drop_path: float = 0.2
    head_drop: float = 0.0

    # Embedding options
    index_t: bool = True
    rel: bool = True

    # Pooling
    global_pool: str = "avg"

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert len(self.depths) == 4, "Must have exactly 4 stages"
        assert len(self.channels) == 4, "Must have 4 channel values"
        assert self.global_pool in ("avg", "max"), "global_pool must be 'avg' or 'max'"

    @classmethod
    def ntu60_xsub_joint(cls) -> "SkateFormerConfig":
        """NTU RGB+D 60 Cross-Subject Joint configuration.

        This matches the pretrained weights from KAIST (SkateFormer_j.pt).
        """
        return cls(
            num_classes=60,
            num_points=24,
            num_people=2,
            num_frames=64,
            in_channels=3,
            embed_dim=96,  # Actual pretrained dimension (was incorrectly 64)
            depths=(2, 2, 2, 2),
            channels=(96, 192, 192, 192),
            num_heads=32,
            kernel_size=7,
            mlp_ratio=4.0,
            type_1_size=(8, 8),
            type_2_size=(8, 12),
            type_3_size=(8, 8),
            type_4_size=(8, 12),
            attn_drop=0.5,
            drop_path=0.2,
            index_t=True,
            rel=True,
        )

    @classmethod
    def ntu120_xsub_joint(cls) -> "SkateFormerConfig":
        """NTU RGB+D 120 Cross-Subject Joint configuration."""
        config = cls.ntu60_xsub_joint()
        config.num_classes = 120
        return config

    @classmethod
    def coco_17_boxing(cls, num_classes: int = 6) -> "SkateFormerConfig":
        """COCO 17 keypoints configuration for combat sports.

        Adapted for 2D pose estimation output (x, y only).
        Uses embed_dim=96 to match NTU pretrained weights for transfer learning.

        Args:
            num_classes: Number of action classes (default 6 for boxing).

        Returns:
            Configuration for COCO 17 keypoint format.
        """
        return cls(
            num_classes=num_classes,
            num_points=17,
            num_people=1,
            num_frames=64,
            in_channels=2,  # x, y only (no depth)
            embed_dim=96,  # Match pretrained weights (was 64)
            depths=(2, 2, 2, 2),
            channels=(96, 192, 192, 192),
            num_heads=32,
            kernel_size=7,
            mlp_ratio=4.0,
            type_1_size=(8, 8),
            type_2_size=(8, 12),
            type_3_size=(8, 8),
            type_4_size=(8, 12),
            attn_drop=0.3,  # Lower dropout for smaller dataset
            drop_path=0.1,
            index_t=True,
            rel=True,
        )

    @classmethod
    def coco_17_mma(cls, num_classes: int = 20) -> "SkateFormerConfig":
        """COCO 17 keypoints configuration for MMA/full combat sports.

        Includes more action classes (punches, kicks, takedowns, etc.).

        Args:
            num_classes: Number of action classes.

        Returns:
            Configuration for COCO 17 keypoint format.
        """
        config = cls.coco_17_boxing(num_classes=num_classes)
        return config

    def get_total_blocks(self) -> int:
        """Return total number of transformer blocks."""
        return sum(self.depths)

    def get_final_channels(self) -> int:
        """Return output channels of the last stage."""
        return self.channels[-1]
