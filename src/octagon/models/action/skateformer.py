"""SkateFormer model for skeleton-based action recognition.

Adapted from the original SkateFormer paper (ECCV 2024) for COCO 17 keypoints
and combat sports action recognition.

Paper: https://arxiv.org/abs/2403.09508
Original: https://github.com/KAIST-VICLab/SkateFormer
"""
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionResult, BaseActionRecognizer, BoxingAction


def get_skate_partitions(
    num_joints: int = 17,
    num_frames: int = 64,
    temporal_kernel: int = 7,
) -> dict:
    """Generate skeletal-temporal partitions for attention.

    Partitions joints and frames into neighboring/distant groups for
    the four Skate-Types of attention patterns.

    Args:
        num_joints: Number of keypoints (17 for COCO).
        num_frames: Number of temporal frames.
        temporal_kernel: Size of temporal neighborhood.

    Returns:
        Dictionary with partition indices for each Skate-Type.
    """
    # COCO skeleton connectivity for neighboring joints
    # (left side, right side, spine)
    coco_neighbors = {
        0: [1, 2],  # nose
        1: [0, 3],  # left_eye
        2: [0, 4],  # right_eye
        3: [1],     # left_ear
        4: [2],     # right_ear
        5: [7, 11], # left_shoulder
        6: [8, 12], # right_shoulder
        7: [5, 9],  # left_elbow
        8: [6, 10], # right_elbow
        9: [7],     # left_wrist
        10: [8],    # right_wrist
        11: [5, 13], # left_hip
        12: [6, 14], # right_hip
        13: [11, 15], # left_knee
        14: [12, 16], # right_knee
        15: [13],   # left_ankle
        16: [14],   # right_ankle
    }

    # Create adjacency matrix
    adj = torch.zeros(num_joints, num_joints)
    for joint, neighbors in coco_neighbors.items():
        for neighbor in neighbors:
            adj[joint, neighbor] = 1
            adj[neighbor, joint] = 1

    return {
        "adjacency": adj,
        "temporal_kernel": temporal_kernel,
    }


class SkateEmbedding(nn.Module):
    """Skeletal-temporal embedding combining joint and temporal features.

    Uses learnable skeletal embeddings and sinusoidal temporal encoding,
    combined via outer product to capture joint-frame relationships.
    """

    def __init__(
        self,
        num_joints: int = 17,
        num_frames: int = 64,
        in_channels: int = 2,
        embed_dim: int = 64,
    ) -> None:
        """Initialize SkateEmbedding.

        Args:
            num_joints: Number of keypoints.
            num_frames: Number of temporal frames.
            in_channels: Input coordinate channels (2 for x,y).
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Project input coordinates to embedding space
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Learnable skeletal embeddings (per joint)
        self.joint_embed = nn.Parameter(torch.randn(1, 1, num_joints, embed_dim))

        # Temporal positional encoding (sinusoidal)
        position = torch.arange(num_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_frames, 1, embed_dim)
        pe[0, :, 0, 0::2] = torch.sin(position * div_term)
        pe[0, :, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("temporal_pe", pe)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skeletal-temporal embedding.

        Args:
            x: Input tensor (B, C, T, V, M) where C=2 for x,y coordinates.

        Returns:
            Embedded tensor (B, T, V, embed_dim).
        """
        B, C, T, V, M = x.shape

        # Reshape for projection: (B, T, V, M, C) -> (B*T*V*M, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)

        # Project to embedding dimension
        x = self.input_proj(x)

        # Reshape back: (B, T, V, M, embed_dim)
        x = x.reshape(B, T, V, M, self.embed_dim)

        # Average over persons dimension: (B, T, V, embed_dim)
        x = x.mean(dim=3)

        # Add skeletal embedding (broadcast over batch and time)
        x = x + self.joint_embed[:, :, :V, :]

        # Add temporal encoding (broadcast over joints)
        x = x + self.temporal_pe[:, :T, :, :]

        # Layer normalization
        x = self.norm(x)

        return x


class SkateAttention(nn.Module):
    """Partition-specific multi-head self-attention.

    Implements efficient attention by processing skeletal-temporal
    relations within partitioned groups.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SkateAttention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor (B, T, V, embed_dim).

        Returns:
            Output tensor (B, T, V, embed_dim).
        """
        B, T, V, D = x.shape

        # Flatten spatial-temporal dimensions for attention
        x = x.reshape(B, T * V, D)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T * V, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T*V, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T * V, D)

        # Project output
        out = self.proj(out)
        out = self.dropout(out)

        # Reshape back
        out = out.reshape(B, T, V, D)

        return out


class TemporalConv(nn.Module):
    """Temporal convolution block for local temporal modeling."""

    def __init__(
        self,
        embed_dim: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        """Initialize TemporalConv.

        Args:
            embed_dim: Embedding dimension.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,  # Depthwise
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution.

        Args:
            x: Input tensor (B, T, V, embed_dim).

        Returns:
            Output tensor (B, T, V, embed_dim).
        """
        B, T, V, D = x.shape

        # Reshape for conv1d: (B*V, D, T)
        x = x.permute(0, 2, 3, 1).reshape(B * V, D, T)

        # Apply convolution
        x = self.conv(x)

        # Reshape back: (B, T, V, D)
        x = x.reshape(B, V, D, T).permute(0, 3, 1, 2)

        # Normalize
        x = self.norm(x)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        embed_dim: int = 64,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize FeedForward.

        Args:
            embed_dim: Input/output dimension.
            expansion: Hidden layer expansion factor.
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor (..., embed_dim).

        Returns:
            Output tensor (..., embed_dim).
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SkateFormerBlock(nn.Module):
    """Single SkateFormer transformer block.

    Combines Skate-MSA, temporal convolution, and FFN with
    residual connections and layer normalization.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        temporal_kernel: int = 7,
    ) -> None:
        """Initialize SkateFormerBlock.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ffn_expansion: FFN expansion factor.
            dropout: Dropout probability.
            temporal_kernel: Temporal conv kernel size.
        """
        super().__init__()

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SkateAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.tconv = TemporalConv(embed_dim, temporal_kernel, dropout)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SkateFormer block.

        Args:
            x: Input tensor (B, T, V, embed_dim).

        Returns:
            Output tensor (B, T, V, embed_dim).
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x))

        # Temporal conv with residual
        x = x + self.tconv(self.norm2(x))

        # FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x


class SkateFormer(nn.Module):
    """SkateFormer model for skeleton-based action recognition.

    Adapted for COCO 17 keypoints and 2D coordinates from the original
    NTU RGB+D 25 joint format.

    Architecture:
        - SkateEmbedding for joint-temporal positional encoding
        - 8 SkateFormerBlocks with temporal downsampling
        - Global average pooling and classification head

    Attributes:
        num_classes: Number of action classes.
        num_joints: Number of skeleton keypoints.
        num_frames: Input sequence length.
        embed_dim: Transformer embedding dimension.
    """

    def __init__(
        self,
        num_classes: int = 6,
        num_joints: int = 17,
        num_frames: int = 64,
        in_channels: int = 2,
        embed_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        temporal_kernel: int = 7,
    ) -> None:
        """Initialize SkateFormer.

        Args:
            num_classes: Number of output classes.
            num_joints: Number of skeleton keypoints (17 for COCO).
            num_frames: Number of input frames.
            in_channels: Input coordinate channels (2 for x,y).
            embed_dim: Transformer embedding dimension.
            num_blocks: Number of SkateFormer blocks.
            num_heads: Number of attention heads.
            ffn_expansion: FFN expansion factor.
            dropout: Dropout probability.
            temporal_kernel: Temporal conv kernel size.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Embedding layer
        self.embedding = SkateEmbedding(
            num_joints=num_joints,
            num_frames=num_frames,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # SkateFormer blocks with temporal downsampling
        self.blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        current_frames = num_frames
        for i in range(num_blocks):
            self.blocks.append(
                SkateFormerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                    temporal_kernel=temporal_kernel,
                )
            )

            # Temporal downsampling after every 2 blocks
            if (i + 1) % 2 == 0 and current_frames > 8:
                new_frames = current_frames // 2
                self.downsample_layers.append(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
                )
                current_frames = new_frames
            else:
                self.downsample_layers.append(nn.Identity())

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input skeleton tensor (B, C, T, V, M).
               C=2 for x,y coordinates, T=frames, V=joints, M=persons.

        Returns:
            Class logits (B, num_classes).
        """
        # Embedding: (B, C, T, V, M) -> (B, T, V, embed_dim)
        x = self.embedding(x)

        # Apply blocks with downsampling
        for block, downsample in zip(self.blocks, self.downsample_layers):
            x = block(x)

            # Temporal downsampling
            if not isinstance(downsample, nn.Identity):
                B, T, V, D = x.shape
                # Reshape: (B, T, V, D) -> (B*V, D, T)
                x = x.permute(0, 2, 3, 1).reshape(B * V, D, T)
                x = downsample(x)
                T_new = x.shape[-1]
                # Reshape back: (B, T_new, V, D)
                x = x.reshape(B, V, D, T_new).permute(0, 3, 1, 2)

        # Final normalization
        x = self.final_norm(x)

        # Global average pooling over time and joints: (B, embed_dim)
        x = x.mean(dim=(1, 2))

        # Classification
        logits = self.classifier(x)

        return logits

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def load_pretrained(
        self,
        pretrained_path: str | Path,
        strict: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Load pre-trained weights with partial matching.

        Handles dimension mismatches between pre-trained model (NTU RGB+D: 25 joints, 3D)
        and this model (COCO: 17 joints, 2D). Loads compatible layers and skips incompatible ones.

        Args:
            pretrained_path: Path to pre-trained checkpoint file.
            strict: If True, raises error on mismatches. If False, skips mismatched layers.
            verbose: If True, prints loading details.

        Returns:
            Dictionary with 'loaded', 'skipped', and 'missing' layer names.
        """
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pre-trained weights not found: {pretrained_path}")

        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            pretrained_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            pretrained_dict = checkpoint["model_state_dict"]
        else:
            pretrained_dict = checkpoint

        # Clean up key names (remove 'module.' prefix if present from DataParallel)
        pretrained_dict = {
            k.replace("module.", ""): v for k, v in pretrained_dict.items()
        }

        model_dict = self.state_dict()

        # Track what we load/skip
        loaded_keys = []
        skipped_keys = []
        missing_keys = []

        # Try to match each pre-trained weight
        for key, pretrained_value in pretrained_dict.items():
            if key in model_dict:
                model_value = model_dict[key]
                if pretrained_value.shape == model_value.shape:
                    model_dict[key] = pretrained_value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {pretrained_value.shape} vs {model_value.shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")

        # Check for missing keys
        for key in model_dict.keys():
            if key not in pretrained_dict and key not in [k.split(" ")[0] for k in skipped_keys]:
                missing_keys.append(key)

        # Load the matched weights
        self.load_state_dict(model_dict, strict=False)

        if verbose:
            print(f"\n{'='*50}")
            print("Pre-trained Weight Loading Summary")
            print(f"{'='*50}")
            print(f"Loaded: {len(loaded_keys)} layers")
            print(f"Skipped: {len(skipped_keys)} layers (dimension mismatch)")
            print(f"Missing: {len(missing_keys)} layers (randomly initialized)")

            if skipped_keys and verbose:
                print(f"\nSkipped layers (expected for different skeleton format):")
                for key in skipped_keys[:10]:
                    print(f"  - {key}")
                if len(skipped_keys) > 10:
                    print(f"  ... and {len(skipped_keys) - 10} more")

        return {
            "loaded": loaded_keys,
            "skipped": skipped_keys,
            "missing": missing_keys,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        num_classes: int = 6,
        num_joints: int = 17,
        num_frames: int = 64,
        in_channels: int = 2,
        freeze_backbone: bool = False,
        verbose: bool = True,
    ) -> "SkateFormer":
        """Create model and load pre-trained weights for transfer learning.

        Args:
            pretrained_path: Path to pre-trained checkpoint.
            num_classes: Number of output classes for new task.
            num_joints: Number of skeleton joints (17 for COCO).
            num_frames: Number of input frames.
            in_channels: Number of input channels (2 for 2D).
            freeze_backbone: If True, freeze all layers except classifier.
            verbose: Print loading details.

        Returns:
            SkateFormer model with pre-trained weights loaded.

        Example:
            >>> model = SkateFormer.from_pretrained(
            ...     'pretrained/ntu_xsub.pt',
            ...     num_classes=6,  # Boxing classes
            ...     freeze_backbone=False,
            ... )
        """
        # Create model with target configuration
        model = cls(
            num_classes=num_classes,
            num_joints=num_joints,
            num_frames=num_frames,
            in_channels=in_channels,
        )

        # Load pre-trained weights (partial matching)
        result = model.load_pretrained(pretrained_path, verbose=verbose)

        # Optionally freeze backbone (everything except classifier)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            if verbose:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"\nFroze backbone: {trainable:,} / {total:,} params trainable")

        return model


class SkateFormerWrapper(BaseActionRecognizer):
    """Inference wrapper for SkateFormer model.

    Provides a high-level interface for action classification from
    skeleton sequences, handling preprocessing and postprocessing.

    Example:
        >>> model = SkateFormerWrapper(checkpoint='weights/skateformer.pth')
        >>> skeleton = np.random.rand(64, 17, 2).astype(np.float32)
        >>> result = model.predict(skeleton)
        >>> print(f"Action: {result.action}, Confidence: {result.confidence:.3f}")
    """

    def __init__(
        self,
        num_classes: int = 6,
        num_joints: int = 17,
        num_frames: int = 64,
        checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize SkateFormerWrapper.

        Args:
            num_classes: Number of action classes.
            num_joints: Number of skeleton keypoints.
            num_frames: Expected input sequence length.
            checkpoint: Path to model weights.
            device: Device to run on ('cuda', 'cpu', or None for auto).
        """
        super().__init__()

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._num_classes = num_classes
        self.num_joints = num_joints
        self.num_frames = num_frames
        self._class_names = [
            "jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"
        ][:num_classes]

        # Initialize model
        self.model = SkateFormer(
            num_classes=num_classes,
            num_joints=num_joints,
            num_frames=num_frames,
            in_channels=2,
        )

        # Load checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    @property
    def num_classes(self) -> int:
        """Return number of action classes."""
        return self._num_classes

    @property
    def class_names(self) -> list[str]:
        """Return list of class names."""
        return self._class_names

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

    def predict(
        self,
        skeleton: np.ndarray,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> ActionResult:
        """Classify action from skeleton sequence.

        Args:
            skeleton: Skeleton array (T, V, C) or (C, T, V, M).
            start_frame: Start frame in original video.
            end_frame: End frame in original video.

        Returns:
            ActionResult with predicted action and confidence.
        """
        # Handle different input formats
        if skeleton.ndim == 3:
            # (T, V, C) -> (C, T, V, M)
            T, V, C = skeleton.shape
            skeleton = skeleton.transpose(2, 0, 1)  # (C, T, V)
            skeleton = skeleton[np.newaxis, ..., np.newaxis]  # (1, C, T, V, 1)
        elif skeleton.ndim == 4:
            # (C, T, V, M) -> (1, C, T, V, M)
            skeleton = skeleton[np.newaxis]

        # Convert to tensor
        x = torch.from_numpy(skeleton).float().to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        # Get prediction
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        probabilities = probs[0].cpu().numpy()

        # Get action name
        action = BoxingAction(pred_class)
        action_name = action.name.lower()

        if end_frame is None:
            end_frame = start_frame + self.num_frames

        return ActionResult(
            action=action_name,
            action_id=pred_class,
            confidence=confidence,
            probabilities=probabilities,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def predict_batch(
        self,
        skeletons: list[np.ndarray],
    ) -> list[ActionResult]:
        """Classify actions for a batch of skeleton sequences.

        Args:
            skeletons: List of skeleton arrays, each (T, V, C).

        Returns:
            List of ActionResult objects.
        """
        results = []
        for skeleton in skeletons:
            results.append(self.predict(skeleton))
        return results
