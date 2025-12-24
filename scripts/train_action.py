#!/usr/bin/env python
"""Training script for SkateFormer action recognition.

This script trains a SkateFormer model on the BoxingVI dataset for
boxing action recognition.

Usage:
    python scripts/train_action.py --config configs/training/skateformer_boxing.yaml

Example:
    # Train with default config
    python scripts/train_action.py

    # Train with custom config
    python scripts/train_action.py --config path/to/config.yaml

    # Resume from checkpoint
    python scripts/train_action.py --resume checkpoints/latest.pth
"""
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from octagon.data import BoxingVIDataset
from octagon.models.action import SkateFormer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str) -> torch.device:
    """Get device based on configuration."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    dataset_config = config["dataset"]

    train_dataset = BoxingVIDataset(
        root_dir=project_root / dataset_config["root_dir"],
        split="train",
        target_frames=dataset_config["target_frames"],
        normalize=dataset_config["normalize"],
        augment=dataset_config["augment"],
    )

    val_dataset = BoxingVIDataset(
        root_dir=project_root / dataset_config["root_dir"],
        split="val",
        target_frames=dataset_config["target_frames"],
        normalize=dataset_config["normalize"],
        augment=False,
    )

    train_config = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
    )

    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create SkateFormer model."""
    model_config = config["model"]

    model = SkateFormer(
        num_classes=model_config["num_classes"],
        num_joints=model_config["num_joints"],
        num_frames=model_config["num_frames"],
        in_channels=model_config["in_channels"],
        embed_dim=model_config["embed_dim"],
        num_blocks=model_config["num_blocks"],
        num_heads=model_config["num_heads"],
        ffn_expansion=model_config["ffn_expansion"],
        dropout=model_config["dropout"],
        temporal_kernel=model_config["temporal_kernel"],
    )

    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer."""
    opt_config = config["optimizer"]

    if opt_config["type"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
            betas=tuple(opt_config["betas"]),
        )
    elif opt_config["type"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
            momentum=opt_config.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    sched_config = config["scheduler"]
    train_config = config["training"]
    total_epochs = train_config["epochs"]

    if sched_config["type"].lower() == "cosine":
        # Cosine annealing with warmup
        warmup_steps = sched_config["warmup_epochs"] * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            min_lr_ratio = sched_config["min_lr"] / config["optimizer"]["lr"]
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_config["type"].lower() == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config["milestones"],
            gamma=sched_config.get("gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_config['type']}")

    return scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, float, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    accumulation_steps = config["training"]["gradient_accumulation"]
    use_amp = config["mixed_precision"]
    log_every = config["logging"]["log_every"]

    optimizer.zero_grad()

    # Progress bar for batch-level progress
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1:3d}",
        leave=False,
        ncols=100,
    )

    for batch_idx, (skeletons, labels) in pbar:
        skeletons = skeletons.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(skeletons)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights with gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Track metrics
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        global_step += 1

        # Update progress bar with current metrics
        current_acc = 100.0 * correct / total if total > 0 else 0.0
        current_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")

        # Log to tensorboard
        if global_step % log_every == 0:
            writer.add_scalar("train/loss", loss.item() * accumulation_steps, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    pbar.close()
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = config["mixed_precision"]

    for skeletons, labels in val_loader:
        skeletons = skeletons.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = model(skeletons)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = total_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_acc: float,
    config: dict,
    save_path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
) -> tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["best_acc"]


def main():
    parser = argparse.ArgumentParser(description="Train SkateFormer on BoxingVI")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/training/skateformer_boxing.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed
    set_seed(config["seed"])

    # Get device
    device = get_device(config["device"])
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataloaders
    print("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure BoxingVI dataset is downloaded to:")
        print(f"  {project_root / config['dataset']['root_dir']}")
        print("\nExpected structure:")
        print("  data/boxingvi/")
        print("  ├── Annotation_files/")
        print("  ├── RGB_videos/")
        print("  └── Skeleton_data/")
        return

    # Create model
    print("Creating model...")
    model = create_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    steps_per_epoch = len(train_loader) // config["training"]["gradient_accumulation"]
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)

    # Mixed precision scaler
    scaler = GradScaler(enabled=config["mixed_precision"])

    # Loss function with class weights
    class_weights = train_loader.dataset.get_class_weights()
    class_weights = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    global_step = 0

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")

    # Create directories
    checkpoint_dir = project_root / config["checkpoint"]["save_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / config["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir / timestamp)

    # Early stopping
    early_stopping_config = config["early_stopping"]
    patience_counter = 0

    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation']}")
    print(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation']}")
    print()

    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, config, device, epoch, writer, global_step,
        )

        # Validate
        if (epoch + 1) % config["validation"]["eval_every"] == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, config)

            # Log to tensorboard
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                if config["checkpoint"]["save_best"]:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, best_acc, config,
                        checkpoint_dir / "best.pth",
                    )
                    print(f"  -> New best model saved (acc: {best_acc:.2f}%)")
            else:
                patience_counter += 1

            # Early stopping
            if early_stopping_config["enabled"]:
                if patience_counter >= early_stopping_config["patience"]:
                    print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                    break
        else:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

        # Save checkpoint periodically
        if config["checkpoint"]["save_every"] and (epoch + 1) % config["checkpoint"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_acc, config,
                checkpoint_dir / f"epoch_{epoch+1}.pth",
            )

        # Save latest
        if config["checkpoint"]["save_last"]:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_acc, config,
                checkpoint_dir / "latest.pth",
            )

    writer.close()
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
