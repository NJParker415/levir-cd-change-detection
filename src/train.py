"""Training loop"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.dataset import build_dataloaders
from src. model import SiameseUNet
from src.loss import BCEDiceLoss
from src.evaluate import MetricTracker

def get_lr_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup + cosine decay"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / (max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        image_A = batch["image_A"].to(device)
        image_B = batch["image_B"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(image_A, image_B)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    tracker = MetricTracker(threshold=0.5)
    num_batches = 0
    total_loss = 0.0

    for batch in loader:
        image_A = batch["image_A"].to(device)
        image_B = batch["image_B"].to(device)
        mask = batch["mask"].to(device)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):

            logits = model(image_A, image_B)
            loss = criterion(logits, mask)

        total_loss += loss.item()
        num_batches += 1
        tracker.update(logits, mask)

    return total_loss / num_batches, tracker.compute()

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, path):
    """Save model checkpoint"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_f1": best_f1,
    }, path)

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
) -> dict:
    """Load checkpoint, restoring model and optionally optimizer/scheduler."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
 
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
 
    return checkpoint
 
 
def train(
    data_root: str,
    output_dir: str = "results",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    patience: int = 15,
    num_workers: int = 4,
    pos_weight: float = 3.0,
    resume: str = None,
) -> None:
    """
    Full training pipeline.
 
    Args:
        data_root: Path to pre-cropped patches (with train/val/test subdirs)
        output_dir: Where to save checkpoints and logs
        epochs: Maximum training epochs
        batch_size: Batch size (8 works well for 256x256 on a single GPU)
        lr: Peak learning rate after warmup
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience (epochs without F1 improvement)
        num_workers: DataLoader workers
        pos_weight: BCE positive class weight for class imbalance
        resume: Path to checkpoint to resume from
    """
    os.makedirs(output_dir, exist_ok=True)
 
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
 
    # Data
    loaders = build_dataloaders(data_root, batch_size, num_workers)
    assert "train" in loaders and "val" in loaders, "Need train and val splits"
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")
 
    # Model
    model = SiameseUNet().to(device)
    print(f"Parameters: {model.count_parameters():,}")
 
    # Loss, optimizer, scheduler
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
 
    num_training_steps = len(loaders["train"]) * epochs
    num_warmup_steps = len(loaders["train"]) * warmup_epochs
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps)
 
    scaler = GradScaler(enabled=device.type == "cuda")
 
    # Resume from checkpoint if provided
    start_epoch = 0
    best_f1 = 0.0
    if resume and os.path.exists(resume):
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
 
    # Training loop
    epochs_without_improvement = 0
 
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
 
        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device
        )
 
        # Validate
        val_loss, val_metrics = validate(model, loaders["val"], criterion, device)
 
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
 
        # Logging
        print(
            "Epoch {}/{} ({:.0f}s) | "
            "LR: {:.2e} | "
            "Train Loss: {:.4f} | "
            "Val Loss: {:.4f} | "
            "F1: {:.4f} | "
            "IoU: {:.4f} | "
            "P: {:.4f} | "
            "R: {:.4f}".format(
                epoch + 1,
                epochs,
                epoch_time,
                current_lr,
                train_loss,
                val_loss,
                val_metrics["f1"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
            )
        )
 
        # Save best by F1, not loss
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_f1,
                os.path.join(output_dir, "best_model.pth"),
            )
            print(f"  -> New best F1: {best_f1:.4f}, saved checkpoint")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
 
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_f1,
            os.path.join(output_dir, "latest_model.pth"),
        )
 
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no F1 improvement for {patience} epochs)")
            break
 
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pth')}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument("--data_root", type=str, default="data/patches")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
 
    train(**vars(args))