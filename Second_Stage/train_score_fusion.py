"""
Training script for the Score Fusion architecture.

This is a simpler pipeline than the multimodal BERT approach:
  - Image path: trainable CNN → class logits
  - Text path: precomputed fuzzy OCR scores (no learnable parameters)
  - Fusion: learned additive scale

Since the text path has no gradients, training is essentially training
an image classifier with an auxiliary fuzzy-text signal.

Usage:
    python train_score_fusion.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --ocr_cache ocr_cache.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset_score_fusion import create_score_fusion_dataloaders
from models.fuzzy_scorer import FuzzyTextScorer
from models.score_fusion import ScoreFusionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to images."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.should_stop


@torch.no_grad()
def evaluate(model, val_loader, criterion, config, mask_no_text=True):
    """Evaluate score fusion model, return (top1_acc, top5_acc, avg_loss)."""
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0

    for batch in val_loader:
        images = batch["image"].to(config.device)
        fuzzy_scores = batch["fuzzy_scores"].to(config.device)
        has_text = batch["has_text"].to(config.device) if mask_no_text else None
        labels = batch["label"].to(config.device)

        with autocast("cuda", enabled=config.fp16):
            logits = model(images, fuzzy_scores, has_text)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(1).sum().item()

        total += images.size(0)

    return (
        100.0 * correct / total,
        100.0 * correct_top5 / total,
        total_loss / total,
    )


def train(config, model, train_loader, val_loader, mask_no_text=True):
    """Train the score fusion model."""
    print("\n" + "=" * 60)
    print("TRAINING: Score Fusion (Image + Fuzzy OCR)")
    print("=" * 60)

    model = model.to(config.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Warmup + Cosine Annealing
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler(enabled=config.fp16)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    best_acc = 0.0
    best_top5_acc = 0.0
    history = []

    for epoch in range(config.epochs):
        start_time = time.time()

        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(config.device)
            fuzzy_scores = batch["fuzzy_scores"].to(config.device)
            has_text = batch["has_text"].to(config.device) if mask_no_text else None
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=config.fp16):
                # Optional mixup on images
                use_mixup = config.use_mixup and random.random() < 0.5
                if use_mixup:
                    images, labels_a, labels_b, lam, index = mixup_data(
                        images, labels, config.mixup_alpha
                    )
                    # Also mix the fuzzy scores and masks to stay consistent
                    fuzzy_scores = lam * fuzzy_scores + (1 - lam) * fuzzy_scores[index]
                    if has_text is not None:
                        has_text = has_text | has_text[index]  # keep text if either sample has it
                    logits = model(images, fuzzy_scores, has_text)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    preds = logits.argmax(1)
                    correct += (lam * (preds == labels_a).float()
                                + (1 - lam) * (preds == labels_b).float()).sum().item()
                else:
                    logits = model(images, fuzzy_scores, has_text)
                    loss = criterion(logits, labels)
                    correct += (logits.argmax(1) == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # ── Validate ─────────────────────────────────────────────────
        val_acc, val_top5_acc, val_loss = evaluate(model, val_loader, criterion, config, mask_no_text)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        text_scale = model.get_text_scale()

        print(
            f"  Epoch {epoch+1}/{config.epochs} ({elapsed:.1f}s) | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val Top-5: {val_top5_acc:.2f}% | "
            f"TextScale: {text_scale:.3f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5_acc,
            "lr": lr,
            "text_scale": text_scale,
        })

        # ── Checkpointing ────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            best_top5_acc = val_top5_acc
            tag = "masked" if mask_no_text else "unmasked"
            ckpt_path = os.path.join(config.checkpoint_dir, f"best_model_score_fusion_{tag}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_top5_acc": val_top5_acc,
                    "text_scale": text_scale,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%, TextScale: {text_scale:.3f})")

        # ── Early Stopping ───────────────────────────────────────────
        if early_stopping(val_acc):
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc: {best_acc:.2f}% | Top-5: {best_top5_acc:.2f}%")
    print(f"Final text_scale: {model.get_text_scale():.4f}")
    print(f"{'='*60}")

    tag = "masked" if mask_no_text else "unmasked"
    history_path = os.path.join(config.log_dir, f"history_score_fusion_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history -> {history_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Score Fusion model (Image + Fuzzy OCR)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classes_csv", type=str, default=None,
                        help="Path to classes.csv (default: data_root/classes.csv)")
    parser.add_argument("--no_mask", action="store_true",
                        help="Always apply OCR text scores, even for samples without OCR text")
    args = parser.parse_args()

    config = Config()
    config.data_root = args.data_root
    config.ocr_cache_path = args.ocr_cache
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.backbone:
        config.image_backbone = args.backbone
    if args.device:
        config.device = args.device

    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
        config.fp16 = False

    print(f"Device: {config.device}")
    print(f"Image backbone: {config.image_backbone}")

    # Load class names for fuzzy scoring
    classes_csv = args.classes_csv or os.path.join(config.data_root, "classes.csv")
    scorer = FuzzyTextScorer.from_csv(classes_csv)

    # Create dataloaders with precomputed fuzzy scores
    train_loader, val_loader, _ = create_score_fusion_dataloaders(config, scorer)

    # Build model
    model = ScoreFusionClassifier(config)

    # Train
    mask_no_text = not args.no_mask
    print(f"Mask no-text samples: {mask_no_text}")
    model = train(config, model, train_loader, val_loader, mask_no_text=mask_no_text)


if __name__ == "__main__":
    main()
