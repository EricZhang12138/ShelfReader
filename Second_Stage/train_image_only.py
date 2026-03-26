"""
Train an image-only baseline for comparison against multimodal fusion models.

Uses the same ImageEncoder + classifier head as the multimodal pipeline,
but without any text input. This gives a clean baseline to measure
text contribution: fusion_acc - image_only_acc.

Usage:
    python train_image_only.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --ocr_cache ocr_cache_paddle.json
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
from dataset import create_dataloaders
from models.image_encoder import ImageEncoder


class ImageOnlyClassifier(nn.Module):
    """Image encoder + classification head (no text)."""

    def __init__(self, config: Config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone_name=config.image_backbone,
            embed_dim=config.image_embed_dim,
            pretrained=config.image_pretrained,
            dropout=config.image_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.image_embed_dim, config.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[ImageOnlyClassifier] Total params: {total:,}")
        print(f"[ImageOnlyClassifier] Trainable params: {trainable:,}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        return self.classifier(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
def evaluate(model, val_loader, criterion, config):
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0

    for batch in val_loader:
        images = batch["image"].to(config.device)
        labels = batch["label"].to(config.device)

        with autocast("cuda", enabled=config.fp16):
            logits = model(images)
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


def train(config, model, train_loader, val_loader):
    print("\n" + "=" * 60)
    print("TRAINING: Image-Only Baseline")
    print("=" * 60)

    model = model.to(config.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=config.fp16):
                use_mixup = config.use_mixup and random.random() < 0.5
                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, config.mixup_alpha
                    )
                    logits = model(images)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    preds = logits.argmax(1)
                    correct += (lam * (preds == labels_a).float()
                                + (1 - lam) * (preds == labels_b).float()).sum().item()
                else:
                    logits = model(images)
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

        val_acc, val_top5_acc, val_loss = evaluate(model, val_loader, criterion, config)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1}/{config.epochs} ({elapsed:.1f}s) | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val Top-5: {val_top5_acc:.2f}%"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5_acc,
            "lr": lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_top5_acc = val_top5_acc
            ckpt_path = os.path.join(config.checkpoint_dir, "best_model_image_only.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_top5_acc": val_top5_acc,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")

        if early_stopping(val_acc):
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc: {best_acc:.2f}% | Top-5: {best_top5_acc:.2f}%")
    print(f"{'='*60}")

    history_path = os.path.join(config.log_dir, "history_image_only.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history -> {history_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train image-only baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache_paddle.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
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

    # Create dataloaders (ocr_cache needed for dataset but text is ignored)
    train_loader, val_loader, _ = create_dataloaders(config)

    model = ImageOnlyClassifier(config)
    model = train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
