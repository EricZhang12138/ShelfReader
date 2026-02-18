"""
Training script for the Multimodal Grocery Product Identification Pipeline.

Supports:
  - Staged training (pretrain encoders separately, then fine-tune end-to-end)
  - Direct end-to-end training
  - Mixed precision (FP16)
  - Cosine annealing with warmup
  - Mixup augmentation
  - Early stopping
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset import create_dataloaders
from models import (
    ImageEncoder,
    ImageClassifier,
    TextEncoder,
    TextClassifier,
    MultimodalClassifier,
    build_model,
)


# ── Utilities ────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # pytorch will then generate the same random number on all CPU 
    torch.cuda.manual_seed_all(seed) # pytorch will then generate the same random number on all GPU 
    torch.backends.cudnn.deterministic = True
"""
# Attempt 1: Only CPU seed
torch.manual_seed(42)
model = nn.Linear(10, 5).cuda()  # Weights initialized on GPU
# Uses GPU RNG - NOT seeded! ❌

# Attempt 2: Both seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
model = nn.Linear(10, 5).cuda()  # Weights initialized on GPU
# Uses GPU RNG - properly seeded! ✅
"""




# in one batch, you mix up the images in one batch
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
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
    
    """
    __call__ is a Python dunder (double underscore) method that lets you use an object like a function.
    When you define __call__ on a class, you can do this:
    early_stopping = EarlyStopping(patience=10)
    early_stopping(val_acc)  # This calls __call__
    """


# ── Stage 1: Pretrain Encoders Separately ────────────────────────────────


def train_image_encoder_stage1(
    config: Config, train_loader, val_loader
) -> ImageEncoder:
    """Pretrain the image encoder with its own classification head."""
    print("\n" + "=" * 60)
    print("STAGE 1a: Pretraining Image Encoder")
    print("=" * 60)

    encoder = ImageEncoder(
        backbone_name=config.image_backbone,
        embed_dim=config.image_embed_dim,
        pretrained=config.image_pretrained,
        dropout=config.image_dropout,
    ).to(config.device)

    model = ImageClassifier(encoder, config.num_classes).to(config.device)


    # AdamW, compared to SGD has two edges: 
    # 1. it has adaptive learning rate, meaning the learning rate adapts as training goes
    # If a parameter has consistently large gradients → reduce its effective learning rate / If a parameter has consistently small gradients → increase its effective learning rate
    # This prevents parameters from overshooting or moving too slowly
    # 2. it adds regularisation 
    # And in this case, the weight_decay is essentially just L2 regularisation 
    optimizer = AdamW(
        model.parameters(), lr=config.stage1_lr, weight_decay=config.weight_decay
    ) # the optimiser we use 

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)# softening target labels during training
    scaler = GradScaler("cuda", enabled=config.fp16)

    best_acc = 0.0
    for epoch in range(config.stage1_epochs):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=config.fp16):
                """
                logits = model(images) — runs the forward pass through the entire model. 
                The output logits is a tensor of shape [batch_size, num_classes]
                — raw scores for each class (e.g., 200 numbers per image, one per grocery product).
                """
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = 100.0 * correct / total

        # Validate
        val_acc = evaluate_image_model(model, val_loader, config)
        print(
            f"  [Image S1] Epoch {epoch+1}/{config.stage1_epochs} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"  [Image S1] Best Val Acc: {best_acc:.2f}%")
    return encoder


def train_text_encoder_stage1(
    config: Config, train_loader, val_loader
) -> TextEncoder:
    """Pretrain the text encoder with its own classification head."""
    print("\n" + "=" * 60)
    print("STAGE 1b: Pretraining Text Encoder")
    print("=" * 60)

    encoder = TextEncoder(
        model_name=config.text_model_name,
        embed_dim=config.text_embed_dim,
        dropout=config.text_dropout,
        freeze_layers=config.freeze_text_layers,
    ).to(config.device)

    model = TextClassifier(encoder, config.num_classes).to(config.device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler(enabled=config.fp16)

    best_acc = 0.0
    for epoch in range(config.stage1_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            with autocast(enabled=config.fp16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * input_ids.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += input_ids.size(0)

        train_acc = 100.0 * correct / total
        val_acc = evaluate_text_model(model, val_loader, config)
        print(
            f"  [Text S1] Epoch {epoch+1}/{config.stage1_epochs} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"  [Text S1] Best Val Acc: {best_acc:.2f}%")
    return encoder


@torch.no_grad()
def evaluate_image_model(model, val_loader, config):
    model.eval()
    correct, total = 0, 0
    for batch in val_loader:
        images = batch["image"].to(config.device)
        labels = batch["label"].to(config.device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_text_model(model, val_loader, config):
    model.eval()
    correct, total = 0, 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"].to(config.device)
        logits = model(input_ids, attention_mask)
        correct += (logits.argmax(1) == labels).sum().item()
        total += input_ids.size(0)
    return 100.0 * correct / total


# ── Stage 2 / Direct: End-to-End Multimodal Training ────────────────────


def train_multimodal(
    config: Config,
    model: MultimodalClassifier,
    train_loader,
    val_loader,
) -> MultimodalClassifier:
    """End-to-end multimodal training (Stage 2 or direct)."""
    print("\n" + "=" * 60)
    print("STAGE 2: End-to-End Multimodal Training")
    print("=" * 60)

    model = model.to(config.device)

    optimizer = AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": config.learning_rate * 0.1},
            {"params": model.text_encoder.parameters(), "lr": config.learning_rate * 0.1},
            {"params": model.fusion.parameters(), "lr": config.learning_rate},
            {"params": model.classifier.parameters(), "lr": config.learning_rate},
        ],
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
    best_acc = 0.0
    best_top5_acc = 0.0

    for epoch in range(config.epochs):
        start_time = time.time()

        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()

            with autocast(enabled=config.fp16):
                # Optional: Mixup on images only
                if config.use_mixup and random.random() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, config.mixup_alpha
                    )
                    logits = model(images, input_ids, attention_mask)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    # For accuracy tracking, use original labels
                    preds = logits.argmax(1)
                    correct += (lam * (preds == labels_a).float()
                                + (1 - lam) * (preds == labels_b).float()).sum().item()
                else:
                    logits = model(images, input_ids, attention_mask)
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
        val_acc, val_top5_acc, val_loss = evaluate_multimodal(
            model, val_loader, criterion, config
        )

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[-1]["lr"]

        print(
            f"  Epoch {epoch+1}/{config.epochs} ({elapsed:.1f}s) | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val Top-5: {val_top5_acc:.2f}%"
        )

        # ── Checkpointing ────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            best_top5_acc = val_top5_acc
            ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pth")
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
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        # ── Early Stopping ───────────────────────────────────────────
        if early_stopping(val_acc):
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc: {best_acc:.2f}% | Top-5: {best_top5_acc:.2f}%")
    print(f"{'='*60}")

    return model


@torch.no_grad()
def evaluate_multimodal(model, val_loader, criterion, config):
    """Evaluate multimodal model, return (top1_acc, top5_acc, avg_loss)."""
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0

    for batch in val_loader:
        images = batch["image"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"].to(config.device)

        with autocast(enabled=config.fp16):
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        # Top-5 accuracy
        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(1).sum().item()

        total += images.size(0)

    return (
        100.0 * correct / total,
        100.0 * correct_top5 / total,
        total_loss / total,
    )


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fusion", type=str, default=None, choices=["concat", "gated", "cross_attention"])
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--no_staged", action="store_true", help="Skip staged training")
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
    if args.fusion:
        config.fusion_strategy = args.fusion
    if args.backbone:
        config.image_backbone = args.backbone
    if args.no_staged:
        config.staged_training = False
    if args.device:
        config.device = args.device

    set_seed(config.seed)

    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
        config.fp16 = False

    print(f"Device: {config.device}")
    print(f"Fusion strategy: {config.fusion_strategy}")
    print(f"Image backbone: {config.image_backbone}")
    print(f"Staged training: {config.staged_training}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    if config.staged_training:
        # Stage 1: Pretrain encoders separately
        image_encoder = train_image_encoder_stage1(config, train_loader, val_loader)
        text_encoder = train_text_encoder_stage1(config, train_loader, val_loader)

        # Stage 2: Build full model with pretrained encoders
        model = build_model(config)
        model.image_encoder.load_state_dict(image_encoder.state_dict())
        model.text_encoder.load_state_dict(text_encoder.state_dict())
    else:
        model = build_model(config)

    # End-to-end training
    model = train_multimodal(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
