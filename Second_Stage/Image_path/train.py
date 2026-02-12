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
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset import create_dataloaders
from image_encoder import (
    ImageEncoder,
    ImageClassifier,
)


# ── Utilities ────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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

    optimizer = AdamW(
        model.parameters(), lr=config.stage1_lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler(enabled=config.fp16)

    best_acc = 0.0
    for epoch in range(config.stage1_epochs):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            with autocast(enabled=config.fp16):
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



# ── Main ─────────────────────────────────────────────────────────────────

def main():
    pass
if __name__ == "__main__":
    main()