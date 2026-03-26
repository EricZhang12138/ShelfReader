"""
Evaluate the image-only baseline model.

Usage:
    python evaluate_image_only.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --checkpoint checkpoints/best_model_image_only.pth
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from train_image_only import ImageOnlyClassifier


@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(config.device)
        labels = batch["label"]

        with autocast("cuda", enabled=config.fp16):
            logits = model(images)
            probs = F.softmax(logits, dim=-1)

        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )


def compute_metrics(preds, labels, probs, num_classes):
    top1_acc = 100.0 * (preds == labels).mean()

    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5_correct = np.array([labels[i] in top5_preds[i] for i in range(len(labels))])
    top5_acc = 100.0 * top5_correct.mean()

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    worst_classes = np.argsort(per_class_acc)[:10]

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "worst_classes": worst_classes.tolist(),
        "per_class_accuracy": per_class_acc.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache_paddle.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", Config())
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.data_root = args.data_root
    config.ocr_cache_path = args.ocr_cache

    model = ImageOnlyClassifier(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    _, _, test_loader = create_dataloaders(config)

    preds, labels, probs = evaluate(model, test_loader, config)
    metrics = compute_metrics(preds, labels, probs, config.num_classes)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (Image-Only Baseline)")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"\n  Worst performing classes (by accuracy):")
    for cls_idx in metrics["worst_classes"][:5]:
        acc = metrics["per_class_accuracy"][cls_idx]
        print(f"    Class {cls_idx}: {100*acc:.1f}%")


if __name__ == "__main__":
    main()
