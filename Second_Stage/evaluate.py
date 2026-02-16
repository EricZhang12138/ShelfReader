"""
Evaluation script for the trained multimodal model.

Outputs:
  - Top-1 and Top-5 accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (saved as image)
  - Analysis of fusion gate values (for gated fusion)
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from dataset import RPCMultimodalDataset, get_val_transforms
from models import build_model


@torch.no_grad()
def evaluate(model, dataloader, config):
    """Run full evaluation and collect predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"]

        with autocast(enabled=config.fp16):
            logits = model(images, input_ids, attention_mask)
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
    """Compute comprehensive metrics."""
    # Top-1
    top1_acc = 100.0 * (preds == labels).mean()

    # Top-5
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5_correct = np.array([labels[i] in top5_preds[i] for i in range(len(labels))])
    top5_acc = 100.0 * top5_correct.mean()

    # Per-class report
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Find worst-performing classes
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    worst_classes = np.argsort(per_class_acc)[:10]

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class_report": report,
        "confusion_matrix": cm,
        "worst_classes": worst_classes.tolist(),
        "per_class_accuracy": per_class_acc.tolist(),
    }


@torch.no_grad()
def analyze_modality_contributions(model, dataloader, config):
    """
    Analyze how much each modality contributes by comparing:
    1. Full multimodal performance
    2. Image-only (zero out text embeddings)
    3. Text-only (zero out image embeddings)
    """
    model.eval()
    results = {"full": 0, "image_only": 0, "text_only": 0, "total": 0}

    for batch in tqdm(dataloader, desc="Modality analysis"):
        images = batch["image"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["label"].to(config.device)

        # Full multimodal
        logits_full = model(images, input_ids, attention_mask)
        results["full"] += (logits_full.argmax(1) == labels).sum().item()

        # Image only: get embeddings, zero text
        emb = model.get_embeddings(images, input_ids, attention_mask)
        x_img = emb["x_img"]
        x_txt_zero = torch.zeros_like(emb["x_txt"])
        x_fused = model.fusion(x_img, x_txt_zero)
        logits_img = model.classifier(x_fused)
        results["image_only"] += (logits_img.argmax(1) == labels).sum().item()

        # Text only: zero image
        x_img_zero = torch.zeros_like(emb["x_img"])
        x_txt = emb["x_txt"]
        x_fused = model.fusion(x_img_zero, x_txt)
        logits_txt = model.classifier(x_fused)
        results["text_only"] += (logits_txt.argmax(1) == labels).sum().item()

        results["total"] += labels.size(0)

    for key in ["full", "image_only", "text_only"]:
        results[f"{key}_acc"] = 100.0 * results[key] / results["total"]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--analyze_modalities", action="store_true")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get("config", Config())
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.data_root = args.data_root
    config.ocr_cache_path = args.ocr_cache

    # Build model and load weights
    model = build_model(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create val dataloader
    with open(config.ocr_cache_path, "r") as f:
        ocr_cache = json.load(f)

    val_dataset = RPCMultimodalDataset(
        data_root=config.data_root,
        split="val2019",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_val_transforms(config),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    # Run evaluation
    preds, labels, probs = evaluate(model, val_loader, config)
    metrics = compute_metrics(preds, labels, probs, config.num_classes)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"\n  Worst performing classes (by accuracy):")
    for cls_idx in metrics["worst_classes"][:5]:
        acc = metrics["per_class_accuracy"][cls_idx]
        print(f"    Class {cls_idx}: {100*acc:.1f}%")

    # Optional: modality contribution analysis
    if args.analyze_modalities:
        print(f"\n{'='*50}")
        print(f"MODALITY CONTRIBUTION ANALYSIS")
        print(f"{'='*50}")
        mod_results = analyze_modality_contributions(model, val_loader, config)
        print(f"  Full multimodal:  {mod_results['full_acc']:.2f}%")
        print(f"  Image only:       {mod_results['image_only_acc']:.2f}%")
        print(f"  Text only:        {mod_results['text_only_acc']:.2f}%")
        improvement = mod_results["full_acc"] - mod_results["image_only_acc"]
        print(f"  Text contribution: +{improvement:.2f}% over image-only")


if __name__ == "__main__":
    main()
