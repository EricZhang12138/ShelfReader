"""
Evaluation script for the Score Fusion model.

Usage:
    python evaluate_score_fusion.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --checkpoint checkpoints/best_model_score_fusion.pth \
        --ocr_cache ocr_cache.json
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from dataset_score_fusion import create_score_fusion_dataloaders
from models.fuzzy_scorer import FuzzyTextScorer
from models.score_fusion import ScoreFusionClassifier


@torch.no_grad()
def evaluate(model, dataloader, config):
    """Run full evaluation and collect predictions."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(config.device)
        fuzzy_scores = batch["fuzzy_scores"].to(config.device)
        has_text = batch["has_text"].to(config.device)
        labels = batch["label"]

        with autocast("cuda", enabled=config.fp16):
            logits = model(images, fuzzy_scores, has_text)
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


@torch.no_grad()
def analyze_modality_contributions(model, dataloader, config):
    """
    Compare performance with and without fuzzy text scores.
    """
    model.eval()
    results = {"full": 0, "image_only": 0, "text_only": 0, "total": 0}

    for batch in tqdm(dataloader, desc="Modality analysis"):
        images = batch["image"].to(config.device)
        fuzzy_scores = batch["fuzzy_scores"].to(config.device)
        has_text = batch["has_text"].to(config.device)
        labels = batch["label"].to(config.device)

        # Full model (image + text scores, masked by has_text)
        logits_full = model(images, fuzzy_scores, has_text)
        results["full"] += (logits_full.argmax(1) == labels).sum().item()

        # Image only (force all has_text to False)
        logits_img = model(images, fuzzy_scores, torch.zeros_like(has_text))
        results["image_only"] += (logits_img.argmax(1) == labels).sum().item()

        # Text only (use fuzzy scores directly as logits)
        results["text_only"] += (fuzzy_scores.argmax(1) == labels).sum().item()

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
    parser.add_argument("--classes_csv", type=str, default=None)
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", Config())
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.data_root = args.data_root
    config.ocr_cache_path = args.ocr_cache

    # Load class names
    classes_csv = args.classes_csv or os.path.join(config.data_root, "classes.csv")
    scorer = FuzzyTextScorer.from_csv(classes_csv)

    # Build model and load weights
    model = ScoreFusionClassifier(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Learned text_scale: {checkpoint.get('text_scale', model.get_text_scale()):.4f}")

    # Create test dataloader
    _, _, test_loader = create_score_fusion_dataloaders(config, scorer)

    # Evaluate
    preds, labels, probs = evaluate(model, test_loader, config)
    metrics = compute_metrics(preds, labels, probs, config.num_classes)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (Score Fusion)")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Text scale:      {model.get_text_scale():.4f}")
    print(f"\n  Worst performing classes (by accuracy):")
    for cls_idx in metrics["worst_classes"][:5]:
        acc = metrics["per_class_accuracy"][cls_idx]
        name = scorer.class_names_raw[cls_idx] if cls_idx < len(scorer.class_names_raw) else f"Class {cls_idx}"
        print(f"    {name} (ID {cls_idx}): {100*acc:.1f}%")

    # Modality analysis
    if args.analyze_modalities:
        print(f"\n{'='*50}")
        print(f"MODALITY CONTRIBUTION ANALYSIS")
        print(f"{'='*50}")
        mod_results = analyze_modality_contributions(model, test_loader, config)
        print(f"  Full (image + fuzzy text):  {mod_results['full_acc']:.2f}%")
        print(f"  Image only:                 {mod_results['image_only_acc']:.2f}%")
        print(f"  Fuzzy text only:            {mod_results['text_only_acc']:.2f}%")
        improvement = mod_results["full_acc"] - mod_results["image_only_acc"]
        print(f"  Text contribution: {improvement:+.2f}% over image-only")


if __name__ == "__main__":
    main()
