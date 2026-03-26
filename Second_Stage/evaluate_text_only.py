"""
Evaluate models on the subset of test samples that have OCR text.
Compares image-only baseline vs fusion models to show text contribution.

Usage:
    python evaluate_text_only.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --ocr_cache ocr_cache_combined.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from models import build_model
from train_image_only import ImageOnlyClassifier


@torch.no_grad()
def evaluate_filtered(model, dataloader, config, ocr_cache, text_only=True, is_image_only=False):
    """Evaluate on samples filtered by OCR text presence."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        images = batch["image"].to(config.device)
        labels = batch["label"]

        if is_image_only:
            with autocast("cuda", enabled=config.fp16):
                logits = model(images)
                probs = F.softmax(logits, dim=-1)
        else:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            with autocast("cuda", enabled=config.fp16):
                logits = model(images, input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)

        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    # Filter by OCR text presence using the dataset's sample list
    dataset = dataloader.dataset
    mask = []
    for rel_path, _ in dataset.samples:
        has_text = bool(ocr_cache.get(rel_path, "").strip())
        if text_only:
            mask.append(has_text)
        else:
            mask.append(not has_text)
    mask = np.array(mask)

    preds_f = preds[mask]
    labels_f = labels[mask]
    probs_f = probs[mask]

    top1 = 100.0 * (preds_f == labels_f).mean()
    top5_preds = np.argsort(probs_f, axis=1)[:, -5:]
    top5_correct = np.array([labels_f[i] in top5_preds[i] for i in range(len(labels_f))])
    top5 = 100.0 * top5_correct.mean()
    report = classification_report(labels_f, preds_f, output_dict=True, zero_division=0)

    return {
        "n_samples": int(mask.sum()),
        "top1": top1,
        "top5": top5,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache_combined.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.ocr_cache) as f:
        ocr_cache = json.load(f)

    ckpt_dir = "checkpoints"
    freeze = 2

    models_to_eval = [
        ("Image-Only", f"{ckpt_dir}/best_model_image_only.pth", True),
        ("Concat", f"{ckpt_dir}/best_model_concat_freeze{freeze}.pth", False),
        ("Gated", f"{ckpt_dir}/best_model_gated_freeze{freeze}.pth", False),
        ("Cross Attention", f"{ckpt_dir}/best_model_cross_attention_freeze{freeze}.pth", False),
    ]

    print(f"\n{'='*70}")
    print(f"EVALUATION: Text-only subset vs No-text subset")
    print(f"{'='*70}")

    for name, ckpt_path, is_image_only in models_to_eval:
        if not os.path.exists(ckpt_path):
            print(f"\n  {name}: checkpoint not found ({ckpt_path}), skipping")
            continue

        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        config = checkpoint.get("config", Config())
        config.device = args.device if torch.cuda.is_available() else "cpu"
        config.data_root = args.data_root
        config.ocr_cache_path = args.ocr_cache

        _, _, test_loader = create_dataloaders(config)

        if is_image_only:
            model = ImageOnlyClassifier(config).to(config.device)
        else:
            model = build_model(config).to(config.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        with_text = evaluate_filtered(model, test_loader, config, ocr_cache,
                                      text_only=True, is_image_only=is_image_only)
        no_text = evaluate_filtered(model, test_loader, config, ocr_cache,
                                    text_only=False, is_image_only=is_image_only)

        print(f"\n  --- {name} ---")
        print(f"  {'Subset':<15} {'N':>6} {'Top-1':>8} {'Top-5':>8} {'Macro F1':>10} {'Wt F1':>10}")
        print(f"  {'With OCR text':<15} {with_text['n_samples']:>6} {with_text['top1']:>7.2f}% {with_text['top5']:>7.2f}% {with_text['macro_f1']:>10.4f} {with_text['weighted_f1']:>10.4f}")
        print(f"  {'No OCR text':<15} {no_text['n_samples']:>6} {no_text['top1']:>7.2f}% {no_text['top5']:>7.2f}% {no_text['macro_f1']:>10.4f} {no_text['weighted_f1']:>10.4f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
