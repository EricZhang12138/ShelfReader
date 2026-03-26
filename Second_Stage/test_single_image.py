"""
Run a single test image through all trained models and compare predictions.
"""

import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from config import Config
from dataset import get_val_transforms
from models import build_model
from models.fuzzy_scorer import FuzzyTextScorer
from models.score_fusion import ScoreFusionClassifier
from train_image_only import ImageOnlyClassifier


def load_class_names(csv_path):
    names = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[1].strip())] = row[0].strip()
    return names


def run_ocr(image_path):
    """Run PaddleOCR v5 on the image."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=True, lang="german")
    result = ocr.predict(image_path)
    texts = []
    for r in result:
        if "rec_texts" in r:
            texts.extend(r["rec_texts"])
    return " ".join(texts).strip()


def predict_image_only(image_tensor, config, class_names):
    ckpt = torch.load("checkpoints/best_model_image_only.pth", map_location=config.device, weights_only=False)
    saved_config = ckpt.get("config", config)
    saved_config.device = config.device
    model = ImageOnlyClassifier(saved_config).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad(), autocast("cuda", enabled=config.fp16):
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=-1)

    return get_top5(probs, class_names)


def predict_multimodal(image_tensor, input_ids, attention_mask, config, fusion, class_names):
    freeze = 2
    ckpt_path = f"checkpoints/best_model_{fusion}_freeze{freeze}.pth"
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    saved_config = ckpt.get("config", config)
    saved_config.device = config.device
    model = build_model(saved_config).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    with torch.no_grad(), autocast("cuda", enabled=config.fp16):
        logits = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)

    return get_top5(probs, class_names)


def predict_score_fusion(image_tensor, ocr_text, config, class_names, scorer):
    ckpt_path = "checkpoints/best_model_score_fusion_masked.pth"
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    saved_config = ckpt.get("config", config)
    saved_config.device = config.device
    model = ScoreFusionClassifier(saved_config).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fuzzy_scores = torch.from_numpy(scorer.score(ocr_text)).unsqueeze(0).to(config.device)
    has_text = torch.tensor([bool(ocr_text.strip())], dtype=torch.bool).to(config.device)

    with torch.no_grad(), autocast("cuda", enabled=config.fp16):
        logits = model(image_tensor, fuzzy_scores, has_text)
        probs = F.softmax(logits, dim=-1)

    return get_top5(probs, class_names)


def get_top5(probs, class_names):
    top5_probs, top5_idx = probs[0].topk(5)
    results = []
    for p, idx in zip(top5_probs, top5_idx):
        name = class_names.get(idx.item(), f"Class {idx.item()}")
        results.append((name, p.item()))
    return results


def main():
    image_path = "test_image/granny_smith_apples.png"
    data_root = "/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"

    config = Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.ocr_cache_path = "ocr_cache_combined.json"

    class_names = load_class_names(os.path.join(data_root, "classes.csv"))

    # Load and preprocess image
    transform = get_val_transforms(config)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(config.device)

    # Run OCR
    print(f"Image: {image_path}")
    print(f"Ground truth: Granny Smith apple")
    ocr_text = run_ocr(image_path)
    print(f"OCR text: \"{ocr_text}\"")

    # Tokenize OCR text
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    encoding = tokenizer(
        ocr_text if ocr_text else "[UNK]",
        max_length=config.max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(config.device)
    attention_mask = encoding["attention_mask"].to(config.device)

    # Fuzzy scorer
    scorer = FuzzyTextScorer.from_csv(os.path.join(data_root, "classes.csv"))

    print(f"\n{'='*60}")
    print(f"PREDICTIONS")
    print(f"{'='*60}")

    # Image-only
    print(f"\n  --- Image-Only Baseline ---")
    top5 = predict_image_only(image_tensor, config, class_names)
    for i, (name, prob) in enumerate(top5):
        marker = " <--" if "granny" in name.lower() else ""
        print(f"    {i+1}. {name:30s} {prob*100:5.1f}%{marker}")

    # Multimodal fusions
    for fusion in ["concat", "gated", "cross_attention"]:
        label = fusion.replace("_", " ").title()
        print(f"\n  --- {label} Fusion ---")
        top5 = predict_multimodal(image_tensor, input_ids, attention_mask, config, fusion, class_names)
        if top5 is None:
            print(f"    Checkpoint not found")
            continue
        for i, (name, prob) in enumerate(top5):
            marker = " <--" if "granny" in name.lower() else ""
            print(f"    {i+1}. {name:30s} {prob*100:5.1f}%{marker}")

    # Score fusion
    print(f"\n  --- Score Fusion (Fuzzy OCR) ---")
    top5 = predict_score_fusion(image_tensor, ocr_text, config, class_names, scorer)
    if top5:
        for i, (name, prob) in enumerate(top5):
            marker = " <--" if "granny" in name.lower() else ""
            print(f"    {i+1}. {name:30s} {prob*100:5.1f}%{marker}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
