"""
Run EasyOCR with Swedish+English on all dataset images, then merge
with existing PaddleOCR results to produce a combined cache.

For each image, keeps whichever result is better (longer text, or
the one with Swedish characters if applicable).

Usage:
    python run_ocr_easyocr.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --paddle_cache ocr_cache_paddle.json \
        --output ocr_cache_combined.json
"""

import argparse
import json
import os
from pathlib import Path

import easyocr


def run_easyocr_on_dataset(data_root: str, languages=("sv", "en")):
    """Run EasyOCR on all images across train/val/test splits."""
    reader = easyocr.Reader(list(languages), gpu=True)
    data_root = Path(data_root)
    cache = {}

    for split in ["train", "val", "test"]:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            print(f"Skipping {split} (no split file)")
            continue

        paths = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.split(",")[0].strip()
                img_path = data_root / rel_path
                if img_path.exists():
                    paths.append((rel_path, str(img_path)))

        print(f"[{split}] Processing {len(paths)} images...")
        for i, (rel_path, img_path) in enumerate(paths):
            try:
                results = reader.readtext(img_path)
                text = " ".join(r[1] for r in results).strip()
            except Exception as e:
                print(f"  Error on {rel_path}: {e}")
                text = ""

            cache[rel_path] = text

            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(paths)} done")

        print(f"  [{split}] Done: {len(paths)} images")

    return cache


def merge_caches(paddle_cache, easyocr_cache):
    """Merge PaddleOCR and EasyOCR results, keeping the better one per image."""
    all_keys = set(list(paddle_cache.keys()) + list(easyocr_cache.keys()))
    merged = {}
    stats = {"paddle_only": 0, "easyocr_only": 0, "easyocr_wins": 0,
             "paddle_wins": 0, "both_empty": 0}

    for key in all_keys:
        p_text = paddle_cache.get(key, "").strip()
        e_text = easyocr_cache.get(key, "").strip()

        if e_text and not p_text:
            merged[key] = e_text
            stats["easyocr_only"] += 1
        elif p_text and not e_text:
            merged[key] = p_text
            stats["paddle_only"] += 1
        elif not p_text and not e_text:
            merged[key] = ""
            stats["both_empty"] += 1
        elif len(e_text) >= len(p_text):
            merged[key] = e_text
            stats["easyocr_wins"] += 1
        else:
            merged[key] = p_text
            stats["paddle_wins"] += 1

    return merged, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--paddle_cache", type=str, default="ocr_cache_paddle.json")
    parser.add_argument("--output", type=str, default="ocr_cache_combined.json")
    parser.add_argument("--easyocr_cache", type=str, default="ocr_cache_easyocr.json")
    args = parser.parse_args()

    if os.path.exists(args.easyocr_cache):
        print(f"Loading existing EasyOCR cache from {args.easyocr_cache}")
        with open(args.easyocr_cache) as f:
            easyocr_cache = json.load(f)
    else:
        print("Running EasyOCR (Swedish + English) on all images...")
        easyocr_cache = run_easyocr_on_dataset(args.data_root)
        with open(args.easyocr_cache, "w") as f:
            json.dump(easyocr_cache, f, ensure_ascii=False, indent=2)
        print(f"Saved EasyOCR cache to {args.easyocr_cache}")

    with open(args.paddle_cache) as f:
        paddle_cache = json.load(f)

    merged, stats = merge_caches(paddle_cache, easyocr_cache)

    with open(args.output, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = len(merged)
    has_text = sum(1 for v in merged.values() if v.strip())
    paddle_had = sum(1 for v in paddle_cache.values() if v.strip())
    easyocr_had = sum(1 for v in easyocr_cache.values() if v.strip())

    print(f"\n{'='*50}")
    print(f"MERGE RESULTS")
    print(f"{'='*50}")
    print(f"  PaddleOCR had text:   {paddle_had}/{len(paddle_cache)} ({100*paddle_had/len(paddle_cache):.1f}%)")
    print(f"  EasyOCR had text:     {easyocr_had}/{len(easyocr_cache)} ({100*easyocr_had/len(easyocr_cache):.1f}%)")
    print(f"  Combined has text:    {has_text}/{total} ({100*has_text/total:.1f}%)")
    print(f"  New images with text: +{has_text - paddle_had}")
    print(f"\n  Merge stats:")
    for k, v in stats.items():
        print(f"    {k:20s}: {v}")

    swedish = sum(1 for v in merged.values() if any(c in v for c in 'åäöÅÄÖ'))
    print(f"\n  Entries with Swedish chars (å/ä/ö): {swedish}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
