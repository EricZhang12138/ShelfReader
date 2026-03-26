"""
Run PaddleOCR v5 with German/Latin character support on all dataset images.
German language model handles Swedish characters (å, ä, ö) properly.

Merges with existing PaddleOCR cache, keeping the better result per image.

Usage:
    python run_ocr_paddlev5.py \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --old_cache ocr_cache_paddle.json \
        --output ocr_cache_combined.json
"""

import argparse
import json
import os
from pathlib import Path

from paddleocr import PaddleOCR


def run_paddlev5_on_dataset(data_root: str):
    """Run PaddleOCR v5 with German lang on all images."""
    ocr = PaddleOCR(use_textline_orientation=True, lang="german")
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
                result = ocr.predict(img_path)
                texts = []
                for r in result:
                    if "rec_texts" in r:
                        texts.extend(r["rec_texts"])
                text = " ".join(texts).strip()
            except Exception as e:
                print(f"  Error on {rel_path}: {e}")
                text = ""

            cache[rel_path] = text

            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(paths)} done")

        print(f"  [{split}] Done: {len(paths)} images")

    return cache


def merge_caches(old_cache, new_cache):
    """Merge old and new OCR results, preferring new (PaddleOCR v5) results."""
    all_keys = set(list(old_cache.keys()) + list(new_cache.keys()))
    merged = {}
    stats = {"new_only": 0, "old_only": 0, "new_wins": 0,
             "old_wins": 0, "both_empty": 0}

    for key in all_keys:
        old_text = old_cache.get(key, "").strip()
        new_text = new_cache.get(key, "").strip()

        if new_text and not old_text:
            merged[key] = new_text
            stats["new_only"] += 1
        elif old_text and not new_text:
            merged[key] = old_text
            stats["old_only"] += 1
        elif not old_text and not new_text:
            merged[key] = ""
            stats["both_empty"] += 1
        elif len(new_text) >= len(old_text):
            merged[key] = new_text
            stats["new_wins"] += 1
        else:
            merged[key] = old_text
            stats["old_wins"] += 1

    return merged, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--old_cache", type=str, default="ocr_cache_paddle.json")
    parser.add_argument("--output", type=str, default="ocr_cache_combined.json")
    parser.add_argument("--v5_cache", type=str, default="ocr_cache_paddlev5.json",
                        help="Save intermediate PaddleOCR v5 results here")
    args = parser.parse_args()

    if os.path.exists(args.v5_cache):
        print(f"Loading existing PaddleOCR v5 cache from {args.v5_cache}")
        with open(args.v5_cache) as f:
            new_cache = json.load(f)
    else:
        print("Running PaddleOCR v5 (German/Latin chars) on all images...")
        new_cache = run_paddlev5_on_dataset(args.data_root)
        with open(args.v5_cache, "w") as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=2)
        print(f"Saved PaddleOCR v5 cache to {args.v5_cache}")

    with open(args.old_cache) as f:
        old_cache = json.load(f)

    merged, stats = merge_caches(old_cache, new_cache)

    with open(args.output, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = len(merged)
    has_text = sum(1 for v in merged.values() if v.strip())
    old_had = sum(1 for v in old_cache.values() if v.strip())
    new_had = sum(1 for v in new_cache.values() if v.strip())

    print(f"\n{'='*50}")
    print(f"MERGE RESULTS")
    print(f"{'='*50}")
    print(f"  Old PaddleOCR had text:  {old_had}/{len(old_cache)} ({100*old_had/len(old_cache):.1f}%)")
    print(f"  PaddleOCR v5 had text:   {new_had}/{len(new_cache)} ({100*new_had/len(new_cache):.1f}%)")
    print(f"  Combined has text:       {has_text}/{total} ({100*has_text/total:.1f}%)")
    print(f"  New images with text:    +{has_text - old_had}")
    print(f"\n  Merge stats:")
    for k, v in stats.items():
        print(f"    {k:20s}: {v}")

    swedish = sum(1 for v in merged.values() if any(c in v for c in 'åäöÅÄÖ'))
    print(f"\n  Entries with Swedish chars (å/ä/ö): {swedish}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
