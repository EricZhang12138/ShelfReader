"""
Batch OCR Preprocessing for RPC Dataset.

Extracts text from all product images using EasyOCR and caches results
to a JSON file so OCR doesn't need to run during training.

Usage:
    python ocr_preprocess.py --data_root /path/to/rpc --output ocr_cache.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import easyocr
from tqdm import tqdm


def extract_text_from_image(reader, image_path: str) -> str:
    try:
        results = reader.readtext(image_path)
        if not results:
            return ""

        texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5 and len(text.strip()) > 1:
                texts.append(text.strip())

        seen = set()
        unique_texts = []
        for t in texts:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_texts.append(t)

        return " ".join(unique_texts)

    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""


def process_dataset(data_root_str: str, output_path: str) -> None:
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    data_root = Path(data_root_str)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths: List[Path] = []
    for ext in image_extensions:
        image_paths.extend(data_root.rglob(f"*{ext}"))

    print(f"Found {len(image_paths)} images in {data_root}")

    ocr_cache: Dict[str, str] = {}
    empty_count = 0

    for img_path in tqdm(image_paths, desc="Extracting OCR"):
        rel_path = str(img_path.relative_to(data_root))
        text = extract_text_from_image(reader, str(img_path))
        ocr_cache[rel_path] = text
        if not text:
            empty_count += 1

    with open(output_path, "w") as f:
        json.dump(ocr_cache, f, indent=2)

    print(f"\nOCR extraction complete:")
    print(f"  Total images: {len(ocr_cache)}")
    print(f"  Images with text: {len(ocr_cache) - empty_count}")
    print(f"  Images without text: {empty_count} ({100*empty_count/len(ocr_cache):.1f}%)")
    print(f"  Cache saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR for RPC dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to RPC dataset root")
    parser.add_argument("--output", type=str, default="ocr_cache.json", help="Output JSON path")
    args = parser.parse_args()

    process_dataset(args.data_root, args.output)