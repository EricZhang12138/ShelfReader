"""
Batch OCR Preprocessing for RPC Dataset.

Extracts text from all product images using PaddleOCR and caches results
to a JSON file so OCR doesn't need to run during training.

Usage:
    python ocr_preprocess.py --data_root /path/to/rpc --output ocr_cache.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from paddleocr import PaddleOCR
from tqdm import tqdm

# The code extracts all unique text with confidence > 0.5
# This function returns a space seperated string of texts from one single image 
def extract_text_from_image(ocr_engine: PaddleOCR, image_path: str) -> str:
    """
    Run OCR on a single image and return cleaned concatenated text.
    """
    try:
        results = ocr_engine.ocr(image_path, cls=True)
        if results is None or len(results) == 0:
            return ""

        texts = []
        for line in results:
            if line is None:
                continue
            for detection in line:
                # detection = [bbox, (text, confidence)]
                if detection and len(detection) >= 2:
                    text = detection[1][0]
                    confidence = detection[1][1]
                    if confidence > 0.5:  # Filter low-confidence detections
                        texts.append(text.strip())

        # Join all detected text, deduplicate while preserving order
        seen = set()
        unique_texts = []
        for t in texts:
            t_lower = t.lower()
            if t_lower not in seen and len(t) > 1:  # Skip single chars
                seen.add(t_lower)
                unique_texts.append(t)

        return " ".join(unique_texts)

    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""


def process_dataset(data_root_str: str, output_path: str) -> None:
    """
    Walk through the RPC dataset directory and extract OCR text for every image.
    Saves results as {relative_image_path: extracted_text}.
    """
    # Initialize PaddleOCR (downloads model on first run)
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False,
        use_gpu=True,
    )

    data_root = Path(data_root_str)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Collect all image paths
    image_paths: List[Path] = []
    for ext in image_extensions:
        image_paths.extend(data_root.rglob(f"*{ext}"))

    print(f"Found {len(image_paths)} images in {data_root}")

    # Extract OCR text
    # ocr_cache is a json file that stores all the 
    ocr_cache: Dict[str, str] = {}
    empty_count = 0

    # tqdm is for progress bar 
    for img_path in tqdm(image_paths, desc="Extracting OCR"):
        # convert absolute path to relative path
        rel_path = str(img_path.relative_to(data_root))
        text = extract_text_from_image(ocr, str(img_path))
        ocr_cache[rel_path] = text
        if not text:
            empty_count += 1

    # Save ocr_cache as a json file 
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
