"""
Batch OCR Preprocessing for GroceryStoreDataset.

Extracts text from all product images and caches results to a JSON file
so OCR doesn't need to run during training.

Supports multiple OCR engines:
  - easyocr     : EasyOCR (pip install easyocr)
  - paddleocr   : PaddleOCR (pip install paddlepaddle paddleocr)
  - tesseract   : Tesseract via pytesseract (pip install pytesseract; apt install tesseract-ocr)

Usage:
    python ocr_preprocess.py --data_root /path/to/dataset --output ocr_cache.json --engine paddleocr
    python ocr_preprocess.py --data_root /path/to/dataset --output ocr_cache.json --engine easyocr --langs en sv
    python ocr_preprocess.py --data_root /path/to/dataset --output ocr_cache.json --engine tesseract --langs eng+swe

    # Merge a new engine's results into an existing cache (keeps non-empty entries from existing)
    python ocr_preprocess.py --data_root /path/to/dataset --output ocr_cache.json --engine paddleocr --merge
"""

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


# ── OCR Engine Interface ─────────────────────────────────────────────────


class OCREngine(ABC):
    """Base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image. Returns concatenated text string."""
        ...


class EasyOCREngine(OCREngine):
    def __init__(self, langs: List[str], gpu: bool = True, confidence_threshold: float = 0.3):
        import easyocr
        self.reader = easyocr.Reader(langs, gpu=gpu)
        self.confidence_threshold = confidence_threshold

    def extract_text(self, image_path: str) -> str:
        try:
            results = self.reader.readtext(image_path)
            if not results:
                return ""

            texts = []
            for (bbox, text, confidence) in results:
                if confidence > self.confidence_threshold and len(text.strip()) > 1:
                    texts.append(text.strip())

            return _deduplicate(texts)
        except Exception as e:
            print(f"  EasyOCR failed for {image_path}: {e}")
            return ""


class PaddleOCREngine(OCREngine):
    def __init__(self, langs: List[str], gpu: bool = True, confidence_threshold: float = 0.3):
        from paddleocr import PaddleOCR

        lang = self._map_lang(langs)
        # PaddleOCR v2.x API: use_angle_cls, lang, use_gpu, show_log
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=gpu, show_log=False)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _map_lang(langs: List[str]) -> str:
        """Map language list to PaddleOCR's single lang parameter."""
        lang_set = set(l.lower() for l in langs)

        # Direct PaddleOCR language names
        paddle_langs = {
            'ch', 'en', 'french', 'german', 'korean', 'japan',
            'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic',
            'cyrillic', 'devanagari',
        }
        for l in lang_set:
            if l in paddle_langs:
                return l

        # Map common codes
        mapping = {
            'sv': 'latin', 'swe': 'latin', 'swedish': 'latin',
            'en': 'en', 'eng': 'en', 'english': 'en',
            'ch_sim': 'ch', 'chinese': 'ch',
            'fr': 'french', 'de': 'german',
        }
        for l in lang_set:
            if l in mapping:
                return mapping[l]

        return 'en'

    def extract_text(self, image_path: str) -> str:
        try:
            # v2.x returns list of list of [bbox, (text, confidence)]
            results = self.ocr.ocr(image_path, cls=True)
            if not results or results[0] is None:
                return ""

            texts = []
            for line in results[0]:
                text, confidence = line[1]
                if confidence > self.confidence_threshold and len(text.strip()) > 1:
                    texts.append(text.strip())

            return _deduplicate(texts)
        except Exception as e:
            print(f"  PaddleOCR failed for {image_path}: {e}")
            return ""


class TesseractEngine(OCREngine):
    def __init__(self, langs: List[str], confidence_threshold: float = 30):
        import pytesseract
        self.pytesseract = pytesseract

        # Tesseract uses '+' separated lang codes like 'eng+swe'
        # Accept both ['eng', 'swe'] and ['eng+swe'] formats
        if len(langs) == 1 and '+' in langs[0]:
            self.lang = langs[0]
        else:
            lang_mapping = {
                'en': 'eng', 'sv': 'swe', 'swedish': 'swe',
                'ch_sim': 'chi_sim', 'chinese': 'chi_sim',
                'fr': 'fra', 'french': 'fra',
                'de': 'deu', 'german': 'deu',
            }
            mapped = [lang_mapping.get(l, l) for l in langs]
            self.lang = '+'.join(mapped)

        # Tesseract confidence is 0-100 (not 0-1)
        self.confidence_threshold = confidence_threshold

    def extract_text(self, image_path: str) -> str:
        from PIL import Image
        try:
            img = Image.open(image_path)
            # Use image_to_data for confidence filtering
            data = self.pytesseract.image_to_data(
                img, lang=self.lang, output_type=self.pytesseract.Output.DICT
            )

            texts = []
            for i, text in enumerate(data['text']):
                text = text.strip()
                conf = int(data['conf'][i])
                if conf > self.confidence_threshold and len(text) > 1:
                    texts.append(text)

            return _deduplicate(texts)
        except Exception as e:
            print(f"  Tesseract failed for {image_path}: {e}")
            return ""


# ── Helpers ──────────────────────────────────────────────────────────────


def _deduplicate(texts: List[str]) -> str:
    """Join texts, removing case-insensitive duplicates while preserving order."""
    seen = set()
    unique = []
    for t in texts:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)
    return " ".join(unique)


def build_engine(
    engine_name: str,
    langs: List[str],
    gpu: bool = True,
    confidence: float = 0.3,
) -> OCREngine:
    """Factory to build the requested OCR engine."""
    if engine_name == "easyocr":
        return EasyOCREngine(langs, gpu=gpu, confidence_threshold=confidence)
    elif engine_name == "paddleocr":
        return PaddleOCREngine(langs, gpu=gpu, confidence_threshold=confidence)
    elif engine_name == "tesseract":
        # Tesseract confidence is 0-100
        conf = confidence if confidence > 1 else confidence * 100
        return TesseractEngine(langs, confidence_threshold=conf)
    else:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Choose from: easyocr, paddleocr, tesseract"
        )


# ── Main Processing ─────────────────────────────────────────────────────


def process_dataset(
    data_root_str: str,
    output_path: str,
    engine: OCREngine,
    merge: bool = False,
) -> None:
    data_root = Path(data_root_str)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths: List[Path] = []
    for ext in image_extensions:
        image_paths.extend(data_root.rglob(f"*{ext}"))

    print(f"Found {len(image_paths)} images in {data_root}")

    # Optionally load existing cache to merge with
    existing_cache: Dict[str, str] = {}
    if merge and Path(output_path).exists():
        with open(output_path, "r") as f:
            existing_cache = json.load(f)
        print(f"Loaded existing cache with {len(existing_cache)} entries (merge mode)")

    ocr_cache: Dict[str, str] = {}
    empty_count = 0

    for img_path in tqdm(image_paths, desc="Extracting OCR"):
        rel_path = str(img_path.relative_to(data_root))

        # In merge mode, skip images that already have non-empty text
        if merge and existing_cache.get(rel_path, ""):
            ocr_cache[rel_path] = existing_cache[rel_path]
            continue

        text = engine.extract_text(str(img_path))
        ocr_cache[rel_path] = text
        if not text:
            empty_count += 1

    # In merge mode, also keep entries from existing cache not in current scan
    if merge:
        for key, val in existing_cache.items():
            if key not in ocr_cache:
                ocr_cache[key] = val

    with open(output_path, "w") as f:
        json.dump(ocr_cache, f, indent=2, ensure_ascii=False)

    total = len(ocr_cache)
    with_text = sum(1 for v in ocr_cache.values() if v)
    without_text = total - with_text

    print(f"\nOCR extraction complete:")
    print(f"  Total images: {total}")
    print(f"  Images with text: {with_text} ({100*with_text/total:.1f}%)")
    print(f"  Images without text: {without_text} ({100*without_text/total:.1f}%)")
    print(f"  Cache saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR preprocessing")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--output", type=str, default="ocr_cache.json",
                        help="Output JSON path")
    parser.add_argument("--engine", type=str, default="paddleocr",
                        choices=["easyocr", "paddleocr", "tesseract"],
                        help="OCR engine to use (default: paddleocr)")
    parser.add_argument("--langs", type=str, nargs="+", default=["en"],
                        help="Languages for OCR (default: en). "
                             "Examples: --langs en sv, --langs latin, --langs eng+swe")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Minimum confidence threshold (0-1 for easyocr/paddleocr, 0-100 for tesseract)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU for OCR")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing cache: only re-OCR images that had empty text")
    args = parser.parse_args()

    engine = build_engine(
        engine_name=args.engine,
        langs=args.langs,
        gpu=not args.no_gpu,
        confidence=args.confidence,
    )

    process_dataset(args.data_root, args.output, engine, merge=args.merge)
