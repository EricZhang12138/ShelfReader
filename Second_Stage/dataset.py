"""
Dataset loader for RPC with multimodal support (image + OCR text).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from config import Config


class RPCMultimodalDataset(Dataset):
    """
    RPC Dataset that returns (image_tensor, tokenized_text, label) tuples.

    Expects COCO-format annotations:
        data_root/
            train2019/
                image1.jpg
                image2.jpg
                ...
            val2019/
                ...
            instances_train2019.json
            instances_val2019.json

    OCR text is loaded from a precomputed JSON cache.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        ocr_cache: Dict[str, str],
        config: Config,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.ocr_cache = ocr_cache
        self.config = config
        self.transform = transform

        # Initialize tokenizer for text branch
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_name)

        # Build image list and labels from COCO-format annotation file
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[int, int] = {}  # category_id -> contiguous index

        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load COCO annotation file
        ann_file = self.data_root / f"instances_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build category_id -> contiguous index mapping
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        for idx, cat in enumerate(categories):
            self.class_to_idx[cat["id"]] = idx

        # Build image_id -> filename mapping
        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        # Build image_id -> category_id mapping from annotations
        # RPC has one category per image, so we take the first annotation per image
        id_to_category = {}
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            if image_id not in id_to_category:
                id_to_category[image_id] = ann["category_id"]

        # Build samples list
        for image_id, category_id in id_to_category.items():
            filename = id_to_filename.get(image_id)
            if filename is None:
                continue
            rel_path = os.path.join(split, filename)
            label = self.class_to_idx[category_id]
            # Verify image exists
            if (self.data_root / rel_path).exists():
                self.samples.append((rel_path, label))

        self.num_classes = len(self.class_to_idx)
        print(f"[{split}] Loaded {len(self.samples)} samples, {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel_path, label = self.samples[idx]
        img_path = self.data_root / rel_path

        # ── Image ────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ── Text (from OCR cache) ────────────────────────────────────────
        ocr_text = self.ocr_cache.get(rel_path, "")

        # If no OCR text, use a placeholder so the text encoder still gets input
        if not ocr_text:
            ocr_text = "[UNK]"

        # Tokenize
        encoding = self.tokenizer(
            ocr_text,
            max_length=self.config.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Transforms ───────────────────────────────────────────────────────────


class Cutout:
    """Randomly mask out square patches from the image."""

    def __init__(self, n_holes: int = 1, length: int = 32):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones_like(img)

        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[:, y1:y2, x1:x2] = 0.0

        return img * mask


def get_train_transforms(config: Config) -> transforms.Compose:
    t = [
        transforms.Resize((config.image_size + 32, config.image_size + 32)),
        transforms.RandomCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if config.use_cutout:
        t.append(Cutout(n_holes=config.cutout_n_holes, length=config.cutout_length))

    return transforms.Compose(t)


def get_val_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── DataLoader Factory ───────────────────────────────────────────────────


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    # Load OCR cache
    with open(config.ocr_cache_path, "r") as f:
        ocr_cache = json.load(f)
    print(f"Loaded OCR cache with {len(ocr_cache)} entries")

    train_dataset = RPCMultimodalDataset(
        data_root=config.data_root,
        split="train2019",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_train_transforms(config),
    )

    val_dataset = RPCMultimodalDataset(
        data_root=config.data_root,
        split="val2019",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_val_transforms(config),
    )

    # Update config with actual number of classes
    config.num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader