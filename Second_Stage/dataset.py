"""
Dataset loader for GroceryStoreDataset with multimodal support (image + OCR text).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from config import Config


class GroceryMultimodalDataset(Dataset):
    """
    GroceryStoreDataset that returns (image_tensor, tokenized_text, label) tuples.

    Expects the dataset layout:
        data_root/
            train.txt
            val.txt
            test.txt
            train/
                Fruit/Apple/Golden-Delicious/Golden-Delicious_001.jpg
                ...
            val/
                ...
            test/
                ...

    Each line in the split files: "relative/path/to/image.jpg, fine_label, coarse_label"
    OCR text is loaded from a precomputed JSON cache keyed by relative image path.
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

        self.tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_name)

        # Parse split file: each line is "rel_path, fine_label, coarse_label"
        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples: List[Tuple[str, int]] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                rel_path = parts[0]
                fine_label = int(parts[1])
                if (self.data_root / rel_path).exists():
                    self.samples.append((rel_path, fine_label))

        self.num_classes = max(label for _, label in self.samples) + 1
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


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders from the pre-defined dataset splits.

    GroceryStoreDataset provides train.txt, val.txt, and test.txt directly,
    so no manual splitting is needed.
    """
    with open(config.ocr_cache_path, "r") as f:
        ocr_cache = json.load(f)
    print(f"Loaded OCR cache with {len(ocr_cache)} entries")

    train_dataset = GroceryMultimodalDataset(
        data_root=config.data_root,
        split="train",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_train_transforms(config),
    )
    val_dataset = GroceryMultimodalDataset(
        data_root=config.data_root,
        split="val",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_val_transforms(config),
    )
    test_dataset = GroceryMultimodalDataset(
        data_root=config.data_root,
        split="test",
        ocr_cache=ocr_cache,
        config=config,
        transform=get_val_transforms(config),
    )

    # num_classes is derived from training data
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader