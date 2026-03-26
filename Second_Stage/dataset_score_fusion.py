"""
Dataset for the Score Fusion architecture.

Returns (image, fuzzy_scores, label) instead of (image, tokenized_text, label).
Fuzzy scores are precomputed at dataset init time for efficiency.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config
from models.fuzzy_scorer import FuzzyTextScorer


class GroceryScoreFusionDataset(Dataset):
    """
    GroceryStoreDataset that returns (image_tensor, fuzzy_scores, label).

    Fuzzy scores are precomputed once at initialization by matching each
    sample's OCR text against all 81 class names.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        ocr_cache: Dict[str, str],
        scorer: FuzzyTextScorer,
        config: Config,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Parse split file
        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples: List[Tuple[str, int]] = []
        self.fuzzy_scores: List[np.ndarray] = []
        self.has_text: List[bool] = []

        ocr_hit = 0
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                rel_path = parts[0]
                fine_label = int(parts[1])
                if not (self.data_root / rel_path).exists():
                    continue

                self.samples.append((rel_path, fine_label))

                # Precompute fuzzy scores for this sample
                ocr_text = ocr_cache.get(rel_path, "")
                has_ocr = bool(ocr_text.strip())
                if has_ocr:
                    ocr_hit += 1
                self.has_text.append(has_ocr)
                self.fuzzy_scores.append(scorer.score(ocr_text))

        self.num_classes = max(label for _, label in self.samples) + 1
        print(f"[{split}] Loaded {len(self.samples)} samples, {self.num_classes} classes "
              f"({ocr_hit} with OCR text)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel_path, label = self.samples[idx]
        img_path = self.data_root / rel_path

        # Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "fuzzy_scores": torch.from_numpy(self.fuzzy_scores[idx]),
            "has_text": torch.tensor(self.has_text[idx], dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_score_fusion_dataloaders(
    config: Config,
    scorer: FuzzyTextScorer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders for score fusion."""
    from dataset import get_train_transforms, get_val_transforms

    with open(config.ocr_cache_path, "r") as f:
        ocr_cache = json.load(f)
    print(f"Loaded OCR cache with {len(ocr_cache)} entries")

    train_dataset = GroceryScoreFusionDataset(
        data_root=config.data_root,
        split="train",
        ocr_cache=ocr_cache,
        scorer=scorer,
        config=config,
        transform=get_train_transforms(config),
    )

    val_dataset = GroceryScoreFusionDataset(
        data_root=config.data_root,
        split="val",
        ocr_cache=ocr_cache,
        scorer=scorer,
        config=config,
        transform=get_val_transforms(config),
    )

    test_dataset = GroceryScoreFusionDataset(
        data_root=config.data_root,
        split="test",
        ocr_cache=ocr_cache,
        scorer=scorer,
        config=config,
        transform=get_val_transforms(config),
    )

    config.num_classes = max(train_dataset.num_classes, test_dataset.num_classes)
    print(f"[Train] {len(train_dataset)} samples")
    print(f"[Val]   {len(val_dataset)} samples")
    print(f"[Test]  {len(test_dataset)} samples")

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
