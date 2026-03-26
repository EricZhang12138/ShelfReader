# ShelfReader

A multimodal grocery product classification system that combines visual features with OCR-extracted text to identify products on store shelves. The main research and code lives in the [`Second_Stage/`](Second_Stage/) directory.

## Project Overview

ShelfReader explores how fusing image and text (OCR) modalities can improve fine-grained grocery product recognition. We compare four multimodal fusion strategies against an image-only baseline on the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset) (81 classes):

- **Image-only baseline** (MobileNetV3)
- **Concatenation fusion**
- **Gated fusion**
- **Cross-attention fusion**
- **Score-level fusion**

The image encoder extracts visual features via a CNN backbone, while PaddleOCR reads on-package text which is encoded by DistilBERT. The two streams are then combined through different fusion mechanisms and fed into a classifier.

## Repository Structure

```
ShelfReader/
├── Second_Stage/       # Main research code (training, evaluation, models)
├── First_Stage/        # Earlier prototyping / first-stage experiments
├── Figures/            # Plotting scripts and output figures
├── papers/             # LaTeX paper draft and references
└── examples/           # Example reports for reference
```

See [`Second_Stage/README.md`](Second_Stage/README.md) for detailed usage instructions (training, evaluation, configuration).
