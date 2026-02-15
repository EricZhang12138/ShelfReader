# Multimodal Grocery Product Identification Pipeline

A two-stream (Image + OCR Text) multimodal architecture for grocery product classification,
designed for the **Retail Product Checkout (RPC)** dataset.

## Architecture

```
Product Image ──► Image Model (EfficientNet-B4) ──► x_img ──┐
                                                              ├──► Multimodal Fusion ──► Classifier ──► y_score
Product Image ──► OCR (PaddleOCR) ──► Text Model (DistilBERT) ──► x_txt ──┘
```

## Project Structure

```
multimodal_grocery/
├── config.py              # All hyperparameters and paths
├── dataset.py             # RPC dataset loader with OCR caching
├── models/
│   ├── image_encoder.py   # Image backbone (EfficientNet-B4 / ConvNeXt)
│   ├── text_encoder.py    # Text encoder (DistilBERT)
│   ├── fusion.py          # Multimodal fusion strategies
│   └── classifier.py      # Full multimodal model
├── ocr_preprocess.py      # Batch OCR extraction and caching
├── train.py               # Training loop
├── evaluate.py            # Evaluation and metrics
└── requirements.txt       # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Preprocess OCR
```bash
python ocr_preprocess.py --data_root /path/to/rpc --output ocr_cache.json
```

### Step 2: Train
```bash
python train.py --data_root /path/to/rpc --ocr_cache ocr_cache.json
```

### Step 3: Evaluate
```bash
python evaluate.py --data_root /path/to/rpc --checkpoint best_model.pth
```
