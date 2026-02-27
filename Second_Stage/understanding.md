# Project Overview

This is a **two-stream multimodal classifier** for grocery product identification (~200 SKUs) on the Retail Product Checkout (RPC) dataset. It fuses **visual features** (EfficientNet-B4) with **OCR-extracted text** (DistilBERT) to classify products.

---

## Key Fusion Logic (`models/fusion.py`)

You implement **three fusion strategies**, which is a great talking point:

1. **ConcatFusion** — Simply concatenates image + text embeddings, projects them down. Baseline approach.

2. **GatedFusion** (the default, and the most interesting to discuss):
   - Projects each modality to a shared dimension
   - Learns a **sigmoid gate** `g` from the concatenated projections
   - Output: `g * img_proj + (1 - g) * txt_proj`
   - **Why this matters**: Some products (fruits, bulk items) have *no text on them*. The gate learns to **dynamically downweight the text stream** when OCR is missing or noisy, and rely more on vision. This is the key design insight.

3. **CrossAttentionFusion** — Bidirectional transformer-style cross-attention (image attends to text, text attends to image). Most expressive but heavier.

---

## OCR Pipeline (`ocr_preprocess.py` + `dataset.py`)

- Uses **EasyOCR** (Chinese + English) to extract text from product images
- Filters by **confidence > 0.5** and **min length > 1 character**
- **Deduplicates** text case-insensitively (OCR often picks up the same word at different scales)
- Results are **pre-cached to JSON** (~4.5MB) so training doesn't pay the OCR cost per batch
- At training time, cached text is tokenized with DistilBERT's tokenizer (max 128 tokens)
- Missing text falls back to `"[UNK]"` — the gated fusion handles this gracefully

---

## What to Talk About in an Interview

### 1. The "Why" — Problem Motivation
> "Many grocery products look visually similar (e.g., different flavors of the same brand). Text on packaging is a strong discriminative signal, so I built a multimodal system that combines vision and OCR."

### 2. Staged Training Strategy (`train.py`)
- **Stage 1**: Train image encoder and text encoder *independently* with separate classification heads
- **Stage 2**: Load pretrained encoders, attach the fusion module + classifier, and train end-to-end with **10x lower LR for encoders** (1e-5 vs 1e-4) to avoid catastrophic forgetting
- This is a well-established technique worth explaining — it gives better convergence than training everything from scratch

### 3. Gated Fusion — The Star of the Show
> "Not every product has readable text. The gated fusion learns per-sample how much to trust each modality. For a banana, it relies on vision; for a cereal box, it can leverage OCR text too."

### 4. Training Tricks (shows depth of ML knowledge)
- **Mixup** applied only to the image branch (mixing text tokens would destroy semantics)
- **Label smoothing** (0.1) for the 200-class problem
- **Warmup + cosine annealing** LR schedule
- **Gradient clipping** (max_norm=1.0)
- **FP16 mixed precision** training

### 5. Evaluation — Modality Contribution Analysis (`evaluate.py`)
- You can **zero out one modality** and measure accuracy drop
- This quantifies how much each stream contributes — great for justifying the multimodal design

### 6. Architecture Choices to Justify

| Component | Choice | Why |
|-----------|--------|-----|
| Image backbone | EfficientNet-B4 | Good accuracy/efficiency tradeoff, pretrained on ImageNet |
| Text encoder | DistilBERT (4/6 layers frozen) | Lightweight; OCR text is short, doesn't need full BERT |
| Fusion | Gated | Handles missing text gracefully |
| Embed dim | 512 | Shared across both streams for clean fusion |

### 7. Potential Follow-up Questions to Prepare For
- "Why not CLIP?" — CLIP aligns image-text pairs; here you have OCR text *from* the image, not captions *about* the image. Different problem.
- "How do you handle no OCR text?" — `[UNK]` token + gated fusion learns to ignore it.
- "Why staged training?" — Random fusion on random encoders leads to poor gradients; pre-training gives a stable starting point.
- "What would you improve?" — Try PaddleOCR for better CJK support, experiment with cross-attention fusion, add data augmentation on text (e.g., random OCR noise).
