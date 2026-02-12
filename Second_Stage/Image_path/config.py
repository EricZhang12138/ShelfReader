"""
Configuration for the Multimodal Grocery Product Identification Pipeline.
Adjust paths and hyperparameters here.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────
    data_root: str = "/path/to/rpc_dataset"
    ocr_cache_path: str = "ocr_cache.json"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Dataset ──────────────────────────────────────────────────────────
    num_classes: int = 200           # RPC has ~200 SKUs
    image_size: int = 224
    max_text_length: int = 128       # Max OCR token length

    # ── Image Encoder ────────────────────────────────────────────────────
    image_backbone: str = "efficientnet_b4"   # Options: efficientnet_b4, convnext_small, vit_small_patch16_224
    image_embed_dim: int = 512
    image_pretrained: bool = True
    image_dropout: float = 0.3

    # ── Text Encoder ─────────────────────────────────────────────────────
    text_model_name: str = "distilbert-base-uncased"
    text_embed_dim: int = 512
    text_dropout: float = 0.2
    freeze_text_layers: int = 4      # Freeze first N transformer layers

    # ── Fusion ───────────────────────────────────────────────────────────
    fusion_strategy: str = "gated"   # Options: concat, gated, cross_attention
    fused_dim: int = 512
    fusion_dropout: float = 0.3
    cross_attention_heads: int = 4   # Only used if fusion_strategy == "cross_attention"

    # ── Classifier Head ──────────────────────────────────────────────────
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.4

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1

    # Staged training: pretrain encoders separately before end-to-end
    staged_training: bool = True
    stage1_epochs: int = 15          # Epochs for individual encoder pretraining
    stage1_lr: float = 3e-4

    # ── Augmentation ─────────────────────────────────────────────────────
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 32

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True                # Mixed precision training
    save_top_k: int = 3
    early_stopping_patience: int = 10
