"""
Full Multimodal Classifier: Image Encoder + Text Encoder + Fusion + Classification Head.
"""

import torch
import torch.nn as nn

from config import Config
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion import build_fusion


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal pipeline:
        Image → ImageEncoder → x_img ──┐
                                        ├──► Fusion → x_fused → Classifier → y_score
        OCR Text → TextEncoder → x_txt ─┘

    Supports graceful degradation when OCR text is missing.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── Encoders ─────────────────────────────────────────────────────
        self.image_encoder = ImageEncoder(
            backbone_name=config.image_backbone,
            embed_dim=config.image_embed_dim,
            pretrained=config.image_pretrained,
            dropout=config.image_dropout,
        )

        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            embed_dim=config.text_embed_dim,
            dropout=config.text_dropout,
            freeze_layers=config.freeze_text_layers,
        )

        # ── Fusion ───────────────────────────────────────────────────────
        self.fusion = build_fusion(
            strategy=config.fusion_strategy,
            img_dim=config.image_embed_dim,
            txt_dim=config.text_embed_dim,
            fused_dim=config.fused_dim,
            dropout=config.fusion_dropout,
            num_heads=config.cross_attention_heads,
        )

        # ── Classification Head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(config.fused_dim, config.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[MultimodalClassifier] Total params: {total_params:,}")
        print(f"[MultimodalClassifier] Trainable params: {trainable_params:,}")

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image:          [B, 3, H, W]
            input_ids:      [B, seq_len]
            attention_mask:  [B, seq_len]

        Returns:
            logits: [B, num_classes]
        """
        # Encode both modalities
        x_img = self.image_encoder(image)                          # [B, image_embed_dim]
        x_txt = self.text_encoder(input_ids, attention_mask)       # [B, text_embed_dim]

        # Fuse
        x_fused = self.fusion(x_img, x_txt)                       # [B, fused_dim]

        # Classify
        logits = self.classifier(x_fused)                          # [B, num_classes]
        return logits

    def get_embeddings(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Return intermediate embeddings for analysis/visualization."""
        x_img = self.image_encoder(image)
        x_txt = self.text_encoder(input_ids, attention_mask)
        x_fused = self.fusion(x_img, x_txt)
        return {"x_img": x_img, "x_txt": x_txt, "x_fused": x_fused}


def build_model(config: Config) -> MultimodalClassifier:
    """Build the full multimodal model from config."""
    return MultimodalClassifier(config)
