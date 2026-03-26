"""
Score Fusion Classifier: Image CNN + Fuzzy OCR Text Scoring.

Instead of encoding text with a neural network, this architecture:
  1. Image path: ImageEncoder → classification head → image logits [B, 81]
  2. Text path:  OCR text → fuzzy string matching → text scores [B, 81]
  3. Fusion:     image_logits + learned_scale * text_scores → final logits

The text scores are precomputed (no gradient), so only the image path
and the fusion scale parameter are trained.
"""

import torch
import torch.nn as nn

from config import Config
from models.image_encoder import ImageEncoder


class ScoreFusionClassifier(nn.Module):
    """
    Complete score-fusion pipeline:
        Image → ImageEncoder → head → image_logits [B, C]
        OCR text → FuzzyTextScorer → text_scores [B, C]  (precomputed, passed in)
        Final = image_logits + text_scale * text_scores

    The text_scale parameter is learned end-to-end so the model
    discovers how much to trust the fuzzy OCR signal.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── Image Encoder ─────────────────────────────────────────────
        self.image_encoder = ImageEncoder(
            backbone_name=config.image_backbone,
            embed_dim=config.image_embed_dim,
            pretrained=config.image_pretrained,
            dropout=config.image_dropout,
        )

        # ── Classification Head (image features → class logits) ──────
        self.classifier = nn.Sequential(
            nn.Linear(config.image_embed_dim, config.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

        # ── Learned scale for fuzzy text scores ──────────────────────
        # Initialized to 0 so the model starts as a pure image classifier
        # and gradually learns to incorporate text if it helps.
        self.text_scale = nn.Parameter(torch.tensor(0.0))

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[ScoreFusionClassifier] Total params: {total_params:,}")
        print(f"[ScoreFusionClassifier] Trainable params: {trainable_params:,}")

    def forward(
        self,
        image: torch.Tensor,
        fuzzy_scores: torch.Tensor,
        has_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image:        [B, 3, H, W]
            fuzzy_scores: [B, num_classes] — precomputed fuzzy match scores in [0, 1]
            has_text:     [B] bool — whether each sample has OCR text.
                          If provided, text scores are zeroed out for samples
                          without OCR so the model relies purely on the image.

        Returns:
            logits: [B, num_classes]
        """
        # Image path
        img_features = self.image_encoder(image)       # [B, image_embed_dim]
        img_logits = self.classifier(img_features)     # [B, num_classes]

        # Mask out text scores for samples without OCR
        if has_text is not None:
            # has_text: [B] → [B, 1] for broadcasting against [B, num_classes]
            fuzzy_scores = fuzzy_scores * has_text.unsqueeze(1).float()

        # Combine: additive fusion with learned scale
        combined_logits = img_logits + self.text_scale * fuzzy_scores

        return combined_logits

    def get_text_scale(self) -> float:
        """Return the current learned text scale value (for logging)."""
        return self.text_scale.item()


def build_score_fusion_model(config: Config) -> ScoreFusionClassifier:
    """Build the score fusion model from config."""
    return ScoreFusionClassifier(config)
