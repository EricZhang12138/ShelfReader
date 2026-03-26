"""
Text Encoder: Transformer-based encoder for OCR-extracted text.

Supports BERT, TinyBERT, DistilBERT, and multilingual variants via AutoModel.
Takes tokenized OCR text and produces a fixed-size embedding.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Encodes OCR text into a fixed-size embedding.

    Architecture:
        Transformer → [CLS] token → FC → ReLU → Dropout → x_txt
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        embed_dim: int = 512,
        dropout: float = 0.2,
        freeze_layers: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Detect model architecture for layer freezing
        # DistilBERT: model.embeddings + model.transformer.layer
        # BERT/TinyBERT: model.embeddings + model.encoder.layer
        if hasattr(self.transformer, "transformer"):
            layers = self.transformer.transformer.layer
        else:
            layers = self.transformer.encoder.layer

        # Freeze early transformer layers to save compute and prevent overfitting
        if freeze_layers > 0:
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = False
            for i in range(min(freeze_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
            print(f"[TextEncoder] Froze embeddings + {freeze_layers}/{len(layers)} transformer layers")

        # Projection: hidden → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        print(f"[TextEncoder] {model_name}: {hidden_size} → {embed_dim}")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x_txt = self.projection(cls_output)
        return x_txt

    def forward_seq(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        all_tokens = outputs.last_hidden_state
        x_seq = self.projection(all_tokens)
        return x_seq, attention_mask

    def freeze_all(self, freeze: bool = True) -> None:
        for param in self.transformer.parameters():
            param.requires_grad = not freeze


class TextClassifier(nn.Module):
    """
    Standalone text classifier for Stage 1 pretraining.
    TextEncoder + classification head.
    """

    def __init__(self, encoder: TextEncoder, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        features = self.encoder(input_ids, attention_mask)
        return self.head(features)
