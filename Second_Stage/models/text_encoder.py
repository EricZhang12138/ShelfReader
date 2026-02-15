"""
Text Encoder: DistilBERT-based encoder for OCR-extracted text.

Takes tokenized OCR text and produces a fixed-size embedding.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):
    """
    Encodes OCR text into a fixed-size embedding using DistilBERT.

    Architecture:
        DistilBERT → [CLS] token → FC → ReLU → Dropout → x_txt
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 512,
        dropout: float = 0.2,
        freeze_layers: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.bert = DistilBertModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size  # 768 for distilbert-base

        # Freeze early transformer layers to save compute and prevent overfitting
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze first N transformer layers
            for i in range(min(freeze_layers, len(self.bert.transformer.layer))):
                for param in self.bert.transformer.layer[i].parameters():
                    param.requires_grad = False
            print(f"[TextEncoder] Froze embeddings + {freeze_layers} transformer layers")

        # Projection: BERT hidden → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        print(f"[TextEncoder] {model_name}: {bert_hidden} → {embed_dim}")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask:  [B, seq_len]
        Returns:
            x_txt: Text embedding [B, embed_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, bert_hidden]

        x_txt = self.projection(cls_output)  # [B, embed_dim]
        return x_txt

    def freeze_all(self, freeze: bool = True) -> None:
        """Freeze/unfreeze all BERT parameters."""
        for param in self.bert.parameters():
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
