"""
Multimodal Feature Fusion Strategies.

Three options:
  1. ConcatFusion     — Simple concatenation + MLP projection
  2. GatedFusion      — Learned gate that weights each modality
  3. CrossAttention   — Transformer-style cross-attention between modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """
    Simplest fusion: concatenate image and text embeddings, then project.

    [x_img; x_txt] → Linear → ReLU → Dropout → x_fused
    """

    def __init__(self, img_dim: int, txt_dim: int, fused_dim: int, dropout: float = 0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(img_dim + txt_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_img, x_txt], dim=-1)  # [B, img_dim + txt_dim]
        return self.projection(combined)               # [B, fused_dim]


class GatedFusion(nn.Module):
    """
    Gated fusion: learns to dynamically weight the contribution of each modality.
    This is especially useful when one modality (e.g. OCR text) may be missing
    or uninformative for certain products (e.g. fruits without packaging).

    gate = σ(W_g · [x_img; x_txt] + b_g)
    x_fused = gate * x_img + (1 - gate) * x_txt
    """

    def __init__(self, img_dim: int, txt_dim: int, fused_dim: int, dropout: float = 0.3):
        super().__init__()

        # Project both modalities to same dimension first
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fused_dim),
            nn.ReLU(inplace=True),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

        # Final refinement
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        img_proj = self.img_proj(x_img)  # [B, fused_dim]
        txt_proj = self.txt_proj(x_txt)  # [B, fused_dim]

        # Compute gate
        combined = torch.cat([img_proj, txt_proj], dim=-1)  # [B, fused_dim * 2]
        g = self.gate(combined)  # [B, fused_dim], values in [0, 1]

        # Weighted combination
        fused = g * img_proj + (1 - g) * txt_proj  # [B, fused_dim]
        return self.output(fused)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: image attends to text and vice versa,
    then combines the attended representations.

    This is the most expressive fusion but also the most compute-heavy.
    Works best when both modalities carry rich, complementary information.
    """

    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        fused_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project to common dimension
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.txt_proj = nn.Linear(txt_dim, fused_dim)

        # Cross-attention: image queries attend to text keys/values
        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: text queries attend to image keys/values
        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Combine the two attended representations
        self.output = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        # Project to common space
        img = self.img_proj(x_img).unsqueeze(1)  # [B, 1, fused_dim]
        txt = self.txt_proj(x_txt).unsqueeze(1)  # [B, 1, fused_dim]

        # Cross-attention in both directions
        img_attended, _ = self.img_to_txt_attn(
            query=img, key=txt, value=txt
        )  # [B, 1, fused_dim]
        txt_attended, _ = self.txt_to_img_attn(
            query=txt, key=img, value=img
        )  # [B, 1, fused_dim]

        # Squeeze and combine
        img_attended = img_attended.squeeze(1)  # [B, fused_dim]
        txt_attended = txt_attended.squeeze(1)  # [B, fused_dim]

        combined = torch.cat([img_attended, txt_attended], dim=-1)  # [B, fused_dim * 2]
        return self.output(combined)  # [B, fused_dim]


def build_fusion(
    strategy: str,
    img_dim: int,
    txt_dim: int,
    fused_dim: int,
    dropout: float = 0.3,
    num_heads: int = 4,
) -> nn.Module:
    """Factory function to build the specified fusion module."""
    strategies = {
        "concat": ConcatFusion,
        "gated": GatedFusion,
        "cross_attention": CrossAttentionFusion,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown fusion strategy '{strategy}'. Choose from: {list(strategies.keys())}")

    kwargs = dict(img_dim=img_dim, txt_dim=txt_dim, fused_dim=fused_dim, dropout=dropout)
    if strategy == "cross_attention":
        kwargs["num_heads"] = num_heads

    module = strategies[strategy](**kwargs)
    print(f"[Fusion] Using {strategy} fusion: ({img_dim}, {txt_dim}) → {fused_dim}")
    return module
