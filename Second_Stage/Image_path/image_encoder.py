"""
Image Encoder: Pretrained backbone → embedding vector.
Using pretrained EfficientNet + training a new projection head
Supports EfficientNet-B4, ConvNeXt-Small, and ViT-Small via timm.
"""

import torch
import torch.nn as nn
import timm # model package, pytorch image models


class ImageEncoder(nn.Module):
    """
    Extracts a fixed-size embedding from a product image.

    Architecture:
        Pretrained backbone (feature extractor) → Global Average Pool → FC → ReLU → Dropout → x_img
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        embed_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load pretrained backbone from timm (removes classification head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, outputs pooled features
        )

        # Get backbone output dimension (freature vector)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]

        # Projection head: backbone features vector → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        print(f"[ImageEncoder] {backbone_name}: {backbone_dim} → {embed_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [B, 3, H, W]
        Returns:
            x_img: Image embedding [B, embed_dim]
        """
        features = self.backbone(x)       # [B, backbone_dim]
        x_img = self.projection(features)  # [B, embed_dim]    x_image represents the final embedding
        return x_img

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone parameters for staged training."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class ImageClassifier(nn.Module):
    """
    Standalone image classifier for Stage 1 pretraining.
    ImageEncoder + classification head.
    """

    def __init__(self, encoder: ImageEncoder, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
    
