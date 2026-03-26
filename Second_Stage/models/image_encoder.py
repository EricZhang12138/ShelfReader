"""
Image Encoder: Pretrained backbone → embedding vector.

Supports EfficientNet-B4, ConvNeXt-Small, and ViT-Small via timm.
"""

import torch
import torch.nn as nn
import timm # pytorch image models

"""
Summary of nn.Module:
A neural network layer is just weight tensors (matrices of numbers)
nn.Module tracks all those weights automatically when you assign layers with self.xxx = ...
This tracking lets you do things in one line that would otherwise be tedious:
model.parameters() → gives all weights to the optimizer
model.to("cuda") → moves all weights to GPU
model.state_dict() → saves all weights to a file
You only write forward() to define how data flows through the layers
PyTorch's autograd handles the backward pass automatically
Everything (weights + input data) must be on the same device (CPU or GPU) for computation to work
"""
"""
When you assign any nn.Module (like nn.Sequential, nn.Linear, or a custom module) to self.something in your __init__, PyTorch automatically handles:
✅ Parameter tracking
✅ Gradient computation
✅ Optimizer updates
✅ GPU movement
✅ Save/load
"""

# Pytorch uses nn.Module as the standard way to build neural network components
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

        # Load pretrained backbone from timm (removes classification head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, outputs pooled features
        )
        self.embed_dim = embed_dim

        # Get backbone output dimensions
        with torch.no_grad(): #torch.no_grad() tells PyTorch not to track gradients for operations inside this block.
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]
            spatial_features = self.backbone.forward_features(dummy)
            if spatial_features.dim() == 4:
                spatial_dim = spatial_features.shape[1]
            else:
                spatial_dim = spatial_features.shape[-1]

        # Projection head: pooled backbone features → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        # Spatial projection: unpooled spatial features → embed_dim (for cross-attention)
        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        print(f"[ImageEncoder] {backbone_name}: {backbone_dim} → {embed_dim} (spatial: {spatial_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [B, 3, H, W]
        Returns:
            x_img: Image embedding [B, embed_dim]
        """
        features = self.backbone(x)       # [B, backbone_dim]
        x_img = self.projection(features)  # [B, embed_dim]
        return x_img

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatial feature sequence (before global pooling) for cross-attention.

        Args:
            x: Image tensor [B, 3, H, W]
        Returns:
            x_seq: Spatial features [B, H*W, embed_dim]
        """
        features = self.backbone.forward_features(x)  # [B, C, H, W] for CNNs
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        # features is now [B, num_tokens, spatial_dim]
        x_seq = self.spatial_projection(features)  # [B, num_tokens, embed_dim]
        return x_seq

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone parameters for staged training."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class ImageClassifier(nn.Module):
    """
    Standalone image classifier for Stage 1 pretraining.
    ImageEncoder + classification head.

    Trains both the pooled path (projection) and the spatial path
    (spatial_projection) so that cross-attention fusion can use
    pretrained spatial features in Stage 2.
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
        # Auxiliary head for spatial path — ensures spatial_projection gets gradients
        self.spatial_head = nn.Sequential(
            nn.Linear(encoder.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primary path: pooled features
        pooled = self.encoder(x)
        logits = self.head(pooled)

        if self.training:
            # Auxiliary path: spatial features → mean pool → classify
            # This trains spatial_projection alongside the main path
            spatial_seq = self.encoder.forward_seq(x)       # [B, H*W, embed_dim]
            spatial_pooled = spatial_seq.mean(dim=1)         # [B, embed_dim]
            aux_logits = self.spatial_head(spatial_pooled)   # [B, num_classes]
            # Combine losses (main + 0.5 * auxiliary)
            return logits, aux_logits

        return logits
