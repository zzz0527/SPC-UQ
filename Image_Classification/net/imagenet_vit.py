import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ImagenetViT(nn.Module):
    """
    ViT-B/16 wrapper with:
      - Optional pretrained backbone (torchvision)
      - Proper CLS token + positional embeddings usage
      - Temperature-scaled logits
      - Exposed CLS feature via `self.feature` (detached)
      - Optional backbone freezing for linear probing
    """

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        temp: float = 1.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)

        # Hidden size & final norm
        self.hidden_dim: int = self.backbone.hidden_dim
        self.norm: nn.Module = self.backbone.encoder.ln  # final LayerNorm

        # New classification head
        self.head: nn.Linear = nn.Linear(self.hidden_dim, num_classes)

        # If using pretrained and keeping 1000-class head, copy weights
        if pretrained and num_classes == 1000:
            with torch.no_grad():
                src = self.backbone.heads.head
                self.head.weight.copy_(src.weight)
                self.head.bias.copy_(src.bias)

        # Optionally freeze everything except the head
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Temperature buffer and CLS feature cache
        self.register_buffer("temperature", torch.tensor(float(temp)))
        self.feature: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
        """
        # Patchify + linear proj (handles resizing logic)
        x = self.backbone._process_input(x)  # (B, N, hidden_dim)

        # CLS token + positional embeddings (use pretrained params)
        n = x.shape[0]
        cls_token = self.backbone.class_token.expand(n, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat((cls_token, x), dim=1)                      # (B, N+1, hidden_dim)

        # Positional embeddings + dropout
        x = x + self.backbone.encoder.pos_embedding               # (B, N+1, hidden_dim)
        x = self.backbone.encoder.dropout(x)

        # Transformer encoder (layers) + final norm
        x = self.backbone.encoder.layers(x)
        x = self.norm(x)

        # CLS feature
        cls = x[:, 0]                     # (B, hidden_dim)
        self.feature = cls.detach()       # cache detached feature

        # Head + optional temperature scaling
        logits = self.head(cls)
        if float(self.temperature) != 1.0:
            logits = logits / self.temperature
        return logits


def imagenet_vit(
    temp: float = 1.0,
    pretrained: bool = True,
    num_classes: int = 1000,
    freeze_backbone: bool = False,
    **kwargs,
) -> ImagenetViT:
    """
    Factory for ImagenetViT.
    """
    return ImagenetViT(
        num_classes=num_classes,
        pretrained=pretrained,
        temp=temp,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )
