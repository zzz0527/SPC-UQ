import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.models import vgg16, VGG16_Weights


class ImagenetVGG16(nn.Module):
    """
    VGG16 wrapper for ImageNet-like classification with:
      - Optional pretrained backbone
      - Optional feature freezing
      - Temperature-scaled logits
      - Exposes penultimate features via `self.feature` (detached)
    """

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        temp: float = 1.0,
        freeze_features: bool = False,
    ) -> None:
        super().__init__()

        # Load base model (weights imply specific preprocessing; handle in dataloader)
        base_model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)

        # Convolutional feature extractor and avgpool
        self.features: nn.Module = base_model.features
        self.avgpool: nn.Module = base_model.avgpool

        # Penultimate FC stack from original VGG16 (remove final classifier layer)
        # VGG16 classifier:
        # [Linear(25088->4096), ReLU, Dropout, Linear(4096->4096), ReLU, Dropout, Linear(4096->1000)]
        self.fc_pre: nn.Sequential = nn.Sequential(*list(base_model.classifier.children())[:-1])

        # New classification head
        self.classifier: nn.Linear = nn.Linear(4096, num_classes)

        # If using pretrained and num_classes matches 1000, copy the final layer weights
        if pretrained and num_classes == 1000:
            with torch.no_grad():
                self.classifier.weight.copy_(base_model.classifier[-1].weight)
                self.classifier.bias.copy_(base_model.classifier[-1].bias)

        # Optional: freeze convolutional features (+ fc_pre) for linear probing
        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.fc_pre.parameters():
                p.requires_grad = False

        # Temperature (applied to logits at inference/training)
        self.register_buffer("temperature", torch.tensor(float(temp)))
        self.feature: Optional[torch.Tensor] = None  # cached detached penultimate features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input batch of shape (B, 3, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_pre(x)

        # Cache penultimate features (detached) for downstream use
        self.feature = x.detach()

        logits = self.classifier(x)
        if self.temperature is not None and float(self.temperature) != 1.0:
            logits = logits / self.temperature
        return logits


def imagenet_vgg16(
    temp: float = 1.0,
    pretrained: bool = True,
    num_classes: int = 1000,
    freeze_features: bool = False,
    **kwargs,
) -> ImagenetVGG16:
    """
    Factory function for ImagenetVGG16.

    Args:
        temp: Temperature applied to logits (T=1.0 disables scaling).
        pretrained: Load pretrained ImageNet weights for the backbone.
        num_classes: Output classes for the final classifier.
        freeze_features: If True, freeze backbone + fc_pre for linear probing.
        **kwargs: Forwarded to the ImagenetVGG16 init (future-proof).

    Returns:
        Initialized ImagenetVGG16 model.
    """
    return ImagenetVGG16(
        num_classes=num_classes,
        pretrained=pretrained,
        temp=temp,
        freeze_features=freeze_features,
        **kwargs,
    )
