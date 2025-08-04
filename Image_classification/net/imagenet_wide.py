import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class ImagenetWideResNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, temp=1.0):
        super().__init__()

        base_model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT if pretrained else None)

        # Adapt to match your original WRN structure
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )

        self.linear = nn.Linear(2048, num_classes)
        self.temp = temp
        self.feature = None

        if pretrained:
            self.linear.load_state_dict(base_model.fc.state_dict())

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        self.feature = out.clone().detach()
        if self.temp == 1:
            out = self.linear(out)
        else:
            out = self.linear(out) / self.temp
        return out


def imagenet_wide(temp=1.0, pretrained=True, **kwargs):
    model = ImagenetWideResNet(pretrained=pretrained, temp=temp, **kwargs)
    return model