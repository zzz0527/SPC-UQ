import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class ImagenetVGG16(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, temp=1.0):
        super().__init__()

        base_model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)

        # VGG16 的 feature 提取部分
        self.features = base_model.features  # 即卷积层和池化层

        # Adaptive pooling 保证输入为任意尺寸时输出为固定形状
        self.avgpool = base_model.avgpool

        # 替换分类层
        self.classifier = nn.Linear(4096, num_classes)
        self.temp = temp
        self.feature = None

        # 提取前置全连接层
        self.fc_pre = nn.Sequential(*list(base_model.classifier.children())[:-1])  # 去掉最后一层

        # 如果使用预训练，则加载最后一层参数
        if pretrained:
            self.classifier.load_state_dict(
                {'weight': base_model.classifier[-1].weight, 'bias': base_model.classifier[-1].bias}
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_pre(x)  # 提取 VGG16 原始的前置特征
        self.feature = x.clone().detach()
        if self.temp == 1:
            x = self.classifier(x)
        else:
            x = self.classifier(x) / self.temp
        return x


def imagenet_vgg16(temp=1.0, pretrained=True, **kwargs):
    model = ImagenetVGG16(pretrained=pretrained, temp=temp, **kwargs)
    return model
