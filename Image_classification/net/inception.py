"""
Pytorch implementation of InceptionV3 model (simplified).
Reference:
[1] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. CVPR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionBlock, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1, stride=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    def __init__(
        self,
        num_classes=10,
        temp=1.0,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        super(InceptionV3, self).__init__()
        self.feature = None

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        # simplified inception blocks
        self.inception5b = InceptionBlock(192, pool_features=32)
        self.inception5c = InceptionBlock(256, pool_features=64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        x = self.inception5b(x)
        x = self.inception5c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.feature = x.clone().detach()
        x = self.dropout(x)
        x = self.fc(x)

        return x


import torchvision.models as models

class MiniInception(nn.Module):
    def __init__(
        self,
        num_classes=10,
        temp=1.0,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        super(MiniInception, self).__init__()
        self.features = models.inception_v3(pretrained=False, aux_logits=True)

        # 修改第一个卷积层，使其适合小尺寸图片输入
        self.features.Conv2d_1a_3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # 替换全局平均池化，以适应小图片尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 替换分类器
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features.Conv2d_1a_3x3(x)
        x = self.features.Conv2d_2a_3x3(x)
        x = self.features.Conv2d_2b_3x3(x)
        x = self.features.maxpool1(x)
        x = self.features.Conv2d_3b_1x1(x)
        x = self.features.Conv2d_4a_3x3(x)
        x = self.features.maxpool2(x)
        x = self.features.Mixed_5b(x)
        x = self.features.Mixed_5c(x)
        x = self.features.Mixed_5d(x)
        x = self.features.Mixed_6a(x)
        x = self.features.Mixed_6b(x)
        x = self.features.Mixed_6c(x)
        x = self.features.Mixed_6d(x)
        x = self.features.Mixed_6e(x)
        x = self.features.Mixed_7a(x)
        x = self.features.Mixed_7b(x)
        x = self.features.Mixed_7c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.feature = x.clone().detach()
        if self.temp==1:
            out = self.fc(out)
        else:
            out = self.fc(out) / self.temp
        return out


class MiniInceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3):
        super(MiniInceptionModule, self).__init__()
        self.branch1 = BasicConv2d(in_channels, out1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out3x3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], dim=1)


class MiniInception32(nn.Module):
    def __init__(
        self,
        num_classes=10,
        temp=1.0,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        super(MiniInception32, self).__init__()

        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, padding=1),
            MiniInceptionModule(32, 32, 32),
            nn.MaxPool2d(2),  # 16x16

            MiniInceptionModule(64, 64, 64),
            nn.MaxPool2d(2),  # 8x8

            MiniInceptionModule(128, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        self.feature = x.clone().detach()
        x = self.classifier(x)
        return x


def inception_v3(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = MiniInception(**kwargs)
    return model
