import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ImagenetViT(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, temp=1.0):
        super().__init__()

        base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)

        # ViT 的 embedding 和 transformer 编码器部分
        self.features = nn.Sequential(
            base_model.conv_proj,     # patch embedding (conv stem)
            base_model.encoder,       # transformer encoder
        )

        # 提取 CLS token 对应的输出
        self.norm = base_model.encoder.ln
        self.hidden_dim = base_model.hidden_dim

        self.linear = nn.Linear(self.hidden_dim, num_classes)
        self.temp = temp
        self.feature = None

        if pretrained:
            self.linear.load_state_dict(base_model.heads.head.state_dict())

    def forward(self, x):
        # 输入为[B, 3, H, W]
        x = self.features[0](x)  # patch embedding: [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
        cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim, device=x.device))
        cls_tokens = cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # prepend cls token

        x = self.features[1](x)  # transformer encoder
        x = self.norm(x)  # final layer norm
        out = x[:, 0]     # 取 CLS token
        self.feature = out.clone().detach()

        if self.temp == 1:
            out = self.linear(out)
        else:
            out = self.linear(out) / self.temp
        return out


def imagenet_vit(temp=1.0, pretrained=True, **kwargs):
    model = ImagenetViT(pretrained=pretrained, temp=temp, **kwargs)
    return model
