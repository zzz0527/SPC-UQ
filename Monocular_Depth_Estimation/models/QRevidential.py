import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DQNormalGamma(nn.Module):
    def __init__(self, num_quantiles):
        super().__init__()
        self.num_quantiles = num_quantiles

    def forward(self, x):
        mu, v, alpha, beta = torch.split(x, self.num_quantiles, dim=1)

        v = F.softplus(v)                   # v >= 0
        alpha = F.softplus(alpha) + 1.0     # alpha >= 1
        beta = F.softplus(beta)             # beta > 0

        return torch.cat([mu, v, alpha, beta], dim=1)  # [B, 4*num_quantiles, H, W]


class QRevidentialUNet(nn.Module):
    def __init__(self, input_channels=3, num_quantiles=3):
        super().__init__()
        self.num_quantiles = num_quantiles

        # -------- Encoder --------
        self.enc1 = self._block(input_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = self._block(256, 512)

        # -------- Decoder --------
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec6 = self._block(512 + 256, 256)
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec7 = self._block(256 + 128, 128)
        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec8 = self._block(128 + 64, 64)
        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec9 = self._block(64 + 32, 32)

        # -------- Final layers --------
        self.conv10 = nn.Conv2d(32, 4 * num_quantiles, kernel_size=1)
        self.evidential = Conv2DQNormalGamma(num_quantiles)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _crop(self, target, refer):
        _, _, h_t, w_t = target.size()
        _, _, h_r, w_r = refer.size()
        dh, dw = h_t - h_r, w_t - w_r
        if dh > 0:
            top = dh // 2
            bottom = dh - top
            target = target[:, :, top:-bottom, :]
        if dw > 0:
            left = dw // 2
            right = dw - left
            target = target[:, :, :, left:-right]
        return target

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # --- Decoder with skip connections ---
        d6 = self.dec6(torch.cat([self.up5(e5), self._crop(e4, self.up5(e5))], dim=1))
        d7 = self.dec7(torch.cat([self.up6(d6), self._crop(e3, self.up6(d6))], dim=1))
        d8 = self.dec8(torch.cat([self.up7(d7), self._crop(e2, self.up7(d7))], dim=1))
        d9 = self.dec9(torch.cat([self.up8(d8), self._crop(e1, self.up8(d8))], dim=1))

        # 如果输入尺寸与解码后略有不对齐，做 padding
        h, w = x.shape[2], x.shape[3]
        ph = h - d9.shape[2]
        pw = w - d9.shape[3]
        d9 = F.pad(d9,
                   (pw//2, pw - pw//2,
                    ph//2, ph - ph//2))

        out = self.conv10(d9)
        out = self.evidential(out)  # [B, 4*num_quantiles, H, W]

        return out