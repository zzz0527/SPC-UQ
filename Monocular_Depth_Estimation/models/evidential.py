import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DNormalGamma(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Split the input into four parameters
        mu, v, alpha, beta = torch.split(x, self.num_classes, dim=1)

        # Apply activation functions to ensure constraints
        v = F.softplus(v)   # v >= 0
        alpha = F.softplus(alpha) + 1.0  # alpha >= 1
        beta = F.softplus(beta)  # beta > 0

        # Concatenate the processed parameters
        output = torch.cat([mu, v, alpha, beta], dim=1)
        return output


class EvidentialUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoder1 = self._block(input_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = self._block(256, 512)

        # Decoder
        self.upconv5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder6 = self._block(512 + 256, 256)
        self.upconv6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder7 = self._block(256 + 128, 128)
        self.upconv7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder8 = self._block(128 + 64, 64)
        self.upconv8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder9 = self._block(64 + 32, 32)

        # Final layers
        self.conv10 = nn.Conv2d(32, 4 * num_classes, kernel_size=1)
        self.evidential = Conv2DNormalGamma(num_classes)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def _crop(self, target, refer):
        _, _, h_t, w_t = target.size()
        _, _, h_r, w_r = refer.size()

        # Calculate crop parameters
        delta_h = h_t - h_r
        delta_w = w_t - w_r

        if delta_h > 0:
            h_start = delta_h // 2
            h_end = delta_h - h_start
            target = target[:, :, h_start:-h_end, :]
        if delta_w > 0:
            w_start = delta_w // 2
            w_end = delta_w - w_start
            target = target[:, :, :, w_start:-w_end]

        return target

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        # Decoder with skip connections
        up5 = self.upconv5(enc5)
        dec6 = self.decoder6(torch.cat([up5, self._crop(enc4, up5)], dim=1))

        up6 = self.upconv6(dec6)
        dec7 = self.decoder7(torch.cat([up6, self._crop(enc3, up6)], dim=1))

        up7 = self.upconv7(dec7)
        dec8 = self.decoder8(torch.cat([up7, self._crop(enc2, up7)], dim=1))

        up8 = self.upconv8(dec8)
        dec9 = self.decoder9(torch.cat([up8, self._crop(enc1, up8)], dim=1))

        # Final padding and convolution
        h, w = x.shape[2], x.shape[3]
        pad_h = h - dec9.shape[2]
        pad_w = w - dec9.shape[3]

        dec9_padded = F.pad(dec9, (pad_w // 2, pad_w - pad_w // 2,
                                   pad_h // 2, pad_h - pad_h // 2))

        conv10 = self.conv10(dec9_padded)
        output = self.evidential(conv10)

        return output

