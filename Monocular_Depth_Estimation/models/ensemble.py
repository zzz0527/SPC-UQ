import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DNormal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, padding='same', **kwargs)

    def forward(self, x):
        output = self.conv(x)
        mu, logsigma = torch.chunk(output, 2, dim=1)
        sigma = F.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=1)


class Single_UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, sigma=True):
        super(Single_UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder1 = conv_block(input_channels, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 128)
        self.encoder4 = conv_block(128, 256)
        self.encoder5 = conv_block(256, 512)

        self.upconv5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder6 = conv_block(512 + 256, 256)
        self.upconv6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder7 = conv_block(256 + 128, 128)
        self.upconv7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder8 = conv_block(128 + 64, 64)
        self.upconv8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder9 = conv_block(64 + 32, 32)

        if sigma:
            self.final = Conv2DNormal(32, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _crop(self, target, refer):
        _, _, h_t, w_t = target.size()
        _, _, h_r, w_r = refer.size()

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
        enc1 = self.encoder1(x)
        p1 = self.pool(enc1)

        enc2 = self.encoder2(p1)
        p2 = self.pool(enc2)

        enc3 = self.encoder3(p2)
        p3 = self.pool(enc3)

        enc4 = self.encoder4(p3)
        p4 = self.pool(enc4)

        enc5 = self.encoder5(p4)

        up5 = self.upconv5(enc5)
        dec6 = self.decoder6(torch.cat([up5, self._crop(enc4, up5)], dim=1))

        up6 = self.upconv6(dec6)
        dec7 = self.decoder7(torch.cat([up6, self._crop(enc3, up6)], dim=1))

        up7 = self.upconv7(dec7)
        dec8 = self.decoder8(torch.cat([up7, self._crop(enc2, up7)], dim=1))

        up8 = self.upconv8(dec8)
        dec9 = self.decoder9(torch.cat([up8, self._crop(enc1, up8)], dim=1))

        h, w = x.shape[2], x.shape[3]
        pad_h = h - dec9.shape[2]
        pad_w = w - dec9.shape[3]

        dec9_padded = F.pad(dec9, (pad_w // 2, pad_w - pad_w // 2,
                                   pad_h // 2, pad_h - pad_h // 2))

        return self.final(dec9_padded)

class Ensemble_UNet(nn.Module):

    def __init__(self, input_channels = 3, num_classes = 1, num_ensembles=5, sigma = True):
        super(Ensemble_UNet, self).__init__()
        self.num_ensembles = num_ensembles
        self.models = nn.ModuleList([
            Single_UNet(sigma=sigma) for _ in range(num_ensembles)
        ])

    def forward(self, x):
        mus = []
        sigmas = []
        for model in self.models:
            mu, sigma = model(x)
            mus.append(mu)
            sigmas.append(sigma)
        return mus, sigmas