import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import aleatoric_loss, compute_uncertainty, cross_entropy_loss


class PureModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0.3):
        super(PureModel, self).__init__()
        self.model = model(num_classes=num_classes, dropout=dropout, monte_carlo=False)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output = self((image, audio, text))
        # import matplotlib.pyplot as plt
        # import numpy as np
        # print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
        # print(image.shape)
        # img = image[0].detach().cpu().numpy()  # 变为 numpy 数组
        # img = np.transpose(img, (1, 2, 0))  # 变换维度 [C, H, W] -> [H, W, C]
        # plt.imshow(img)
        # plt.axis("off")  # 关闭坐标轴
        # plt.title("Sample Image")
        # plt.show()
        loss = cross_entropy_loss(output, target)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def val_test_shared_step(self, batch):
        image, audio, text, target = batch
        output = self((image, audio, text))
        loss = cross_entropy_loss(output, target)
        return loss, output, target

    def test_step(self, batch, batch_idx):
        loss, output, target = self.val_test_shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.val_test_shared_step(batch)
        self.val_acc(output, target)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
