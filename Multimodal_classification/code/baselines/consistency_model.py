import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import mar_loss, compute_uncertainty, softmax_entropy, mar_uncertainty, cross_entropy_loss


class SCModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0.0):
        super(SCModel, self).__init__()
        self.stage = 'cls'
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes, dropout=dropout, monte_carlo= False)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

    def set_stage(self, stage: str):
        assert stage in ['cls', 'mar']
        self.stage = stage

        if stage == 'cls':
            for param in self.model.parameters():
                param.requires_grad = True
        elif stage == 'mar':
            allow_keywords = ['mar']
            for name, param in self.model.named_parameters():
                if any(k in name for k in allow_keywords):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output = self((image, audio, text))
        # print(output)
        if self.stage == 'cls':
            loss = cross_entropy_loss(output[0], target)
        else:
            loss = mar_loss(output, target, self.num_classes)
        self.log('train_loss', loss)
        acc = self.train_acc(output[0], target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def val_test_shared_step(self, batch):
        image, audio, text, target = batch
        output = self((image, audio, text))
        if self.stage == 'cls':
            loss = cross_entropy_loss(output[0], target)
            uncertainty = torch.zeros_like(output[0]).sum(dim=-1)  # placeholder
        else:
            loss = mar_loss(output, target, self.num_classes)
            uncertainty = mar_uncertainty(output)

        entropy_ale = softmax_entropy(output)

        return loss, output, target, entropy_ale, uncertainty

    def test_step(self, batch, batch_idx):
        loss, output, target, entropy_ale, entropy_ep = self.val_test_shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(output[0], target)
        return loss, entropy_ale, entropy_ep

    def validation_step(self, batch, batch_idx):
        loss, output, target, entropy_ale, entropy_ep = self.val_test_shared_step(batch)
        self.val_acc(output[0], target)
        return loss, entropy_ale, entropy_ep

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', torch.stack([x[0] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('val_entropy_ale', torch.cat([x[1] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('val_entropy_epi', torch.cat([x[2] for x in outputs], dim=0).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_ale', torch.cat([x[1] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[2] for x in outputs], dim=0).mean(), prog_bar=True)
        self.aleatoric_uncertainties = torch.cat([x[1] for x in outputs], dim=0).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat([x[2] for x in outputs], dim=0).detach().cpu().numpy()

    def configure_optimizers(self):
        # lr = 1e-4
        if self.stage == 'cls':
            lr = 1e-2
        if self.model.__class__.__name__ == 'ImageMAR':
            lr = 1e-4
        if self.stage == 'mar':
            lr = 1e-2

        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
