import numpy as np
import time
import os
from pathlib import Path
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .utils import interval_score, ece_pi
from torch.utils.data import random_split, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.05
TAU = 1.0 - ALPHA


def pinball_loss(q_pred, target, tau):
    """
    Compute pinball (quantile) loss.
    """
    errors = target - q_pred
    loss = torch.max(tau * errors, (tau - 1) * errors)
    return torch.mean(loss)


def calibration_loss(y_pred, y_true, quantile, scale=True):
    """
    Coverage-based calibration loss.
    Penalize under- or over-coverage based on expected quantile.
    """
    diff = y_true - y_pred
    under_mask = (y_true <= y_pred)
    over_mask = ~under_mask
    coverage = torch.mean(under_mask.float())

    if coverage < quantile:
        loss = torch.mean(diff[over_mask])
    else:
        loss = torch.mean(-diff[under_mask])

    if scale:
        loss *= torch.abs(quantile - coverage)
    return loss


class SPC:
    def __init__(self, model, dataset="", noise="", tag="", learning_rate=1e-3,load_model=True, model_dir='save'):
        self.model = model.to(device)
        self.criterion = nn.MSELoss()

        # Optimizers for stagewise or joint training
        self.optimizer1 = optim.Adam(
            list(model.hidden.parameters()) + list(model.output1.parameters()), lr=learning_rate
        )
        self.optimizer2 = optim.Adam(
            list(model.hidden2.parameters()) + list(model.output2.parameters()), lr=learning_rate
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.load_model=load_model
        trainer = self.__class__.__name__
        save_subdir = f"{dataset}" if noise == '' else f"{dataset}_{noise}"
        self.save_dir = Path(model_dir) / save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{trainer}_{tag}.pth"

        self.calibration = False
        self.epoch = 0

    def uq_loss(self, y, preds, mar, mar_up, mar_down, q_up, q_down):
        """
        Multi-headed uncertainty and residual loss.
        """
        residual = torch.abs(y - preds)
        diff = y - preds.detach()

        loss_mar = self.criterion(mar, residual)

        mask_up = diff > 0
        mask_down = diff <= 0

        loss_mar_up = self.criterion(mar_up[mask_up], y[mask_up] - preds[mask_up])
        loss_mar_down = self.criterion(mar_down[mask_down], preds[mask_down] - y[mask_down])

        # loss_q_up = pinball_loss(q_up[mask_up], y[mask_up] - preds[mask_up], TAU)
        # loss_q_down = pinball_loss(q_down[mask_down], preds[mask_down] - y[mask_down], TAU)

        loss_cali_up = calibration_loss(q_up[mask_up], y[mask_up] - preds[mask_up], TAU)
        loss_cali_down = calibration_loss(q_down[mask_down], preds[mask_down] - y[mask_down], TAU)

        loss = loss_mar + loss_mar_up + loss_mar_down + loss_cali_up + loss_cali_down
        return loss

    def train_step_regression(self, x, y):
        self.model.train()
        preds, *_ = self.model(x)
        loss = self.criterion(preds, y)
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()
        return loss.item()

    def train_step_uncertainty(self, x, y):
        self.model.train()
        preds, mar, mar_up, mar_down, q_up, q_down = self.model(x)
        loss = self.uq_loss(y, preds, mar, mar_up, mar_down, q_up, q_down)
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()
        return loss.item(), mar, mar_up, mar_down

    def train_step_joint(self, x, y):
        self.model.train()
        preds, mar, mar_up, mar_down, q_up, q_down = self.model(x)
        loss_reg = self.criterion(preds, y)
        loss_uq = self.uq_loss(y, preds, mar, mar_up, mar_down, q_up, q_down)
        loss = loss_reg + loss_uq
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x, y, y_mu, y_scale):
        self.model.eval()
        tic=time.time()
        predictions, mar, mar_up, mar_down, q_up, q_down = self.model(x)
        aleatoric=(mar_up+mar_down)
        epistemic = torch.sqrt(abs(2 * mar_up * mar_down - mar * (mar_up + mar_down)))
        test_time = (time.time() - tic)* 1000

        # De-standardize predictions
        y = y.detach().cpu().numpy()
        predictions=predictions.detach().cpu().numpy()
        y = ((y * y_scale) + y_mu).squeeze()
        predictions = ((predictions * y_scale) + y_mu).squeeze()
        q_up = (q_up.detach().cpu().numpy()* y_scale).squeeze()
        q_down = (q_down.detach().cpu().numpy()* y_scale).squeeze()
        aleatoric = aleatoric.detach().cpu().numpy().squeeze()
        epistemic = epistemic.detach().cpu().numpy().squeeze()
        uncertainty_pred = aleatoric + epistemic

        #### Recalibration ####
        if self.calibration:
            d_up = (mar * mar_down) / ((2 * mar_down - mar) * mar_up)
            d_down = (mar * mar_up) / ((2 * mar_up - mar) * mar_down)
            d_up = np.clip(d_up, 1, None)
            d_down = np.clip(d_down, 1, None)
            q_up *= d_up
            q_down *= d_down


        ece=ece_pi(y, (predictions - q_down), (predictions + q_up), num_bins=10)

        is_score = interval_score(y, predictions, (predictions - q_down), (predictions + q_up), alpha=0.05)

        mpi_width = np.mean(q_up + q_down)

        within_interval = ((y >= (predictions - q_down)) & (y <= (predictions + q_up))).astype(float)
        picp = np.mean(within_interval)

        within_interval_down = (y[predictions>y] >= (predictions[predictions>y] - q_down[predictions>y])).astype(float)
        picp_down = np.mean(within_interval_down)

        within_interval_up = (y[predictions<y] <= (predictions[predictions<y] + q_up[predictions<y])).astype(float)
        picp_up = np.mean(within_interval_up)

        rmse = np.sqrt(np.mean((y - predictions) ** 2))

        err=(np.abs(y - predictions)).squeeze()
        Spearman_rmse, p_rmse = spearmanr(err, uncertainty_pred)

        return rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, test_time

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, train_dataset, test_dataset, y_mu, y_scale, batch_size=128, num_epochs=200, verbose=True):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        train_loss_curve = []
        test_rmse_curve = []
        train_time_log = []
        test_time_log = []

        print(self.save_path)

        if self.load_model and self.save_path.exists():
            self.model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        else:
            # Stage 1: Train mean predictor
            for self.epoch in range(1, num_epochs + 1):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    self.train_step_regression(data, target)

            for param in self.model.hidden.parameters():
                param.requires_grad = False
            for param in self.model.output1.parameters():
                param.requires_grad = False

            # Stage 2: Train uncertainty
            for self.epoch in range(1, num_epochs + 1):
                epoch_loss = 0.0
                tic = time.time()
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss, *_ = self.train_step_uncertainty(data, target)
                    epoch_loss += loss
                train_time_log.append((time.time() - tic) * 1000)
                train_loss_curve.append(epoch_loss / len(train_loader))

            self.save()

        # Final Evaluation
        for batch_idx, (data, target) in enumerate(test_loader):
            rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, test_time = self.evaluate(data, target, y_mu, y_scale)
            test_rmse_curve.append(rmse)
            test_time_log.append(test_time)


        if verbose:
            plt.figure(figsize=(10, 6))
            # plt.plot(uncertainty_curve, label='uncertainty', color='orange')
            plt.plot(train_loss_curve, label='loss', color='green')
            plt.legend()
            plt.title('loss and uncertainty in training')
            plt.show()

        return self.model, rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, np.mean(train_time_log), np.mean(test_time_log)