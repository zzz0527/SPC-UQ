import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from pathlib import Path
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calibration_loss(y_pred, y_true, q, scale=True):
    diff = y_true - y_pred
    under_mask = (y_true <= y_pred)
    over_mask = ~under_mask
    coverage = torch.mean(under_mask.float())
    if coverage < q:
        loss = torch.mean(diff[over_mask])
    else:
        loss = torch.mean(-diff[under_mask])
    if scale:
        loss *= torch.abs(q - coverage)
    return loss

def pinball_loss(q_pred, target, tau):
    errors = target - q_pred
    loss = torch.max(tau * errors, (tau - 1) * errors)
    return torch.mean(loss)

class SPC:
    def __init__(self, model, dataset="", learning_rate=1e-3, tag="", lam=1e-3, l=0.5, drop_prob=0.1, sigma=False):
        self.model = model.to(device)
        self.l = l
        self.lam = lam

        self.reg_loss = nn.MSELoss()
        self.optimizer_all = optim.Adam(self.model.parameters(), lr=learning_rate)
        allow_keywords = ['hidden', 'mar', 'q_up', 'q_down']
        uq_params = [
            param for name, param in self.model.named_parameters()
            if any(k in name for k in allow_keywords)
        ]
        self.optimizer_uq = optim.Adam(uq_params, lr=learning_rate)

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save', f"{current_time}_{dataset}_{trainer}_{tag}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def uq_loss(self, y, pred, mar, mar_up, mar_down, q_up, q_down):
        # UQ-related losses
        residual = torch.abs(y - pred)
        diff = y - pred.detach()
        loss_mar = nn.MSELoss()(mar, residual)

        mask_up = (diff > 0)
        mask_down = (diff <= 0)
        loss_mar_up = nn.MSELoss()(mar_up[mask_up], (y[mask_up] - pred[mask_up]))
        loss_mar_down = nn.MSELoss()(mar_down[mask_down], (pred[mask_down] - y[mask_down]))

        loss_cali_up = calibration_loss(q_up[mask_up], (y[mask_up] - pred[mask_up]), 0.95)
        loss_cali_down = calibration_loss(q_down[mask_down], (pred[mask_down] - y[mask_down]), 0.95)

        loss = loss_mar + loss_mar_up + loss_mar_down + (loss_cali_up + loss_cali_down)
        return loss

    def run_train_step(self, x, y, stage='joint'):
        # Stage options: 'reg' (only regression), 'uq' (only UQ params), 'joint' (both)
        if stage == 'uq':
            allow_keywords = ['hidden', 'mar', 'q_up', 'q_down']
            for name, param in self.model.named_parameters():
                param.requires_grad = any(k in name for k in allow_keywords)

        x, y = x.to(device), y.to(device)
        self.model.train()
        pred, mar, mar_up, mar_down, q_up, q_down = self.model(x)

        if stage == 'reg':
            self.optimizer_all.zero_grad()
            loss = self.reg_loss(pred, y)
            loss.backward()
            self.optimizer_all.step()
        elif stage == 'uq':
            self.optimizer_uq.zero_grad()
            loss = self.uq_loss(y, pred, mar, mar_up, mar_down, q_up, q_down)
            loss.backward()
            self.optimizer_uq.step()
        else:
            self.optimizer_all.zero_grad()
            loss_pred = self.reg_loss(pred, y)
            loss_mar = self.uq_loss(y, pred, mar, mar_up, mar_down, q_up, q_down)
            loss = loss_pred + loss_mar
            loss.backward()
            self.optimizer_all.step()
        return loss.item(), pred

    def evaluate(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.eval()
        pred, mar, mar_up, mar_down, q_up, q_down = self.model(x)
        rmse = torch.sqrt(F.mse_loss(pred, y))
        return pred, rmse.item()

    def get_batch(self, x, y, batch_size):
        idx = np.sort(np.random.choice(x.shape[0], batch_size, replace=False))
        if isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            x_ = np.transpose(x[idx, ...], (0, 3, 1, 2))
            y_ = np.transpose(y[idx, ...], (0, 3, 1, 2))
            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0
            x_ = torch.tensor(x_ / x_divisor, dtype=torch.float32)
            y_ = torch.tensor(y_ / y_divisor, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown dataset type: {type(x)}, {type(y)}")
        return x_, y_

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{name}.pth"))

    def train(self, x_train, x_val, y_train, y_val, x_test, y_test, batch_size=128, iters=10000, verbose=True):
        tic = time.time()
        for self.iter in range(iters + 1):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, _ = self.run_train_step(x_input_batch, y_input_batch, stage='joint')
            if self.iter % 500 == 0:
                x_val_batch, y_val_batch = self.get_batch(x_val, y_val, min(100, y_val.shape[0]))
                _, rmse = self.evaluate(x_val_batch, y_val_batch)
                if verbose:
                    print(f"[{self.iter}] RMSE: {rmse:.4f} train_loss: {loss:.4f} t: {time.time() - tic:.2f} sec")
                tic = time.time()
        self.save("final")
        return self.model
