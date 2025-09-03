import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import os
import time
import datetime
from pathlib import Path
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NIG_NLL(y, gamma, v, alpha, beta, w_i_mean, quantile, reduce=True):
    """
    Negative log-likelihood for Normal-Inverse-Gamma.
    """
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    twoBlambda = 4.0 * beta * (1.0 + tau_two * w_i_mean * v)

    nll = 0.5 * torch.log(math.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    """
    Kullback-Leibler divergence between two Normal-Inverse-Gamma distributions.
    """
    kl = 0.5 * (a1 - 1.0) / b1 * (v2 * (mu2 - mu1) ** 2) \
        + 0.5 * (v2 / v1) \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 \
        + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1

    return kl


def tilted_loss(q, e):
    """
    Tilted quantile loss.
    """
    return torch.maximum(q * e, (q - 1.0) * e)


def NIG_Reg(y, gamma, v, alpha, beta, w_i_mean, quantile, omega=0.01, reduce=True, use_kl=False):
    """
    Regularization term for NIG, can use KL or evidential penalty.
    """
    error = tilted_loss(quantile, y - gamma)
    if use_kl:
        kl_val = KL_NIG(gamma, v, alpha, beta,
                        gamma, omega, 1.0 + omega, beta)
        reg = error * kl_val
    else:
        evidence = 2.0 * v + alpha + 1.0 / beta
        reg = error * evidence
    return torch.mean(reg) if reduce else reg


def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    """
    Combined NLL and evidential regularization loss.
    """
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    w_i_mean = beta / (alpha - 1.0)
    mu = gamma + theta * w_i_mean

    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_mean, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_mean, quantile, reduce=reduce)

    return loss_nll + coeff * loss_reg


def quant_evi_loss_upt(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    """
    An alternative loss with additional error-based term.
    """
    loss_main = quant_evi_loss(y_true, gamma.detach(), v, alpha, beta, quantile,
                                coeff=coeff, reduce=False)
    error = tilted_loss(quantile, y_true - gamma)
    loss = loss_main + error
    return torch.sum(loss) if reduce else loss


class QRevidential:
    def __init__(self, model, dataset="", learning_rate=1e-3, lam=0.3, epsilon=1e-2, maxi_rate=1e-4, tag=""):
        self.model = model.to(device)
        self.nll_loss_function = NIG_NLL
        self.reg_loss_function = NIG_Reg
        self.quantiles = [0.025, 0.5, 0.975]
        self.num_quantiles = len(self.quantiles)
        self.coeff = 0.5
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch = 0

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save', f"{current_time}_{dataset}_{trainer}_{tag}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def loss_function(self, y, mu, v, alpha, beta):
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            loss += quant_evi_loss(
                y, mu[:, i].unsqueeze(1), v[:, i].unsqueeze(1),
                alpha[:, i].unsqueeze(1), beta[:, i].unsqueeze(1),
                q, coeff=self.coeff
            )
        return loss

    def run_train_step(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.train()
        y_hat = self.model(x)
        mu, v, alpha, beta = torch.chunk(y_hat, 4, dim=1)
        loss = self.loss_function(y, mu, v, alpha, beta)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), mu, v, alpha, beta

    def evaluate(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(x)
            mu, v, alpha, beta = torch.chunk(y_hat, 4, dim=1)
            rmse = torch.sqrt(F.mse_loss(mu, y))
            loss = self.loss_function(y, mu, v, alpha, beta)
        return mu, v, alpha, beta, loss, rmse.item()

    def get_batch(self, x, y, batch_size):
        idx = np.sort(np.random.choice(x.shape[0], batch_size, replace=False))

        if isinstance(x, (np.ndarray, h5py.Dataset)):
            x_ = np.transpose(x[idx, ...], (0, 3, 1, 2))
            y_ = np.transpose(y[idx, ...], (0, 3, 1, 2))

            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0

            x_ = torch.tensor(x_ / x_divisor, dtype=torch.float32)
            y_ = torch.tensor(y_ / y_divisor, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown dataset type {type(x)}, {type(y)}")

        return x_, y_

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{name}.pth"))

    def train(self, x_train, x_val, y_train, y_val, x_test, y_test, batch_size=128, iters=10000, verbose=True):
        tic = time.time()
        for self.iter in range(iters):
            x_batch, y_batch = self.get_batch(x_train, y_train, batch_size)
            loss, mu, v, alpha, beta = self.run_train_step(x_batch, y_batch)

            if self.iter % 500 == 0:
                x_val_batch, y_val_batch = self.get_batch(x_val, y_val, min(100, y_val.shape[0]))
                mu, v, alpha, beta, val_loss, rmse = self.evaluate(x_val_batch, y_val_batch)
                if verbose:
                    print(f"[{self.iter}] RMSE: {rmse:.4f} train_loss: {loss:.4f} time: {time.time() - tic:.2f}s")
                tic = time.time()

        self.save("final")
        return self.model
