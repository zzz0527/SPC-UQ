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


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    """
    Negative log-likelihood for Normal-Inverse-Gamma.
    """
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)
    alpha = torch.clamp(alpha, min=epsilon)
    beta = torch.clamp(beta, min=epsilon)

    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(torch.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return nll.mean() if reduce else nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    """
    Kullback-Leibler divergence between two Normal-Inverse-Gamma distributions.
    """
    epsilon = 1e-6
    v1 = torch.clamp(v1, min=epsilon)
    v2 = torch.clamp(v2, min=epsilon)
    b1 = torch.clamp(b1, min=epsilon)
    b2 = torch.clamp(b2, min=epsilon)

    term1 = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2)
    term2 = 0.5 * v2 / v1
    term3 = -0.5 * torch.log(v2 / v1)
    term4 = -0.5
    term5 = a2 * torch.log(b1 / b2)
    term6 = -(torch.lgamma(a1) - torch.lgamma(a2))
    term7 = (a1 - a2) * torch.digamma(a1)
    term8 = -(b1 - b2) * a1 / b1

    kl = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
    return kl


def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    """
    Regularization term for evidential learning using either KL divergence or evidence penalty.
    """
    error = torch.abs(y - gamma)

    if kl:
        kl_div = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl_div
    else:
        evidence = 2 * v + alpha
        reg = error * evidence

    return reg.mean() if reduce else reg


def edl_loss(y, mu, v, alpha, beta, lam=0.2, epsilon=0.0, reduce=True, return_comps=False):
    """
    Evidential regression loss combining NLL and evidence-based regularization.
    """
    nll = NIG_NLL(y, mu, v, alpha, beta, reduce)
    reg = NIG_Reg(y, mu, v, alpha, beta, reduce)

    loss = nll + lam * (reg - epsilon)
    return (loss, (nll, reg)) if return_comps else loss


class Evidential:
    def __init__(self, model, dataset="", learning_rate=1e-3, lam=0.3, epsilon=1e-2, tag=""):
        self.model = model.to(device)
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join("save", f"{timestamp}_{dataset}_{trainer}_{tag}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def loss_function(self, y, mu, v, alpha, beta, reduce=True, return_comps=False):
        """
        Composite loss with NLL and evidence regularization.
        """
        nll = NIG_NLL(y, mu, v, alpha, beta, reduce)
        reg = NIG_Reg(y, mu, v, alpha, beta, reduce)
        loss = nll + self.lam * (reg - self.epsilon)
        return (loss, (nll, reg)) if return_comps else loss

    def run_train_step(self, x, y):
        """
        One forward-backward optimization step.
        """
        x, y = x.to(device), y.to(device)
        self.model.train()

        y_hat = self.model(x)
        mu, v, alpha, beta = torch.chunk(y_hat, 4, dim=1)
        loss, (nll, reg) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), nll.item(), reg.item(), mu, v, alpha, beta

    def evaluate(self, x, y):
        """
        Evaluate on validation/test set.
        """
        x, y = x.to(device), y.to(device)
        self.model.eval()

        with torch.no_grad():
            y_hat = self.model(x)
            mu, v, alpha, beta = torch.chunk(y_hat, 4, dim=1)
            rmse = torch.sqrt(F.mse_loss(mu, y))
            loss, (nll, reg) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

        return mu, v, alpha, beta, loss, rmse.item(), nll.item(), reg.item()

    def get_batch(self, x, y, batch_size):
        """
        Get a batch from numpy or h5py dataset.
        """
        idx = np.sort(np.random.choice(x.shape[0], batch_size, replace=False))

        if isinstance(x, (np.ndarray, h5py.Dataset)):
            x_ = np.transpose(x[idx], (0, 3, 1, 2))
            y_ = np.transpose(y[idx], (0, 3, 1, 2))

            x_ = torch.tensor(x_ / 255.0 if x_.dtype == np.uint8 else x_, dtype=torch.float32)
            y_ = torch.tensor(y_ / 255.0 if y_.dtype == np.uint8 else y_, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        return x_, y_

    def save(self, name):
        """
        Save model to file.
        """
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{name}.pth"))

    def train(self, x_train, x_val, y_train, y_val, x_test, y_test, batch_size=128, iters=10000, verbose=True):
        """
        Training loop.
        """
        tic = time.time()
        for self.iter in range(iters):
            xb, yb = self.get_batch(x_train, y_train, batch_size)
            loss, nll, reg, mu, v, alpha, beta = self.run_train_step(xb, yb)

            if self.iter % 500 == 0:
                xv, yv = self.get_batch(x_val, y_val, min(100, y_val.shape[0]))
                mu, v, alpha, beta, val_loss, rmse, nll_val, reg_val = self.evaluate(xv, yv)

                if verbose:
                    print(f"[{self.iter}] RMSE: {rmse:.4f} TrainLoss: {loss:.4f} Time: {time.time() - tic:.2f}s")
                tic = time.time()

        self.save("final")
        return self.model
