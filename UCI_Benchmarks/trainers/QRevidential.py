import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from .utils import interval_score, ece_pi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NIG_NLL(y, gamma, v, alpha, beta, w_i_mean, quantile, reduce=True):
    """Negative log-likelihood for Normal-Inverse-Gamma distribution."""
    tau = 2.0 / (quantile * (1.0 - quantile))
    twoBlambda = 4.0 * beta * (1.0 + tau * w_i_mean * v)
    nll = 0.5 * torch.log(math.pi / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return torch.mean(nll) if reduce else nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    """KL divergence between two NIG distributions."""
    kl = 0.5 * (a1 - 1.0) / b1 * v2 * (mu2 - mu1) ** 2 \
        + 0.5 * (v2 / v1) \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 \
        + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return kl


def tilted_loss(q, e):
    """Pinball loss used in quantile regression."""
    return torch.maximum(q * e, (q - 1.0) * e)


def NIG_Reg(y, gamma, v, alpha, beta, w_i_mean, quantile, omega=0.01, reduce=True, use_kl=False):
    """Regularization term: either KL-divergence or uncertainty penalty."""
    error = tilted_loss(quantile, y - gamma)
    if use_kl:
        kl_val = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1.0 + omega, beta)
        reg = error * kl_val
    else:
        evidence = 2.0 * v + alpha + 1.0 / beta
        reg = error * evidence
    return torch.mean(reg) if reduce else reg


def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    """Evidential quantile regression loss: NLL + regularization."""
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    w_i_mean = beta / (alpha - 1.0)
    mu = gamma + theta * w_i_mean
    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_mean, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_mean, quantile, reduce=reduce)
    return loss_nll + coeff * loss_reg


class QRevidential:
    """Trainer class for Evidential Quantile Regression."""

    def __init__(self, model, dataset="", noise='', tag="", learning_rate=1e-3, lam=0.0,load_model=True, model_dir='save'):
        self.model = model.to(device)
        self.quantiles = [0.025, 0.5, 0.975]
        self.coeff = 0.5
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch = 0
        self.load_model=load_model
        trainer = self.__class__.__name__
        save_subdir = f"{dataset}" if noise == '' else f"{dataset}_{noise}"
        self.save_dir = Path(model_dir) / save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{trainer}_{tag}.pth"

    def loss_function(self, y, mu, v, alpha, beta):
        """Aggregate evidential quantile loss across all quantiles."""
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            loss += quant_evi_loss(y, mu[:, i:i+1], v[:, i:i+1], alpha[:, i:i+1], beta[:, i:i+1], q, coeff=self.coeff)
        return loss

    def run_train_step(self, x, y):
        """Run one training step."""
        self.model.train()
        mu, v, alpha, beta = self.model(x)
        loss = self.loss_function(y, mu, v, alpha, beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x, y, y_mu, y_scale):
        """Evaluate model performance and uncertainty."""
        self.model.eval()
        tic = time.time()
        mu, v, alpha, beta = self.model(x)

        mu_low, mu_mid, mu_high = torch.unbind(mu, dim=1)
        v_low, v_mid, v_high = torch.unbind(v, dim=1)
        alpha_low, alpha_mid, alpha_high = torch.unbind(alpha, dim=1)
        beta_low, beta_mid, beta_high = torch.unbind(beta, dim=1)

        alpha_mid = torch.clamp(alpha_mid, min=1e-6 + 1)
        aleatoric = (beta_mid / (alpha_mid - 1) +
                     beta_low / (alpha_low - 1) +
                     beta_high / (alpha_high - 1))
        epistemic = (beta_mid / (v_mid * (alpha_mid - 1)) +
                     beta_low / (v_low * (alpha_low - 1)) +
                     beta_high / (v_high * (alpha_high - 1)))

        test_time = (time.time() - tic) * 1000

        y_true = y.detach().cpu().numpy() * y_scale + y_mu
        pred = mu_mid.detach().cpu().numpy() * y_scale + y_mu
        lower = mu_low.detach().cpu().numpy() * y_scale + y_mu
        upper = mu_high.detach().cpu().numpy() * y_scale + y_mu
        uncertainty = (aleatoric + epistemic).detach().cpu().numpy().squeeze()

        rmse = np.sqrt(np.mean((y_true.squeeze() - pred.squeeze()) ** 2))
        picp = np.mean((y_true.squeeze() >= lower.squeeze()) & (y_true.squeeze() <= upper.squeeze()))
        picp_up = np.mean((y_true.squeeze()[pred.squeeze() < y_true.squeeze()] <= upper.squeeze()[pred.squeeze() < y_true.squeeze()]))
        picp_down = np.mean((y_true.squeeze()[pred.squeeze() > y_true.squeeze()] >= lower.squeeze()[pred.squeeze() > y_true.squeeze()]))
        mpiw = np.mean(upper - lower)
        ece = ece_pi(y_true.squeeze(), lower.squeeze(), upper.squeeze())
        is_score = interval_score(y_true.squeeze(), pred.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.05)
        spearman_corr, _ = spearmanr(np.abs(y_true.squeeze() - pred.squeeze()), uncertainty)

        return rmse, mpiw, picp, picp_up, picp_down, is_score, ece, spearman_corr, test_time

    def save(self):
        """Save the model checkpoint."""
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, train_dataset, test_dataset, y_mu, y_scale, batch_size=128, num_epochs=400, verbose=True):
        """Train the model and evaluate after training."""
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        train_losses = []
        test_rmse_curve = []
        train_times, test_times = [], []

        if self.load_model and self.save_path.exists():
            self.model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        else:
            for self.epoch in range(1, num_epochs + 1):
                epoch_loss = 0.0
                tic = time.time()
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    loss = self.run_train_step(x_batch, y_batch)
                    epoch_loss += loss
                train_losses.append(epoch_loss / len(train_loader))
                train_times.append((time.time() - tic) * 1000)
            self.save()

        for x_batch, y_batch in test_loader:
            results = self.evaluate(x_batch, y_batch, y_mu, y_scale)
            rmse, mpiw, picp, picp_up, picp_down, is_score, ece, spearman_corr, test_time = results
            test_rmse_curve.append(rmse)
            test_times.append(test_time)

        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', color='green')
            plt.title('Training Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(test_rmse_curve, label='Test RMSE')
            plt.title('Test RMSE Curve')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.grid(True)
            plt.legend()
            plt.show()

        return self.model, rmse, mpiw, picp, picp_up, picp_down, is_score, ece, spearman_corr, \
               np.mean(train_times), np.mean(test_times)
