import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, t as student_t

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import interval_score, ece_pi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predictive_interval(mu_pred, v_pred, alpha_pred, beta_pred, confidence=0.95):
    dof = 2.0 * alpha_pred
    scale = torch.sqrt((1.0 + v_pred) * beta_pred / (alpha_pred * v_pred))
    alpha = 1.0 - confidence
    upper_q = 1.0 - alpha / 2.0
    t_upper = student_t.ppf(upper_q, df=dof.detach().cpu().numpy())
    interval = t_upper * scale.detach().cpu().numpy()
    return interval


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)
    alpha = torch.clamp(alpha, min=epsilon)
    beta = torch.clamp(beta, min=epsilon)
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(torch.pi / v) - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll.mean() if reduce else nll


def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True):
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi
    return reg.mean() if reduce else reg


def evidential_loss(y, mu, v, alpha, beta, lam=0.0, epsilon=1e-2):
    nll = NIG_NLL(y, mu, v, alpha, beta, reduce=True)
    reg = NIG_Reg(y, mu, v, alpha, beta, reduce=True)
    return nll + lam * (reg - epsilon)


class Evidential:
    def __init__(self, model, dataset="", noise='', tag="", learning_rate=1e-3, lam=0.0, load_model=True, model_dir='save'):
        self.model = model.to(device)
        self.criterion = evidential_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lam = lam
        self.epoch = 0
        self.load_model=load_model
        trainer = self.__class__.__name__
        save_subdir = f"{dataset}" if noise == '' else f"{dataset}_{noise}"
        self.save_dir = Path(model_dir) / save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{trainer}_{tag}.pth"

    def run_train_step(self, x, y):
        self.model.train()
        mu, v, alpha, beta = self.model(x)
        loss = self.criterion(y, mu, v, alpha, beta, lam=self.lam)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x, y, y_mu, y_scale):
        self.model.eval()
        tic = time.time()
        mu, v, alpha, beta = self.model(x)

        alpha = torch.clamp(alpha, min=1e-6 + 1)
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))
        test_time = (time.time() - tic) * 1000

        y = ((y.detach().cpu().numpy() * y_scale) + y_mu).squeeze()
        predictions = ((mu.detach().cpu().numpy() * y_scale) + y_mu).squeeze()
        sigmas = (predictive_interval(mu, v, alpha, beta, confidence=0.95) * y_scale).squeeze()
        aleatoric = aleatoric.detach().cpu().numpy().squeeze()
        epistemic = epistemic.detach().cpu().numpy().squeeze()

        uncertainty_pred = aleatoric + epistemic
        ece = ece_pi(y, predictions - sigmas, predictions + sigmas)
        is_score = interval_score(y, predictions, predictions - sigmas, predictions + sigmas, alpha=0.05)
        mpi_width = np.mean(sigmas) * 2
        within_interval = ((y >= (predictions - sigmas)) & (y <= (predictions + sigmas))).astype(float)
        picp = np.mean(within_interval)
        within_interval_down = (y[predictions > y] >= (predictions[predictions > y] - sigmas[predictions > y])).astype(float)
        picp_down = np.mean(within_interval_down)
        within_interval_up = (y[predictions < y] <= (predictions[predictions < y] + sigmas[predictions < y])).astype(float)
        picp_up = np.mean(within_interval_up)
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        err = np.abs(y - predictions).squeeze()
        spearman_rmse, _ = spearmanr(err, uncertainty_pred)

        return rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, spearman_rmse, test_time

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, train_dataset, test_dataset, y_mu, y_scale, batch_size=128, num_epochs=400, verbose=True):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        train_loss_curve = []
        train_times, test_times = [], []

        if self.load_model and self.save_path.exists():
            self.model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        else:
            for self.epoch in range(1, num_epochs + 1):
                epoch_loss = 0.0
                tic = time.time()
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss = self.run_train_step(data, target)
                    epoch_loss += loss
                train_times.append((time.time() - tic) * 1000)
                train_loss_curve.append(epoch_loss / len(train_loader))
            self.save()

        for data, target in test_loader:
            results = self.evaluate(data, target, y_mu, y_scale)
            rmse, mpiw, picp, picp_up, picp_down, iscore, ece, s_rmse, test_time = results
            test_times.append(test_time)

        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_curve, label='Training Loss', color='green')
            plt.legend()
            plt.title('Training Loss Curve')
            plt.show()

        return self.model, rmse, mpiw, picp, picp_up, picp_down, iscore, ece, s_rmse, np.mean(train_times), np.mean(test_times)
