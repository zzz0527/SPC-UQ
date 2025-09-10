import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import interval_score, ece_pi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_nll_loss(y, mu, sigma, reduce=True):
    """Gaussian negative log-likelihood loss."""
    sigma = torch.clamp(sigma, min=1e-6)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
    logprob = -torch.log(sigma) - 0.5 * log_2pi - ((y - mu) ** 2) / (2 * sigma ** 2)
    nll = -logprob

    if y.dim() > 1:
        loss = nll.mean(dim=tuple(range(1, y.dim())))
    else:
        loss = nll

    return loss.mean() if reduce else loss


class Ensemble:
    def __init__(self, model, dataset="", noise='', tag="", learning_rate=1e-3, load_model=True, model_dir='pretrained_model_weights'):
        self.model = model.to(device)
        self.criterion = gaussian_nll_loss
        self.optimizers = [optim.Adam(m.parameters(), lr=learning_rate) for m in self.model.models]
        self.epoch = 0
        self.load_model=load_model
        trainer = self.__class__.__name__
        save_subdir = f"{dataset}" if noise == '' else f"{dataset}_{noise}"
        self.save_dir = Path(model_dir) / save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{trainer}_{tag}.pth"

    def run_train_step_single(self, x, y, model_index):
        model = self.model.models[model_index]
        optimizer = self.optimizers[model_index]
        model.train()
        mu, sigma = model(x)
        loss = self.criterion(y, mu, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, x, y, y_mu, y_scale):
        self.model.eval()
        tic = time.time()
        mus, sigmas = self.model(x)
        stacked_mus = torch.stack(mus, dim=0)
        mean_mu = stacked_mus.mean(dim=0)
        stacked_sigmas = torch.stack(sigmas, dim=0)
        mean_sigma = stacked_sigmas.mean(dim=0)
        aleatoric = mean_sigma
        epistemic = stacked_mus.std(dim=0, unbiased=False)
        test_time = (time.time() - tic) * 1000

        y = ((y.detach().cpu().numpy() * y_scale) + y_mu).squeeze()
        predictions = ((mean_mu.detach().cpu().numpy() * y_scale) + y_mu).squeeze()
        sigma_ensemble = (mean_sigma.detach().cpu().numpy() * y_scale).squeeze()
        aleatoric = aleatoric.detach().cpu().numpy().squeeze()
        epistemic = epistemic.detach().cpu().numpy().squeeze()

        uncertainty_pred = aleatoric + epistemic
        ece = ece_pi(y, predictions - 2 * sigma_ensemble, predictions + 2 * sigma_ensemble)
        is_score = interval_score(y, predictions, predictions - 2 * sigma_ensemble, predictions + 2 * sigma_ensemble, alpha=0.05)
        mpi_width = np.mean(4 * sigma_ensemble)
        within_interval = ((y >= (predictions - 2 * sigma_ensemble)) & (y <= (predictions + 2 * sigma_ensemble))).astype(float)
        picp = np.mean(within_interval)
        within_interval_down = (y[predictions > y] >= (predictions[predictions > y] - 2 * sigma_ensemble[predictions > y])).astype(float)
        picp_down = np.mean(within_interval_down)
        within_interval_up = (y[predictions < y] <= (predictions[predictions < y] + 2 * sigma_ensemble[predictions < y])).astype(float)
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
                for model_index in range(len(self.model.models)):
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        loss = self.run_train_step_single(data, target, model_index)
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
