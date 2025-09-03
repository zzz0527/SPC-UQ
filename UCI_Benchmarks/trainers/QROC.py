import numpy as np
import time
import os
from pathlib import Path
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from .utils import interval_score, ece_pi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pinball_loss(q_pred, target, tau):
    errors = target - q_pred
    loss = torch.max(tau * errors, (tau - 1) * errors)
    return torch.mean(loss)

def featurize_loader(network, loader):
    features = []
    for (x, _) in loader:
        x = x.to(next(network.parameters()).device)
        feats = network.features(x)
        features.append(feats.detach())
    return torch.cat(features).squeeze().cpu()

def featurize_loader2(network, loader):
    features = []
    all_features = network.features(loader)
    return all_features.cpu()

def uncertainty_certificates(network,
                             loader,
                             certs_loss="bce",
                             certs_k=100,
                             certs_epochs=10,
                             certs_reg=0,
                             certs_bias=0):
    """
    Compute uncertainty using linear certificates (ours)
    """
    features = featurize_loader(network, loader)

    def target(x):
        return torch.zeros(x.size(0), certs_k)

    certificates = torch.nn.Linear(features.size(1), certs_k, bias=certs_bias)
    opt = torch.optim.Adam(certificates.parameters())
    sig = torch.nn.Sigmoid()

    if certs_loss == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    elif certs_loss == "mse":
        loss = torch.nn.L1Loss(reduction="none")

    loader_f = DataLoader(TensorDataset(features, features),
                          batch_size=64,
                          shuffle=True)

    for epoch in range(certs_epochs):
        for f, _ in loader_f:
            opt.zero_grad()
            error = loss(certificates(f), target(f)).mean()
            penalty = certs_reg * \
                      (certificates.weight @ certificates.weight.t() -
                       torch.eye(certs_k)).pow(2).mean()
            (error + penalty).backward()
            opt.step()

    def handle(network, loader_test, args=None):
        f = featurize_loader2(network, loader_test)
        output = certificates(f)
        if certs_loss == "bce":
            return sig(output).pow(2).mean(1).detach()
        else:
            return output.pow(2).mean(1).detach()

    return handle
def multi_quantile_loss(target, q_low, q_mid, q_high, tau_low=0.025, tau_mid=0.5, tau_high=0.975):
    loss_low = pinball_loss(q_low, target, tau_low)
    loss_mid = pinball_loss(q_mid, target, tau_mid)
    loss_high = pinball_loss(q_high, target, tau_high)
    return (loss_low +loss_mid + loss_high)

class QROC:
    def __init__(self, model, dataset="", noise='', tag="", learning_rate=1e-3,load_model=True, model_dir='save'):
        self.model = model.to(device)
        self.criterion = multi_quantile_loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.epoch = 0
        self.load_model=load_model
        trainer = self.__class__.__name__
        save_subdir = f"{dataset}" if noise == '' else f"{dataset}_{noise}"
        self.save_dir = Path(model_dir) / save_subdir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"{trainer}_{tag}.pth"

    def run_train_step(self, x, y):
        self.model.train()
        model = self.model
        optimizer = self.optimizer
        q_low, q_mid, q_high = model(x)
        loss = self.criterion(y, q_low, q_mid, q_high)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, x, y, y_mu, y_scale):
        self.model.eval()
        tic=time.time()
        q_low, q_mid, q_high = self.model(x)
        aleatoric=(q_high-q_low)
        epistemic = -self.OC(self.model,x)
        test_time = (time.time() - tic)* 1000

        y = y.detach().cpu().numpy()
        predictions = q_mid.detach().cpu().numpy()
        y = ((y * y_scale) + y_mu).squeeze()
        predictions = ((predictions * y_scale) + y_mu).squeeze()
        low_bound = (q_low.detach().cpu().numpy()* y_scale + y_mu).squeeze()
        high_bound = (q_high.detach().cpu().numpy()* y_scale + y_mu).squeeze()
        aleatoric = aleatoric.detach().cpu().numpy().squeeze()
        epistemic = epistemic.detach().cpu().numpy().squeeze()
        uncertainty_pred = aleatoric + epistemic

        ece = ece_pi(y, low_bound, high_bound)

        is_score = interval_score(y, predictions, low_bound, high_bound, alpha=0.05)

        mpi_width = np.mean(high_bound-low_bound)

        within_interval = ((y >= low_bound) & (y <= high_bound)).astype(float)
        picp = np.mean(within_interval)

        within_interval_down = (y[predictions > y] >= low_bound[predictions > y]).astype(float)
        picp_down = np.mean(within_interval_down)

        within_interval_up = (y[predictions < y] <= high_bound[predictions < y]).astype(float)
        picp_up = np.mean(within_interval_up)

        rmse = np.sqrt(np.mean(((y - predictions)) ** 2))

        err=(np.abs(y - predictions)).squeeze()
        Spearman_rmse, p_rmse = spearmanr(err, uncertainty_pred)


        return rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, test_time

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, train_dataset, test_dataset, y_mu, y_scale, batch_size=128, num_epochs=400, verbose=True):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
        train_loss_curve=[]
        test_rmse_curve=[]
        train_average_time = []
        test_average_time = []

        if self.load_model and self.save_path.exists():
            self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))

        else:
            for self.epoch in range(1, num_epochs + 1):
                epoch_loss = 0.0
                tic = time.time()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    loss = self.run_train_step(data, target)
                    epoch_loss += loss

                train_average_time.append((time.time() - tic)* 1000)
                average_loss = epoch_loss / len(train_loader)

                train_loss_curve.append(average_loss)

            self.save()
        self.OC = uncertainty_certificates(self.model,train_loader)

        for batch_idx, (data, target) in enumerate(test_loader):
            rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, test_time = self.evaluate(data, target, y_mu, y_scale)
            test_rmse_curve.append(rmse)
            test_average_time.append(test_time)


        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_curve, label='loss', color='green')
            plt.legend()
            plt.title('loss in training')
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(test_rmse_curve, label='test_rmse_curve')
            plt.legend()
            plt.title('rmse in testing')
            plt.show()

        return self.model, rmse, mpi_width, picp, picp_up, picp_down, is_score, ece, Spearman_rmse, np.mean(train_average_time), np.mean(test_average_time)