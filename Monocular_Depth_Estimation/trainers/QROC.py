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
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pinball_loss(q_pred, target, tau):
    errors = target - q_pred
    loss = torch.max(tau * errors, (tau - 1) * errors)
    return torch.mean(loss)


class QROC:
    def __init__(self, model, dataset="", learning_rate=1e-3, tag="", lam=1e-3, l=0.5, drop_prob=0.1, sigma=False):
        self.model = model.to(device)
        self.l = l
        self.lam = lam

        self.loss_function = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.1)

        self.min_rmse = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save', f"{current_time}_{dataset}_{trainer}_{tag}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def multi_quantile_loss(self, target, q_low, q_mid, q_high, tau_low=0.025, tau_mid=0.5, tau_high=0.975):
        loss_low = pinball_loss(q_low, target, tau_low)
        loss_mid = pinball_loss(q_mid, target, tau_mid)
        loss_high = pinball_loss(q_high, target, tau_high)
        return (loss_low + loss_mid + loss_high)

    def train_OC(self,
                 network,
                 loader,
                 certs_loss="bce",
                 certs_k=100,
                 certs_reg=0,
                 certs_bias=0):

        loader = loader.to(device)
        features, certificates = network.Orthonormal_Certificates(loader)

        def target(x):
            return torch.zeros(x.size(0), certs_k, x.size(2), x.size(3))

        opt = torch.optim.Adam(network.certificates.parameters())
        sig = torch.nn.Sigmoid()

        if certs_loss == "bce":
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif certs_loss == "mse":
            loss = torch.nn.L1Loss(reduction="none")

        opt.zero_grad()
        error = loss(certificates, target(features).to(device)).mean()
        # penalty = certs_reg * \
        #           (network.certificates.weight @ network.certificates.weight.t() -
        #            torch.eye(certs_k)).pow(2).mean()
        (error).backward()
        opt.step()

        # loader_f = DataLoader(TensorDataset(features, features),
        #                       batch_size=64,
        #                       shuffle=True)
        #
        # for epoch in range(certs_epochs):
        #     for f, _ in loader_f:
        #         opt.zero_grad()
        #         error = loss(certificates(f), target(f)).mean()
        #         penalty = certs_reg * \
        #                   (certificates.weight @ certificates.weight.t() -
        #                    torch.eye(certs_k)).pow(2).mean()
        #         (error + penalty).backward()
        #         opt.step()

    def run_train_step(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.train()
        q_low, q_mid, q_high,_ = self.model(x)
        loss = self.multi_quantile_loss(y, q_low, q_mid, q_high)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.eval()
        q_low, q_mid, q_high,_ = self.model(x)  # Forward pass

        rmse = torch.sqrt(F.mse_loss(q_mid, y))


        return q_mid, rmse.item()

    def get_batch(self, x, y, batch_size):
        idx = np.sort(np.random.choice(x.shape[0], batch_size, replace=False))

        if isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            x_ = np.transpose(x[idx, ...], (0, 3, 1, 2))
            y_ = np.transpose(y[idx, ...], (0, 3, 1, 2))

            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0

            x_ = torch.tensor(x_/x_divisor, dtype=torch.float32)
            y_ = torch.tensor(y_/y_divisor, dtype=torch.float32)
        else:
            print(f"Unknown dataset type {type(x)}, {type(y)}")

        return x_, y_

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{name}.pth"))

    def train(self, x_train, x_val, y_train, y_val, x_test, y_test, batch_size=128, iters=10000, verbose=True):
        tic = time.time()
        for self.iter in range(iters+1):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss = self.run_train_step(x_input_batch, y_input_batch)

            if self.iter % 500 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                pred,rmse = self.evaluate(x_test_batch, y_test_batch)

                if rmse < self.min_rmse:
                    self.min_rmse = rmse
                #     self.save(f"model_rmse_{self.iter}")

                if verbose:
                    print(
                        f"[{self.iter}] RMSE: {rmse:.4f} train_loss: {loss:.4f} t: {time.time() - tic:.2f} sec")
                tic = time.time()

        for self.iter in range(5000+1):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            # x_input_batch, y_input_batch = self.get_batch(x_val, y_val, batch_size)
            self.train_OC(self.model, x_input_batch)

        self.save(f"final")

        return self.model, self.min_rmse
