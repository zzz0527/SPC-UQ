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

class Dropout:
    def __init__(self, model, dataset="", learning_rate=1e-3, tag="", lam=1e-3, l=0.5, drop_prob=0.1, sigma=True):
        # self.model = model
        self.model = model.to(device)
        self.l = l
        self.drop_prob = drop_prob
        self.mse = not sigma
        self.lam = lam

        self.loss_function = nn.MSELoss() if self.mse else nn.GaussianNLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save', f"{current_time}_{dataset}_{trainer}_{tag}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def run_train_step(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.train()
        self.optimizer.zero_grad()
        y_hat = self.model(x)  # Forward pass

        if self.mse:
            loss = self.loss_function(y_hat, y)
        else:
            mu, sigma = torch.chunk(y_hat, 2, dim=1)
            loss = self.loss_function(mu, y, sigma)

        # L2 Regularization
        # loss += self.lam * sum(torch.norm(p, p=2) for p in self.model.parameters())

        loss.backward()
        self.optimizer.step()

        return loss.item(), y_hat

    def evaluate(self, x, y):
        x, y = x.to(device), y.to(device)
        self.model.train()
        preds = torch.stack([self.model(x) for _ in range(5)], dim=0)  # Forward pass multiple times
        mu, var = torch.mean(preds, dim=0), torch.var(preds, dim=0)

        if self.mse:
            mean_mu = mu
            loss = self.loss_function(mean_mu, y)
        else:
            mean_mu, mean_sigma = torch.chunk(mu, 2, dim=1)
            loss = self.loss_function(mean_mu, y, mean_sigma)

        rmse = torch.sqrt(F.mse_loss(mean_mu, y))


        return mu, var, loss.item(), rmse.item()

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
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, y_hat = self.run_train_step(x_input_batch, y_input_batch)

            if self.iter % 500 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                mu, var, vloss, rmse = self.evaluate(x_test_batch, y_test_batch)

                if verbose:
                    print(
                        f"[{self.iter}] RMSE: {rmse:.4f} train_loss: {loss:.4f} t: {time.time() - tic:.2f} sec")
                    tic = time.time()

        self.save(f"final")

        return self.model
