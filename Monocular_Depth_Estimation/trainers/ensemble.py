import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from pathlib import Path
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ensemble:
    def __init__(self, models, dataset="", learning_rate=1e-3, tag="",sigma=True):
        self.mse = not sigma

        self.model = models.to(device)
        self.loss_function = nn.MSELoss() if self.mse else nn.GaussianNLLLoss()
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in self.model.models]

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save', f'{current_time}_{dataset}_{trainer}_{tag}')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def run_train_step(self, x, y):
        x, y = x.to(device), y.to(device)
        losses = []
        y_hats = []
        for model, optimizer in zip(self.model.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            outputs = model(x)
            if self.mse:
                mu = outputs
                loss = self.loss_function(y, mu)
            else:
                mu, sigma = torch.chunk(outputs, 2, dim=1)
                loss = self.loss_function(mu, y, sigma)
            loss.backward()
            optimizer.step()
            y_hats.append(mu.detach())
            losses.append(loss.detach())
        return torch.stack(losses).mean(), torch.stack(y_hats).mean(dim=0)

    def evaluate(self, x, y):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = torch.stack([model(x) for model in self.model.models], dim=0)
            if self.mse:
                mean_mu = preds.mean(dim=0)
                var = preds.var(dim=0)
                loss = self.loss_function(y, mean_mu)
            else:
                mus, sigmas = torch.chunk(preds, 2, dim=2)
                mean_mu = mus.mean(dim=0)
                mean_sigma = sigmas.mean(dim=0)
                loss = self.loss_function(mean_mu, y, mean_sigma)
                var=mean_sigma
            rmse = torch.sqrt(F.mse_loss(mean_mu, y))

        return mean_mu, var, loss, rmse

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
