import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ConformalRegressionNet(nn.Module):
    """
    Simple feedforward regression model with dropout.
    Output: point prediction only (no uncertainty head).
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


class ConformalRegressor:
    """
    Quantile-based conformal prediction regression model.
    """
    def __init__(self, quantile=0.9, learning_rate=5e-3):
        torch.manual_seed(24)
        self.quantile = quantile
        self.model = ConformalRegressionNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.quantile_up = 0.0
        self.quantile_down = 0.0

    def train(self, x, y, num_epochs=5000):
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")

    def calibrate(self, x_calib, y_calib):
        """
        Compute empirical quantiles from residuals on calibration set.
        """
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_calib).detach().cpu().numpy().squeeze()
            y_calib_np = y_calib.detach().cpu().numpy().squeeze()
            residuals = y_calib_np - pred

            res_up = residuals[residuals > 0]
            res_down = -residuals[residuals <= 0]

            self.quantile_up = np.quantile(res_up, self.quantile) if len(res_up) > 0 else 0.0
            self.quantile_down = np.quantile(res_down, self.quantile) if len(res_down) > 0 else 0.0

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x).detach().cpu().numpy().squeeze()
            upper = y_pred + self.quantile_up
            lower = y_pred - self.quantile_down
            uncertainty = np.zeros_like(y_pred)  # Conformal prediction doesn't model epistemic

        return y_pred, upper, lower, uncertainty
