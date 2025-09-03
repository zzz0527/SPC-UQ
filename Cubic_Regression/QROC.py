import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def pinball_loss(q_pred, target, tau):
    """
    Compute the quantile (pinball) loss for a given quantile level tau
    """
    error = target - q_pred
    return torch.mean(torch.max(tau * error, (tau - 1) * error))

def multi_quantile_loss(q_low, q_mid, q_high, target, tau_low=0.05, tau_high=0.95):
    """
    Compute total loss across low, median, and high quantiles
    """
    loss_l = pinball_loss(q_low, target, tau_low)
    loss_m = pinball_loss(q_mid, target, 0.5)
    loss_h = pinball_loss(q_high, target, tau_high)
    return loss_l + loss_m + loss_h


class QuantileRegressionNN(nn.Module):
    """
    Fully-connected quantile regression network with three outputs:
    lower, median, upper quantiles.
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.out(x)
        return out[:, :1], out[:, 1:2], out[:, 2:]

    def extract_features(self, x):
        """
        Extract penultimate-layer features (for certificate head)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


def build_certificate_head(features, out_dim=20, epochs=500):
    """
    Train an orthogonal linear projection (certificate head) on extracted features
    """
    projection = nn.Linear(features.size(1), out_dim)
    loader = DataLoader(TensorDataset(features), shuffle=True, batch_size=128)
    optimizer = optim.Adam(projection.parameters())

    for _ in range(epochs):
        for (feature_batch,) in loader:
            optimizer.zero_grad()
            output = projection(feature_batch)
            cert_loss = output.pow(2).mean()

            identity = torch.eye(out_dim, device=projection.weight.device)
            ortho_penalty = (projection.weight @ projection.weight.T - identity).pow(2).mean()

            (cert_loss + ortho_penalty).backward()
            optimizer.step()

    return projection


class QROC:
    """
    Single-model Quantile Regression with Orthogonal Certificate head (QROC)
    Provides aleatoric and epistemic uncertainty estimation
    """
    def __init__(self, learning_rate=5e-3, tau_low=0.05, tau_high=0.95):
        torch.manual_seed(42)
        self.model = QuantileRegressionNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.certificate_head = None

    def train(self, x, y, epochs=3000):
        """
        Train the quantile regression network and then fit certificate head on extracted features
        """
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            q_low, q_mid, q_high = self.model(x)
            loss = multi_quantile_loss(q_low, q_mid, q_high, y, self.tau_low, self.tau_high)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 500 == 0:
                print(f"[{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        # Train certificate head using the final feature representation
        self.model.eval()
        with torch.no_grad():
            features = self.model.extract_features(x).detach()
        self.certificate_head = build_certificate_head(features)

    def predict(self, x):
        """
        Predict quantiles and return aleatoric and epistemic uncertainties
        """
        self.model.eval()
        with torch.no_grad():
            q_low, q_mid, q_high = self.model(x)

            # Epistemic: projection energy via orthogonal certificate head
            features = self.model.extract_features(x)
            epistemic = self.certificate_head(features).pow(2).mean(dim=1).cpu().numpy()

        return (
            q_mid.squeeze().numpy(),
            q_high.squeeze().numpy(),
            q_low.squeeze().numpy(),
            epistemic
        )
