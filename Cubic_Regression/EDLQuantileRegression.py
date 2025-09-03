import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math


class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.out_units = int(out_units)
        self.linear = nn.Linear(in_features, 4 * self.out_units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, log_v, log_alpha, log_beta = torch.chunk(output, chunks=4, dim=-1)
        v = self.evidence(log_v)
        alpha = self.evidence(log_alpha) + 1.0
        beta = self.evidence(log_beta)
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def extra_repr(self):
        return f"out_units={self.out_units}"


class EDLQRNet(nn.Module):
    def __init__(self, input_dim=1, num_quantiles=3, hidden_dim=64, num_layers=2, activation=nn.ReLU()):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
        layers.append(DenseNormalGamma(in_features, num_quantiles))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        mu, v, alpha, beta = torch.chunk(output, 4, dim=-1)
        return mu, v, alpha, beta


def nig_nll(y, mu, v, alpha, beta, wi_mean, quantile, reduce=True):
    tau2 = 2.0 / (quantile * (1.0 - quantile))
    two_b_lambda = 4.0 * beta * (1.0 + tau2 * wi_mean * v)

    nll = 0.5 * torch.log(math.pi / v) \
        - alpha * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + two_b_lambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll


def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    kl = 0.5 * (a1 - 1.0) / b1 * (v2 * (mu2 - mu1) ** 2) \
        + 0.5 * (v2 / v1) \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 \
        + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return kl


def tilted_loss(q, e):
    return torch.maximum(q * e, (q - 1.0) * e)


def nig_regularization(y, mu, v, alpha, beta, wi_mean, quantile, lambda_reg=0.01, reduce=True, use_kl=False):
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    error = tilted_loss(quantile, y - mu)

    if use_kl:
        kl_val = kl_nig(mu, v, alpha, beta, mu, lambda_reg, 1.0 + lambda_reg, beta)
        reg = error * kl_val
    else:
        evidential_term = 2.0 * v + alpha + 1.0 / beta
        reg = error * evidential_term

    return torch.mean(reg) if reduce else reg


def quantile_evidential_loss(y_true, mu, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    wi_mean = beta / (alpha - 1.0)
    mu_adj = mu + theta * wi_mean

    loss_nll = nig_nll(y_true, mu_adj, v, alpha, beta, wi_mean, quantile, reduce)
    loss_reg = nig_regularization(y_true, mu, v, alpha, beta, wi_mean, quantile, reduce)
    return loss_nll + coeff * loss_reg


class EDLQuantileRegressor:
    def __init__(self, tau_low=0.05, tau_high=0.95, learning_rate=5e-4):
        torch.manual_seed(42)
        self.model = EDLQRNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.quantiles = [tau_low, 0.5, tau_high]
        self.coeff = 0.05

    def loss_function(self, y, mu, v, alpha, beta):
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            total_loss += quantile_evidential_loss(
                y, mu[:, i].unsqueeze(1), v[:, i].unsqueeze(1),
                alpha[:, i].unsqueeze(1), beta[:, i].unsqueeze(1),
                q, coeff=self.coeff
            )
        return total_loss

    def train(self, x, y, num_epochs=5000):
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            mu, v, alpha, beta = self.model(x)
            loss = self.loss_function(y, mu, v, alpha, beta)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, v, alpha, beta = self.model(x)
            mu_low, mu_mid, mu_high = torch.unbind(mu, dim=1)
            v_low, v_mid, v_high = torch.unbind(v, dim=1)
            alpha_low, alpha_mid, alpha_high = torch.unbind(alpha, dim=1)
            beta_low, beta_mid, beta_high = torch.unbind(beta, dim=1)

            aleatoric = beta_mid / (alpha_mid - 1.0)
            epistemic_mid = beta_mid / (v_mid * (alpha_mid - 1.0))
            epistemic_low = beta_low / (v_low * (alpha_low - 1.0))
            epistemic_high = beta_high / (v_high * (alpha_high - 1.0))
            uncertainty = epistemic_mid

        plt.figure(figsize=(10, 6))
        plt.plot(x, epistemic_mid, label='Mid', color='orange')
        plt.plot(x, epistemic_low, label='Low', color='red')
        plt.plot(x, epistemic_high, label='High', color='green')
        plt.legend()
        plt.title('Epistemic Uncertainty')
        plt.show()

        return mu_mid.numpy().squeeze(), mu_high.numpy().squeeze(), mu_low.numpy().squeeze(), uncertainty.numpy().squeeze()
