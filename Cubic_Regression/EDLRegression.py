import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import t as student_t
import numpy as np
import math


class EDLRegressionNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_mu = nn.Linear(hidden_dim, output_dim)
        self.output_logv = nn.Linear(hidden_dim, output_dim)
        self.output_alpha = nn.Linear(hidden_dim, output_dim)
        self.output_beta = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        mu = self.output_mu(x)
        v = F.softplus(self.output_logv(x))
        alpha = F.softplus(self.output_alpha(x)) + 1.0 + 1e-6
        beta = F.softplus(self.output_beta(x))
        return mu, v, alpha, beta


def nig_nll(y, mu, v, alpha, beta, reduce=True):
    two_b_lambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + two_b_lambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll


def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    eps = 1e-6
    v1 = torch.clamp(v1, min=eps)
    v2 = torch.clamp(v2, min=eps)
    b1 = torch.clamp(b1, min=eps)
    b2 = torch.clamp(b2, min=eps)

    term1 = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2)
    term2 = 0.5 * v2 / v1
    term3 = -0.5 * torch.log(v2 / v1)
    term4 = -0.5
    term5 = a2 * torch.log(b1 / b2)
    term6 = -(torch.lgamma(a1) - torch.lgamma(a2))
    term7 = (a1 - a2) * torch.digamma(a1)
    term8 = -(b1 - b2) * a1 / b1

    kl = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
    return kl


def nig_regularization(y, mu, v, alpha, beta, omega=0.01, reduce=True, use_kl=False):
    error = torch.abs(y - mu)

    if use_kl:
        kl = kl_nig(mu, v, alpha, beta, mu, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evidential = 2 * v + alpha
        reg = error * evidential

    return reg.mean() if reduce else reg


def edl_loss(y, mu, v, alpha, beta, lam=0.0, reduce=True, return_components=False):
    nll = nig_nll(y, mu, v, alpha, beta, reduce=reduce)
    reg = nig_regularization(y, mu, v, alpha, beta, reduce=reduce)
    loss = nll  # optionally: loss = nll + lam * reg

    return (loss, (nll, reg)) if return_components else loss


def predictive_interval(mu, v, alpha, beta, confidence=0.95):
    mu = torch.as_tensor(mu)
    v = torch.as_tensor(v)
    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)

    dof = 2.0 * alpha
    scale = torch.sqrt((1.0 + v) * beta / (alpha * v))
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q

    t_l = student_t.ppf(lower_q, df=dof.cpu().numpy())
    t_u = student_t.ppf(upper_q, df=dof.cpu().numpy())

    lower = mu + torch.from_numpy(t_l).to(mu.device) * scale
    upper = mu + torch.from_numpy(t_u).to(mu.device) * scale

    return lower, upper


class EDLRegressor:
    def __init__(self, learning_rate=5e-4):
        torch.manual_seed(33)
        self.model = EDLRegressionNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = edl_loss
        self.lambda_ = 0.01

    def train(self, x, y, num_epochs=5000):
        torch.manual_seed(33)
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            mu, v, alpha, beta = self.model(x)
            loss = self.criterion(y, mu, v, alpha, beta, lam=self.lambda_)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, v, alpha, beta = self.model(x)

            # Uncertainty decomposition
            aleatoric = torch.sqrt(beta / (alpha - 1.0 + 1e-6))
            epistemic = torch.sqrt(beta / (v * (alpha - 1.0 + 1e-6)))

            mu_np = mu.detach().cpu().numpy()
            aleatoric_np = aleatoric.detach().cpu().numpy()
            epistemic_np = epistemic.detach().cpu().numpy()

            lower, upper = predictive_interval(mu, v, alpha, beta, confidence=0.95)
            lower_np = lower.detach().cpu().numpy()
            upper_np = upper.detach().cpu().numpy()

        return mu_np.squeeze(), upper_np.squeeze(), lower_np.squeeze(), epistemic_np.squeeze()
