import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNormal(nn.Module):
    def __init__(self, in_features, units):

        super(DenseNormal, self).__init__()
        self.units = units
        self.dense = nn.Linear(in_features, 2 * units)  # 输出均值和 logsigma

    def forward(self, x):
        output = self.dense(x)
        mu, logsigma = torch.chunk(output, 2, dim=-1)
        sigma = F.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)

class NLL_net(nn.Module):
    def __init__(self, input_dim, num_neurons=50, num_layers=1, activation=nn.ReLU()):
        super(NLL_net, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output = DenseNormal(in_features=num_neurons, units=1)  # 输出 mu 和 sigma

    def forward(self, x):
        x = self.hidden(x)
        mu_sigma = self.output(x)
        mu, sigma = torch.chunk(mu_sigma, 2, dim=-1)
        return mu, sigma

class Ensemble_model(nn.Module):
    def __init__(self, input_shape, num_neurons=50, num_layers=1, activation=nn.ReLU(), num_ensembles=5, sigma=True):
        super(Ensemble_model, self).__init__()
        self.num_ensembles = num_ensembles
        input_dim = input_shape
        self.models = nn.ModuleList([
            NLL_net(input_dim, num_neurons, num_layers, activation) for _ in range(num_ensembles)
        ])

    def forward(self, x):
        mus = []
        sigmas = []
        for model in self.models:
            mu, sigma = model(x)
            mus.append(mu)
            sigmas.append(sigma)
        return mus, sigmas