import torch.nn as nn
import torch
import torch.nn.functional as F

class DenseNormal(nn.Module):
    def __init__(self, in_features, units):
        super(DenseNormal, self).__init__()
        self.units = units
        self.dense = nn.Linear(in_features, 2 * units)

    def forward(self, x):
        output = self.dense(x)
        mu, sigma = torch.chunk(output, 2, dim=-1)
        sigma = F.softplus(sigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)

class Dropout_model(nn.Module):
    def __init__(self, input_shape, num_neurons=50, num_layers=1, activation=nn.ReLU(), drop_prob=0.2, sigma=True):
        super(Dropout_model, self).__init__()
        layers = []
        in_features = input_shape
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            layers.append(nn.Dropout(p=drop_prob))
            in_features = num_neurons

        self.n_out = 2 if sigma else 1
        if self.n_out == 2:
            layers.append(DenseNormal(in_features=num_neurons, units=1))  # 输出 mu 和 sigma
        else:
            layers.append(nn.Linear(in_features, self.n_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.n_out == 2:
            mu_sigma = self.model(x)
            mu, sigma = torch.chunk(mu_sigma, 2, dim=-1)
            return mu, sigma
        else:
            output = self.model(x)
            return output