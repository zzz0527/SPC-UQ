import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = nn.Linear(in_features=in_features, out_features=4 * self.units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.chunk(output, chunks=4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def extra_repr(self):
        return f'units={self.units}'

class QRevidential_net(nn.Module):
    def __init__(self, input_shape, num_quantiles=3, num_neurons=50, num_layers=1, activation=nn.ReLU()):
        super(QRevidential_net, self).__init__()
        layers = []
        in_features = input_shape
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons

        layers.append(DenseNormalGamma(in_features=num_neurons, units=num_quantiles))  # 输出 mu 和 sigma

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        mu, v, alpha, beta = torch.chunk(output, 4, dim=-1)
        return mu, v, alpha, beta