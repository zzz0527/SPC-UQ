import torch
import torch.nn as nn
import torch.nn.functional as F

class Evidential_cls_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=256, num_layers=2, activation=nn.ReLU()):
        super(Evidential_cls_net, self).__init__()
        layers = []
        in_features = 1
        for num in input_shape:
            in_features *= num
        self.Flattening_shape = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output_alpha = nn.Linear(num_neurons, output_shape)
        self.output_beta = nn.Linear(num_neurons, output_shape)
        self.output_nu = nn.Linear(num_neurons, output_shape)
        self.output_gamma = nn.Linear(num_neurons, output_shape)


    def forward(self, x):
        x = x.view(-1, self.Flattening_shape)
        x = self.hidden(x)
        alpha = self.output_alpha(x)
        beta = self.output_beta(x)
        nu = self.output_nu(x)
        gamma = self.output_gamma(x)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1e-6
        nu = F.softplus(nu) + 1e-6
        gamma = F.softplus(gamma) + 1e-6
        return alpha, beta, nu, gamma