import torch
import torch.nn as nn
import torch.nn.functional as F

class QR_net(nn.Module):
    def __init__(self, input_dim, num_neurons=50, num_layers=1, activation=nn.ReLU()):
        super(QR_net, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_neurons, 3)

    def forward(self, x):
        x = self.hidden(x)
        output = self.output(x)
        q_low, q_mid, q_high = torch.chunk(output, 3, dim=-1)
        return q_low, q_mid, q_high
class QREnsemble_model(nn.Module):
    def __init__(self, input_shape, num_neurons=50, num_layers=1, activation=nn.ReLU(), num_ensembles=5, sigma=True):
        super(QREnsemble_model, self).__init__()
        self.num_ensembles = num_ensembles
        input_dim = input_shape
        self.models = nn.ModuleList([
            QR_net(input_dim, num_neurons, num_layers, activation) for _ in range(num_ensembles)
        ])

    def forward(self, x):
        q_lows = []
        q_mids = []
        q_highs = []
        for model in self.models:
            q_low, q_mid, q_high = model(x)
            q_lows.append(q_low)
            q_mids.append(q_mid)
            q_highs.append(q_high)
        return q_lows, q_mids, q_highs