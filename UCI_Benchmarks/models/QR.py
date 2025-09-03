import torch
import torch.nn as nn
import torch.nn.functional as F

class QR_net(nn.Module):
    def __init__(self, input_shape, num_neurons=50, num_layers=1, activation=nn.ReLU()):
        super(QR_net, self).__init__()
        layers = []
        in_features = input_shape
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

    def features(self, x):
        return self.hidden(x)