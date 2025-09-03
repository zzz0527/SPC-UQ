import torch
import torch.nn as nn
import torch.nn.functional as F

class SPC_net(nn.Module):
    def __init__(self, input_shape, num_neurons=50, num_layers=1, activation=nn.ReLU()):
        super(SPC_net, self).__init__()
        layers = []
        in_features = input_shape
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output1 = nn.Linear(num_neurons, 1)
        self.hidden2 = nn.Linear(num_neurons, 50)
        self.output2 = nn.Linear(50, 5)  # mae, mae_up, mae_down


    def forward(self, x):
        x = self.hidden(x)
        v = self.output1(x)
        x = self.hidden2(x)
        x = nn.ReLU()(x)
        output = self.output2(x)
        mae, mae_up, mae_down, q_up, q_down = torch.chunk(output, 5, dim=-1)
        mae = F.softplus(mae) + 1e-6
        mae_up = F.softplus(mae_up) + 1e-6
        mae_down = F.softplus(mae_down) + 1e-6
        q_up = F.softplus(q_up) + 1e-6
        q_down = F.softplus(q_down) + 1e-6
        return v, mae, mae_up, mae_down, q_up, q_down