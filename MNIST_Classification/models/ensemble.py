import torch
import torch.nn as nn
import torch.nn.functional as F

class single_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=256, num_layers=2, activation=nn.ReLU()):
        super(single_net, self).__init__()
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
        self.output = nn.Linear(in_features, output_shape)

    def forward(self, x):
        x = x.view(-1, self.Flattening_shape)
        x = self.hidden(x)
        output = self.output(x)
        return output

class Ensemble_cls_model(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=256, num_layers=2, activation=nn.ReLU(), num_ensembles=5):
        super(Ensemble_cls_model, self).__init__()
        self.num_ensembles = num_ensembles
        self.models = nn.ModuleList([
            single_net(input_shape, output_shape, num_neurons, num_layers, activation) for _ in range(num_ensembles)])

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        return outputs