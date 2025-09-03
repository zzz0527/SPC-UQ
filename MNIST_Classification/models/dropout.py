import torch.nn as nn
import torch
import torch.nn.functional as F

class Dropout_cls_model(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=256, num_layers=2, activation=nn.ReLU(), drop_prob=0.1):
        super(Dropout_cls_model, self).__init__()
        layers = []
        in_features = 1
        for num in input_shape:
            in_features *= num
        self.Flattening_shape = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            layers.append(nn.Dropout(p=drop_prob))
            in_features = num_neurons
        layers.append(nn.Linear(in_features, output_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.Flattening_shape)
        output = self.model(x)
        return output