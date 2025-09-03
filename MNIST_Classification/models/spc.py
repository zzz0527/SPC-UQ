import torch.nn as nn

class SPC_cls_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=256, num_layers=2, activation=nn.ReLU()):
        super(SPC_cls_net, self).__init__()
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
        self.hidden_pred = nn.Linear(num_neurons, 64)
        self.output_pred = nn.Linear(64, output_shape)
        self.hidden_mar = nn.Linear(num_neurons, 64)
        self.output_mar = nn.Linear(64, output_shape)

    def forward(self, x):
        x = x.view(-1, self.Flattening_shape)
        x = self.hidden(x)
        hid_pred= self.hidden_pred(x)
        hid_pred = nn.ReLU()(hid_pred)
        pred = self.output_pred(hid_pred)
        hid_mar= self.hidden_mar(x)
        hid_mar = nn.ReLU()(hid_mar)
        mar = self.output_mar(hid_mar)
        return pred, mar