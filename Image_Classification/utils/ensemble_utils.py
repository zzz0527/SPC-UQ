"""
Utilities for processing a deep ensemble.
"""
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from net.vgg import vgg16
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn

from metrics.uncertainty_confidence import entropy_prob, mutual_information_prob


models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


def load_ensemble(ensemble_loc, model_name, device, num_classes=10, ensemble_len=5, num_epochs=350, seed=1, **kwargs):
    ensemble = []
    cudnn.benchmark = True
    for e in range(ensemble_len):
        g = torch.Generator().manual_seed(seed*10+e)
        i = torch.randint(0, 25, (1,), generator=g).item()
        ensemble_loc2 = os.path.join(ensemble_loc, ("Run" + str(i + 1)))
        sn_enabled = kwargs.get('spectral_normalization', False)
        coeff = kwargs.get('coeff', None)
        if sn_enabled:
            load_name=ensemble_loc2 + '/' + model_name + "_sn_" + str(coeff) + "_mod_" + str(1+i) + "_" + str(num_epochs) + ".model"
        else:
            load_name=ensemble_loc2 + '/' + model_name + '_' + str(1+i) + "_" + str(num_epochs) + ".model"
        print(load_name)
        net = models[model_name](num_classes=num_classes, temp=1.0, **kwargs).to(device)
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.load_state_dict(
            torch.load(load_name)
        )
        ensemble.append(net)
    return ensemble


def ensemble_forward_pass(model_ensemble, data):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
    outputs = []
    for i, model in enumerate(model_ensemble):
        output = F.softmax(model(data), dim=1)
        outputs.append(torch.unsqueeze(output, dim=0))

    outputs = torch.cat(outputs, dim=0)
    mean_output = torch.mean(outputs, dim=0)
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)

    return mean_output, predictive_entropy, mut_info



def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader, disable=True):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


class MLP_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=512, num_layers=1, activation=nn.ReLU()):
        super(MLP_net, self).__init__()
        layers = []
        in_features = input_shape
        for _ in range(num_layers):
            num_neurons //= 2
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_neurons, output_shape)

    def forward(self, x):
        x = self.hidden(x)
        v = self.output(x)
        return v

class Ensemble_model(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=512, num_layers=1, activation=nn.ReLU(), num_ensembles=5):
        super(Ensemble_model, self).__init__()
        self.num_ensembles = num_ensembles
        self.models = nn.ModuleList([
            MLP_net(input_shape, output_shape, num_neurons, num_layers, activation) for _ in range(num_ensembles)
        ])

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        return outputs


def Ensemble_fit(embeddings, labels, feature_dim, num_classes, device):
    inputshape=feature_dim
    Ensembl_model = Ensemble_model(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
    epochs = 100
    learning_rate = 1e-4
    batch_size=128

    Ensembl_model.to(device)
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in Ensembl_model.models]

    dataset = TensorDataset(embeddings, labels)
    embedding_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_loss = 0
    num_samples = 0
    Ensembl_model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in embedding_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            for i, model in enumerate(Ensembl_model.models):
                optimizers[i].zero_grad()
                batch_x = batch_x.float()
                batch_y = batch_y.long()
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizers[i].step()

            train_loss += loss.item()
            num_samples += len(batch_x)
        if num_classes >= 1000:
            print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))

    return Ensembl_model

def Ensemble_load(path, inputshape, num_classes, device):
    Ensembl_model = Ensemble_model(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1,activation=nn.ReLU())

    Ensembl_model.load_state_dict(torch.load(path))
    Ensembl_model.to(device)
    Ensembl_model.eval()

    return Ensembl_model

def Ensemble_evaluate(net, Ensembl_model, loader, feature_dim, device):
    test_embeddings, test_labels = get_embeddings(
        net,
        loader,
        num_dim=feature_dim,
        dtype=torch.double,
        device=device,
        storage_device=device,
    )

    test_embeddings = test_embeddings.float()

    outputs = []
    for model in Ensembl_model.models:
        output = F.softmax(model(test_embeddings), dim=1)  # [N, C]
        outputs.append(torch.unsqueeze(output, dim=0))     # [1, N, C]

    outputs = torch.cat(outputs, dim=0)       # [M, N, C]
    mean_output = torch.mean(outputs, dim=0)  # [N, C]
    predictive_entropy = entropy_prob(mean_output)         # [N]
    mut_info = mutual_information_prob(outputs)            # [N]

    return mean_output, predictive_entropy, mut_info, test_labels
