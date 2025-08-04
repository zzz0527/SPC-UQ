import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from tqdm import tqdm

def get_logits_labels(model, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            logit = model(data)
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits, labels

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

def evidential_loss(alpha, target, lambda_reg=0.0001):
    num_classes = alpha.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    S = alpha.sum(dim=1, keepdim=True)

    log_likelihood = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    kl_divergence = lambda_reg * torch.sum((alpha - 1) * (1 - target_one_hot), dim=1)

    loss = log_likelihood + kl_divergence
    return torch.mean(loss)

class EDL_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=512, num_layers=1, activation=nn.ReLU()):
        super(EDL_net, self).__init__()
        layers = []
        in_features = input_shape
        for _ in range(num_layers):
            num_neurons//=2
            layers.append(nn.Linear(in_features, num_neurons))
            layers.append(activation)
            in_features = num_neurons
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_neurons, output_shape)

    def forward(self, x):
        x = self.hidden(x)
        v = self.output(x)
        v = F.softplus(v) + 1
        return v

def EDL_fit(embeddings, labels, feature_dim, num_classes, device):
    inputshape=feature_dim
    if num_classes>=1000:
        EDL_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=4096, num_layers=2, activation=nn.ReLU())
        epochs=100
        learning_rate=0.0001
    elif num_classes>=100:
        EDL_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=2048, num_layers=2, activation=nn.ReLU())
        epochs=100
        learning_rate=0.0001
    else:
        EDL_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=512, num_layers=1, activation=nn.ReLU())
        epochs=100
        learning_rate=0.0001
    EDL_model.to(device)
    optimizer = optim.Adam(EDL_model.parameters(), lr=learning_rate)
    criterion = evidential_loss

    import psutil
    import os
    def get_cpu_memory_mb():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 ** 2

    labels = labels.to(torch.long)  # 转换为 LongTensor
    dataset = TensorDataset(embeddings, labels)
    embedding_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(embedding_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            data = data.to(EDL_model.hidden[0].weight.dtype)
            alpha = EDL_model(data)
            loss = criterion(alpha, labels)
            loss.backward()
            optimizer.step()
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

            # torch.cuda.synchronize()
            # gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            # gpu_max = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            # cpu_mem = get_cpu_memory_mb()
            # print(
            #     f"[Epoch {epoch} | Batch {batch_idx}] GPU: {gpu_mem:.2f} MB (Max: {gpu_max:.2f} MB) | CPU: {cpu_mem:.2f} MB")

    return EDL_model


def EDL_load(path, feature_dim, num_classes, device):
    inputshape=feature_dim
    if num_classes>=1000:
        loaded_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=4096, num_layers=2, activation=nn.ReLU())
    elif num_classes>=100:
        loaded_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=2048, num_layers=2, activation=nn.ReLU())
    else:
        loaded_model = EDL_net(inputshape, output_shape=num_classes, num_neurons=512, num_layers=1, activation=nn.ReLU())

    loaded_model.load_state_dict(torch.load(path))
    loaded_model.to(device)
    loaded_model.eval()

    return loaded_model

def EDL_evaluate(net, EDL_model, loader, feature_dim, device):
    test_embeddings, test_labels = get_embeddings(
        net,
        loader,
        num_dim=feature_dim,
        dtype=torch.double,
        device=device,
        storage_device=device,
    )
    test_embeddings = test_embeddings.to(EDL_model.hidden[0].weight.dtype)
    alpha = EDL_model(test_embeddings)
    return alpha.detach(), test_labels.detach()