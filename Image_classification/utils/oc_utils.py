import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

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

def oc_fit(embeddings, device, certs_loss="bce",certs_k=100,certs_epochs=20,certs_reg=1,certs_bias=0):
    """
    Compute uncertainty using linear certificates (ours)
    """

    def target(x):
        return torch.zeros(x.size(0), certs_k, device=x.device)

    certificates = torch.nn.Linear(embeddings.size(1), certs_k, bias=certs_bias)
    certificates.to(device)
    opt = torch.optim.Adam(certificates.parameters())

    if certs_loss == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    elif certs_loss == "mse":
        loss = torch.nn.L1Loss(reduction="none")

    loader_f = DataLoader(TensorDataset(embeddings, embeddings),
                          batch_size=64,
                          shuffle=True)

    import psutil
    import os
    def get_cpu_memory_mb():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 ** 2

    for epoch in range(certs_epochs):
        for f, _ in loader_f:
            f=f.float()
            opt.zero_grad()
            error = loss(certificates(f), target(f)).mean()
            penalty = certs_reg * (
                (certificates.weight @ certificates.weight.t() -
                 torch.eye(certs_k).to(certificates.weight.device)).pow(2).mean()
            )
            (error + penalty).backward()
            opt.step()

            # torch.cuda.synchronize()
            # gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            # gpu_max = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            # cpu_mem = get_cpu_memory_mb()
            # print(f"GPU: {gpu_mem:.2f} MB (Max: {gpu_max:.2f} MB) | CPU: {cpu_mem:.2f} MB")

    return certificates

def oc_evaluate(net, oc_model, loader, feature_dim, device):
    test_embeddings, test_labels = get_embeddings(
        net,
        loader,
        num_dim=feature_dim,
        dtype=torch.double,
        device=device,
        storage_device=device,
    )
    test_embeddings = test_embeddings.float()
    output = oc_model(test_embeddings)
    logits, _ = get_logits_labels(net, loader, device)
    sig = torch.nn.Sigmoid()

    return logits, sig(output).pow(2).mean(1).detach()