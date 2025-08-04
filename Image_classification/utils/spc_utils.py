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

class MAR_net(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons=512, num_layers=1, activation=nn.ReLU(), dropout_p=0.2):
        super(MAR_net, self).__init__()

        def make_branch():
            layers = []
            in_features = input_shape
            neurons = num_neurons
            for _ in range(num_layers):
                layers.append(nn.Linear(in_features, neurons))
                layers.append(activation)
                # layers.append(nn.Dropout(dropout_p))
                in_features = neurons
                # neurons //= 2
            return nn.Sequential(*layers), nn.Linear(in_features, output_shape)

        self.hidden_mar, self.mar = make_branch()
        self.hidden_mar_up, self.mar_up = make_branch()
        self.hidden_mar_down, self.mar_down = make_branch()

    def forward(self, x):
        mar = self.mar(self.hidden_mar(x))
        mar_up = self.mar_up(self.hidden_mar_up(x))
        mar_down = self.mar_down(self.hidden_mar_down(x))
        return mar, mar_up, mar_down


def SPC_fit(net_last_layer, topt, embeddings, labels, feature_dim, num_classes, device):
    inputshape=feature_dim
    if num_classes>=1000:
        MAR_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
        epochs = 100
        learning_rate = 1e-5
        batch_size=128
    elif num_classes>=100:
        MAR_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
        epochs=300
        learning_rate=1e-4
        batch_size=128
    else:
        MAR_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
        epochs=300
        learning_rate=1e-4
        batch_size=128
    MAR_model.to(device)
    # optimizer = optim.SGD(MAR_model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(MAR_model.parameters(), lr=learning_rate)#, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
    criterion = nn.MSELoss()
    criterion_no_reduction = torch.nn.MSELoss(reduction='none')

    labels = labels.to(torch.long)
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()#*num_classes
    dataset = TensorDataset(embeddings, labels_one_hot)
    embedding_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    import psutil
    import os
    def get_cpu_memory_mb():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 ** 2

    train_loss = 0
    num_samples = 0
    MAR_model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(embedding_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # data = data.to(MAR_model.hidden[0].weight.dtype)
            data = data.to(MAR_model.hidden_mar[0].weight.dtype)
            mar, mar_up, mar_down = MAR_model(data)
            # mean=F.softmax(net_last_layer(data)/topt, dim=1)
            mean=F.softmax(net_last_layer(data), dim=1)#*num_classes
            confidences, predictions = torch.max(mean, dim=1)
            preds_one_hot = torch.nn.functional.one_hot(predictions, num_classes=num_classes).float() *num_classes
            # mean = torch.round(mean * num_classes) / num_classes
            # print(mean)
            mar_targets = abs(targets - mean.detach())
            # mar_targets = 2 * mean.detach() * (1 - mean.detach())
            # loss_mar = criterion(mar, mar_targets)
            loss_values_mar = criterion_no_reduction(mar, mar_targets)
            weights = torch.ones_like(mar_targets, device=device)
            # weights += targets*num_classes
            # weights+=preds_one_hot
            # print(weights)
            loss_mar = (loss_values_mar * weights).mean()

            mask_up = (targets > 0).float()
            mask_down = (targets <= 0).float()
            mar_up_targets = (targets - mean).detach() * mask_up
            mar_down_targets = (mean).detach() * mask_down

            loss_mar_up = criterion_no_reduction(mar_up, mar_up_targets)
            # weights = torch.ones_like(mar_up_targets, device=device)
            # weights += targets*num_classes
            loss_mar_up = (loss_mar_up * weights).mean()

            loss_mar_down = criterion_no_reduction(mar_down, mar_down_targets)
            # weights = torch.ones_like(mar_down_targets, device=device)#*(num_classes+1)
            # weights -= targets*num_classes
            # weights += targets*num_classes
            loss_mar_down = (loss_mar_down * weights).mean()

            # loss_mar_up = criterion(mar_up, mar_up_targets)
            # loss_mar_down = criterion(mar_down, mar_down_targets)


            # print(loss_mar, loss_mar_up, loss_mar_down)
            loss = loss_mar + loss_mar_up + loss_mar_down
            loss.backward()

            # torch.cuda.synchronize()
            # gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            # gpu_max = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            # cpu_mem = get_cpu_memory_mb()
            # print(
            #     f"[Epoch {epoch} | Batch {batch_idx}] GPU: {gpu_mem:.2f} MB (Max: {gpu_max:.2f} MB) | CPU: {cpu_mem:.2f} MB")

            train_loss += loss.item()
            optimizer.step()
            num_samples += len(data)
        if num_classes >= 1000:
            print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
        scheduler.step()

    return MAR_model


def SPC_load(path, feature_dim, num_classes, device):
    inputshape=feature_dim
    if num_classes>=1000:
        loaded_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
    elif num_classes>=100:
        loaded_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())
    else:
        loaded_model = MAR_net(inputshape, output_shape=num_classes, num_neurons=inputshape, num_layers=1, activation=nn.ReLU())

    loaded_model.load_state_dict(torch.load(path))
    loaded_model.to(device)
    loaded_model.eval()

    return loaded_model

def SPC_evaluate(net, MAR_model, loader, feature_dim, num_classes, device):
    test_embeddings, test_labels = get_embeddings(
        net,
        loader,
        num_dim=feature_dim,
        dtype=torch.double,
        device=device,
        storage_device=device,
    )
    test_embeddings = test_embeddings.to(MAR_model.hidden_mar[0].weight.dtype)
    MAR_model.eval()
    mar, mar_up, mar_down = MAR_model(test_embeddings)
    logits, _ = get_logits_labels(net, loader, device)
    return logits, [logits.detach(), mar.detach(), mar_up.detach(), mar_down.detach()]