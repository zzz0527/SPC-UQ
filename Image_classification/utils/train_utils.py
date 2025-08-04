"""
This module contains methods for training models.
"""

import torch
from torch.nn import functional as F
from torch import nn


def evidential_loss(alpha, target, lam=0.0001):
    num_classes = alpha.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    S = alpha.sum(dim=1, keepdim=True)

    log_likelihood = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    kl_divergence = lam * torch.sum((alpha - 1) * (1 - target_one_hot), dim=1)

    loss = log_likelihood + kl_divergence
    return torch.mean(loss)

def mar_loss(logits, target):
    pred=logits[0]
    mar=logits[1]
    mar_up=logits[2]
    mar_down=logits[3]

    loss_cls = F.cross_entropy(pred, target)
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    mean = F.softmax(pred, dim=1)  # *num_classes
    confidences, predictions = torch.max(mean, dim=1)
    mar_targets = abs(target_one_hot - mean.detach())
    loss_values_mar = nn.MSELoss()(mar, mar_targets)

    mask_up = (target_one_hot > 0).float()
    mask_down = (target_one_hot <= 0).float()
    mar_up_targets = (target_one_hot - mean).detach() * mask_up
    mar_down_targets = (mean).detach() * mask_down

    loss_mar_up = nn.MSELoss()(mar_up, mar_up_targets)
    loss_mar_down = nn.MSELoss()(mar_down, mar_down_targets)

    loss = loss_cls + loss_values_mar + loss_mar_up + loss_mar_down
    return torch.mean(loss)

def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction='mean'):
    """
    Compute focal loss for multi-class classification.
    Args:
        logits: Tensor of shape [B, C], raw outputs from model.
        targets: Tensor of shape [B], integer class labels.
        gamma: focusing parameter.
        alpha: class weighting factor. None for uniform.
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        Tensor: focal loss value
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # shape: [B]
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)         # shape: [B]
    focal_weight = (1 - pt) ** gamma

    if alpha is not None:
        if isinstance(alpha, (list, torch.Tensor)):
            alpha = torch.tensor(alpha, device=logits.device)
        at = alpha[targets]
        loss = at * focal_weight * ce_loss
    else:
        loss = focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # 'none'

loss_function_dict = {
    "cross_entropy": F.cross_entropy,
    "edl_loss": evidential_loss,
    "focal_loss": focal_loss,
    "mar_loss": mar_loss,
}


def train_single_epoch(
    epoch, model, train_loader, optimizer, device, loss_function="cross_entropy", loss_mean=False,
):
    """
    Util method for training a model for a single epoch.
    """
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = loss_function_dict[loss_function](logits, labels)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        # if batch_idx % log_interval == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch,
        #             batch_idx * len(data),
        #             len(train_loader) * len(data),
        #             100.0 * batch_idx / len(train_loader),
        #             loss.item(),
        #         )
        #     )

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
    return train_loss / num_samples


def test_single_epoch(epoch, model, test_val_loader, device, loss_function="cross_entropy"):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for data, labels in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += loss_function_dict[loss_function](logits, labels).item()
            num_samples += len(data)

    print("======> Test set loss: {:.4f}".format(loss / num_samples))
    return loss / num_samples


def model_save_name(model_name, sn, mod, coeff, seed):
    if sn:
        if mod:
            strn = "_sn_" + str(coeff) + "_mod_"
        else:
            strn = "_sn_" + str(coeff) + "_"
    else:
        if mod:
            strn = "_mod_"
        else:
            strn = "_"

    return str(model_name) + strn + str(seed)
