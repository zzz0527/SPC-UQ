"""
Metrics to measure classification performance
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from utils.ensemble_utils import ensemble_forward_pass

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def evidential_loss(alpha, target, lambda_reg=0.001):
    num_classes = alpha.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    S = alpha.sum(dim=1, keepdim=True)

    log_likelihood = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    kl_divergence = lambda_reg * torch.sum((alpha - 1) * (1 - target_one_hot), dim=1)

    loss = log_likelihood + kl_divergence
    return torch.mean(loss)

def create_adversarial_dataloader(model, data_loader, device, epsilon=0.03, batch_size=32, edl=False, joint=False):
    adv_examples = []
    adv_labels = []

    model.eval()
    for data, label in data_loader:
        data = data.to(device).detach().requires_grad_(True)
        label = label.to(device)

        model.zero_grad()
        logit = model(data)
        if edl:
            loss = evidential_loss(logit, label)
        if joint:
            loss = F.cross_entropy(logit[0], label)
        else:
            loss = F.cross_entropy(logit, label)
        loss.backward()

        signed_grad = data.grad.sign()
        # print(data)
        # data_adv = torch.clamp(data + epsilon * signed_grad, 0, 1)
        data_adv = data + epsilon * signed_grad

        # print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
        # print(data.shape)
        # print(data_adv.shape)
        # img = data[0].detach().cpu().numpy()  # 变为 numpy 数组
        # img = np.transpose(img, (1, 2, 0))  # 变换维度 [C, H, W] -> [H, W, C]
        # plt.imshow(img)
        # plt.axis("off")  # 关闭坐标轴
        # plt.title("Sample Image")
        # plt.show()
        # img = data_adv[0].detach().cpu().numpy()  # 变为 numpy 数组
        # img = np.transpose(img, (1, 2, 0))  # 变换维度 [C, H, W] -> [H, W, C]
        # plt.imshow(img)
        # plt.axis("off")  # 关闭坐标轴
        # plt.title("Sample Image")
        # plt.show()

        adv_examples.append(data_adv.detach().cpu())
        adv_labels.append(label.detach().cpu())

    adv_examples = torch.cat(adv_examples, dim=0)
    adv_labels = torch.cat(adv_labels, dim=0)

    adv_dataset = TensorDataset(adv_examples, adv_labels)
    adv_dataloader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)

    return adv_dataloader


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

def get_logits_labels_uq(model, data_loader, device):
    """
    Utility function to get logits as a list: [pred_all, mar_all, mar_up_all, mar_down_all]
    and labels as a single tensor.
    """
    model.eval()
    preds = []
    mars = []
    mars_up = []
    mars_down = []
    labels = []

    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            pred, mar, mar_up, mar_down = model(data)
            preds.append(pred)
            mars.append(mar)
            mars_up.append(mar_up)
            mars_down.append(mar_down)
            labels.append(label)

    # 分别拼接每一类输出
    pred_all = torch.cat(preds, dim=0)
    mar_all = torch.cat(mars, dim=0)
    mar_up_all = torch.cat(mars_up, dim=0)
    mar_down_all = torch.cat(mars_down, dim=0)
    labels_all = torch.cat(labels, dim=0)

    logits = [pred_all, mar_all, mar_up_all, mar_down_all]
    return logits, labels_all


def test_classification_net_softmax(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.detach().cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net_logits(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels)

def test_classification_uq(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels_uq(model, data_loader, device)
    return test_classification_net_logits(logits[0], labels)

def test_classification_net_ensemble(model_ensemble, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset
    for a deep ensemble.
    """
    for model in model_ensemble:
        model.eval()
    softmax_prob = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            softmax, _, _ = ensemble_forward_pass(model_ensemble, data)
            softmax_prob.append(softmax)
            labels.append(label)
    softmax_prob = torch.cat(softmax_prob, dim=0)
    labels = torch.cat(labels, dim=0)

    return test_classification_net_softmax(softmax_prob, labels)

def test_classification_net_logits_edl(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    predicted_probs = logits / logits.sum(dim=1, keepdim=True)
    confidence_vals, predictions = torch.max(predicted_probs, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )

def test_classification_net_edl(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits_edl(logits, labels)