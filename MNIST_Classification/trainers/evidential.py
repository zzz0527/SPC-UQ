import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evidential_loss(alpha, beta, nu, gamma, target, num_classes, lambda_reg=0.01):
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    S = alpha.sum(dim=1, keepdim=True)
    log_likelihood = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    kl_divergence = lambda_reg * torch.sum((alpha - 1) * (1 - target_one_hot), dim=1)
    return torch.mean(log_likelihood + kl_divergence)


def compute_uncertainty(alpha, num_classes):
    return num_classes / alpha.sum(dim=1)


class Evidential:
    def __init__(self, model, learning_rate=1e-3, optimizer_type='ADAM'):
        self.model = model.to(device)
        self.criterion = evidential_loss
        if optimizer_type == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        self.epoch = 0
        self.max_acc = -1
        self.max_acc_confident = -1
        self.max_acc_uncertain = -1
        self.max_auroc = -1

    def run_train_step(self, data, target, num_classes, verbose=True):
        self.model.train()
        alpha, beta, nu, gamma = self.model(data)
        alpha = torch.clamp(alpha, min=1e-6 + 1)
        loss = self.criterion(alpha, beta, nu, gamma, target, num_classes)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if verbose:
            uncertainty = compute_uncertainty(alpha, num_classes).mean().item()
        else:
            uncertainty = 0
        return loss.item(), uncertainty

    def evaluate(self, test_loader, ood_loader, num_classes, threshold):
        self.model.eval()
        y, alpha_list, prob_list = [], [], []

        with torch.no_grad():
            start_time = time.time()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                alpha, beta, nu, gamma = self.model(data)
                alpha = torch.clamp(alpha, min=1e-6 + 1)
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                y.append(target)
                alpha_list.append(alpha)
                prob_list.append(probs)

        y = torch.cat(y, dim=0)
        alpha = torch.cat(alpha_list, dim=0)
        probs = torch.cat(prob_list, dim=0)
        confidences, predictions = torch.max(probs, dim=1)
        uncertainty = compute_uncertainty(alpha, num_classes)
        test_time = (time.time() - start_time) * 1000

        threshold = np.quantile(uncertainty.cpu().numpy(), 0.5)
        correct_confident, correct_uncertain = 0, 0
        total_confident = (uncertainty < threshold).sum().item()
        total_uncertain = (uncertainty >= threshold).sum().item()

        for i in range(len(y)):
            if uncertainty[i] < threshold:
                correct_confident += int(predictions[i] == y[i])
            else:
                correct_uncertain += int(predictions[i] == y[i])

        accuracy = (predictions == y).sum().item() / len(y)
        acc_conf = correct_confident / total_confident if total_confident > 0 else 0
        acc_unc = correct_uncertain / total_uncertain if total_uncertain > 0 else 0

        alpha_ood = []
        with torch.no_grad():
            for data, _ in ood_loader:
                data = data.to(device)
                alpha_tmp, _, _, _ = self.model(data)
                alpha_ood.append(torch.clamp(alpha_tmp, min=1e-6 + 1))

        alpha_ood = torch.cat(alpha_ood, dim=0)
        uncertainty_ood = compute_uncertainty(alpha_ood, num_classes)

        bin_labels = torch.cat([torch.zeros_like(uncertainty), torch.ones_like(uncertainty_ood)])
        all_uncertainties = torch.cat([uncertainty, uncertainty_ood])
        auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), all_uncertainties.cpu().numpy())

        return accuracy, acc_conf, acc_unc, auroc, test_time

    def train(self, train_dataset, test_dataset, ood_dataset, num_classes,
              batch_size=128, num_epochs=40, verbose=True, freq=1):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=128, shuffle=False)

        loss_curve, uncertainty_curve = [], []
        acc_curve, acc_conf_curve, acc_unc_curve = [], [], []
        train_times, test_times = [], []

        for self.epoch in range(1, num_epochs + 1):
            total_loss, total_uncertainty = 0.0, 0.0
            start_time = time.time()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                loss, uncertainty = self.run_train_step(data, target, num_classes, verbose)
                total_loss += loss
                total_uncertainty += uncertainty
            train_times.append((time.time() - start_time) * 1000)

            avg_loss = total_loss / len(train_loader)
            avg_uncertainty = total_uncertainty / len(train_loader)
            loss_curve.append(avg_loss)
            uncertainty_curve.append(avg_uncertainty)

            if self.epoch % freq == 0:
                acc, acc_conf, acc_unc, auroc, test_time = self.evaluate(test_loader, ood_loader, num_classes, avg_uncertainty)
                acc_curve.append(acc)
                acc_conf_curve.append(acc_conf)
                acc_unc_curve.append(acc_unc)
                test_times.append(test_time)

                if acc > self.max_acc:
                    self.max_acc = acc
                    self.max_acc_confident = acc_conf
                    self.max_acc_uncertain = acc_unc
                    self.max_auroc = auroc

        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_curve, label='Train Loss')
            plt.title('Training Loss')
            plt.legend(); plt.grid(True)
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(uncertainty_curve, label='Train Uncertainty', color='orange')
            plt.title('Training Uncertainty')
            plt.legend(); plt.grid(True)
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(acc_curve, label='Test Accuracy')
            plt.plot(acc_conf_curve, label='Confident Accuracy')
            plt.plot(acc_unc_curve, label='Uncertain Accuracy')
            plt.title('Test Accuracy Curves')
            plt.legend(); plt.grid(True)
            plt.show()

        return (self.model, self.max_acc, self.max_acc_confident, self.max_acc_uncertain,
                self.max_auroc, np.mean(train_times), np.mean(test_times))
