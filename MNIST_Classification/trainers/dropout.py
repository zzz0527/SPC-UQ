import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout:
    def __init__(self, model, learning_rate=1e-3, optimizer_type='ADAM'):
        self.mc_samples = 5
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._init_optimizer(optimizer_type, learning_rate)

        self.epoch = 0
        self.max_acc = -1
        self.max_acc_confident = -1
        self.max_acc_uncertain = -1
        self.max_auroc = -1

    def _init_optimizer(self, optimizer_type, lr):
        if optimizer_type == 'ADAM':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("Unsupported optimizer type")

    def run_train_step(self, data, target, num_classes, verbose=True):
        self.model.train()
        logits = self.model(data)
        loss = self.criterion(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if verbose:
            predictions = []
            one_hot_targets = F.one_hot(target, num_classes=num_classes)
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    logits = self.model(data)
                    predictions.append(torch.softmax(logits, dim=1))
            predictions = torch.stack(predictions, dim=0)
            uncertainty = torch.std(predictions, dim=0, unbiased=False)[one_hot_targets != 0].mean().item()
        else:
            uncertainty = 0

        return loss.item(), uncertainty

    def evaluate(self, test_loader, ood_loader, num_classes, threshold):
        self.model.train()
        y, logits_list = [], []
        with torch.no_grad():
            start_time = time.time()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                mc_logits = [torch.softmax(self.model(data), dim=1) for _ in range(self.mc_samples)]
                y.append(target)
                logits_list.append(mc_logits)

        y = torch.cat(y, dim=0)
        transposed = zip(*logits_list)
        stacked = [torch.cat(tensors, dim=0) for tensors in transposed]
        dropout_logits = torch.stack(stacked, dim=0)
        mean_logits = dropout_logits.mean(dim=0)
        confidences, predictions = torch.max(mean_logits, dim=1)
        one_hot_preds = F.one_hot(predictions, num_classes=num_classes)
        uncertainty = torch.std(dropout_logits, dim=0, unbiased=False)[one_hot_preds != 0]
        test_time = (time.time() - start_time) * 1000

        threshold = np.quantile(uncertainty.cpu().numpy(), 0.5)
        accuracy = (predictions == y).float().mean().item()

        cer_mask = (uncertainty < threshold).cpu().numpy()
        unc_mask = ~cer_mask
        acc_conf = (predictions[cer_mask] == y[cer_mask]).float().mean().item()
        acc_unc = (predictions[unc_mask] == y[unc_mask]).float().mean().item()

        logits_list = []
        with torch.no_grad():
            for data, _ in ood_loader:
                data = data.to(device)
                mc_logits = [torch.softmax(self.model(data), dim=1) for _ in range(self.mc_samples)]
                logits_list.append(mc_logits)

        transposed = zip(*logits_list)
        stacked = [torch.cat(tensors, dim=0) for tensors in transposed]
        dropout_logits_ood = torch.stack(stacked, dim=0)
        mean_logits_ood = dropout_logits_ood.mean(dim=0)
        confidences_ood, preds_ood = torch.max(mean_logits_ood, dim=1)
        one_hot_preds_ood = F.one_hot(preds_ood, num_classes=num_classes)
        uncertainty_ood = torch.std(dropout_logits_ood, dim=0, unbiased=False)[one_hot_preds_ood != 0]

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
