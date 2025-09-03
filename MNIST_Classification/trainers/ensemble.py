import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ensemble:
    def __init__(self, model, learning_rate=1e-3, optimizer_type='ADAM'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizers = self._init_optimizers(model.models, optimizer_type, learning_rate)

        self.epoch = 0
        self.separate_training = False
        self.max_acc = -1
        self.max_acc_confident = -1
        self.max_acc_uncertain = -1
        self.max_auroc = -1

    def _init_optimizers(self, models, optimizer_type, lr):
        if optimizer_type == 'ADAM':
            return [optim.Adam(m.parameters(), lr=lr) for m in models]
        elif optimizer_type == 'SGD':
            return [optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) for m in models]
        else:
            raise ValueError("Unsupported optimizer type")

    def run_train_step_single(self, data, target, model_index):
        self.model.train()
        model = self.model.models[model_index]
        optimizer = self.optimizers[model_index]
        logits = model(data)
        loss = self.criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def run_train_step_all(self, data, target, num_classes, verbose=True):
        self.model.train()
        losses = []
        one_hot_targets = F.one_hot(target, num_classes=num_classes)
        for model, optimizer in zip(self.model.models, self.optimizers):
            logits = model(data)
            loss = self.criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logits = self.model(data)
        if verbose:
            stacked_logits = torch.stack([torch.softmax(logit, dim=1) for logit in logits], dim=0)
            uncertainty = torch.std(stacked_logits, dim=0, unbiased=False)[one_hot_targets != 0].mean().item()
        else:
            uncertainty = 0
        return sum(losses) / len(losses), uncertainty

    def evaluate(self, test_loader, ood_loader, num_classes, threshold):
        self.model.eval()
        y, logits_list = [], []
        with torch.no_grad():
            start_time = time.time()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = self.model(data)
                softmax_logits = [torch.softmax(logit, dim=1) for logit in logits]
                y.append(target)
                logits_list.append(softmax_logits)

        y = torch.cat(y, dim=0)
        transposed = zip(*logits_list)
        stacked = [torch.cat(tensors, dim=0) for tensors in transposed]
        stacked_logits = torch.stack(stacked, dim=0)
        mean_logits = stacked_logits.mean(dim=0)
        confidences, predictions = torch.max(mean_logits, dim=1)
        one_hot_preds = F.one_hot(predictions, num_classes=num_classes)
        uncertainty = torch.std(stacked_logits, dim=0, unbiased=False)[one_hot_preds != 0]
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
                logits = self.model(data)
                softmax_logits = [torch.softmax(logit, dim=1) for logit in logits]
                logits_list.append(softmax_logits)

        transposed = zip(*logits_list)
        stacked = [torch.cat(tensors, dim=0) for tensors in transposed]
        stacked_logits_ood = torch.stack(stacked, dim=0)
        mean_logits_ood = stacked_logits_ood.mean(dim=0)
        confidences_ood, preds_ood = torch.max(mean_logits_ood, dim=1)
        one_hot_preds_ood = F.one_hot(preds_ood, num_classes=num_classes)
        uncertainty_ood = torch.std(stacked_logits_ood, dim=0, unbiased=False)[one_hot_preds_ood != 0]

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
            epoch_loss = 0.0
            epoch_uncertainty = 0.0
            start_time = time.time()

            if not self.separate_training:
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss, uncertainty = self.run_train_step_all(data, target, num_classes, verbose)
                    epoch_loss += loss
                    epoch_uncertainty += uncertainty

            else:
                for i in range(len(self.model.models)):
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        loss = self.run_train_step_single(data, target, i)
                        epoch_loss += loss

                if verbose:
                    for data, target in train_loader:
                        one_hot_targets = F.one_hot(target, num_classes=num_classes)
                        logits = self.model(data)
                        stacked_logits = torch.stack([torch.softmax(logit, dim=1) for logit in logits], dim=0)
                        mean_logits = stacked_logits.mean(dim=0)
                        loss = self.criterion(mean_logits, target).item()
                        uncertainty = torch.std(stacked_logits, dim=0, unbiased=False)[one_hot_targets != 0].mean()
                        epoch_uncertainty += uncertainty.item()

            train_times.append((time.time() - start_time) * 1000)
            avg_loss = epoch_loss / len(train_loader)
            avg_uncertainty = epoch_uncertainty / len(train_loader)
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
