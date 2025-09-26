import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SPC:
    """
    Classifier with Split-Point Self-Consistency uncertainty estimation (SPC).
    Supports training, evaluation, and uncertainty quantification.
    """
    def __init__(self, model, learning_rate=1e-3, optimizer_type='ADAM'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._init_optimizer(optimizer_type, learning_rate)
        self.optimizer_cls = self._init_optimizer_cls(optimizer_type, learning_rate)
        self.optimizer_mar = self._init_optimizer_mar(optimizer_type, learning_rate)


        self.epoch = 0
        self.max_acc = -1
        self.max_acc_confident = -1
        self.max_acc_uncertain = -1
        self.max_auroc = -1

    def _init_optimizer(self, optimizer_type, lr):
        if optimizer_type.upper() == 'ADAM':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type.upper() == 'SGD':
            return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _init_optimizer_cls(self, optimizer_type, lr):
        if optimizer_type.upper() == 'ADAM':
            return optim.Adam(list(self.model.hidden.parameters()) + list(self.model.hidden_pred.parameters()) + list(self.model.output_pred.parameters()), lr=lr)
        elif optimizer_type.upper() == 'SGD':
            return optim.SGD(list(self.model.hidden.parameters()) + list(self.model.hidden_pred.parameters()) + list(self.model.output_pred.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _init_optimizer_mar(self, optimizer_type, lr):
        if optimizer_type.upper() == 'ADAM':
            return optim.Adam(list(self.model.hidden_mar.parameters()) + list(self.model.output_mar.parameters()), lr=lr)
        elif optimizer_type.upper() == 'SGD':
            return optim.SGD(list(self.model.hidden_mar.parameters()) + list(self.model.output_mar.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def cls_loss(self, target, logits):
        loss_cls = self.criterion(logits, target)
        return loss_cls

    def mar_loss(self, target, num_classes, logits, mar):
        """
        residual-based MAR uncertainty loss.
        MAR targets = |onehot label - predicted prob|
        """
        one_hot_labels = F.one_hot(target, num_classes=num_classes).float()
        prob = F.softmax(logits, dim=1).detach()
        mar_target = torch.abs(one_hot_labels - prob)
        loss_mar = F.mse_loss(mar, mar_target)
        return loss_mar

    def joint_train_step(self, data, target, num_classes):
        """Single training step."""
        self.model.train()
        logits, mar = self.model(data)
        loss_cls = self.cls_loss(target, logits)
        loss_mar = self.mar_loss(target, num_classes, logits, mar)
        loss = loss_cls + loss_mar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), mar

    def cls_train_step(self, data, target):
        """Single training step."""
        self.model.train()
        logits, mar = self.model(data)
        loss_cls = self.cls_loss(target, logits)
        loss = loss_cls
        self.optimizer_cls.zero_grad()
        loss.backward()
        self.optimizer_cls.step()
        return loss.item(), mar

    def mar_train_step(self, data, target, num_classes):
        """Single training step."""
        self.model.train()
        logits, mar = self.model(data)
        loss_mar = self.mar_loss(target, num_classes, logits, mar)
        loss = loss_mar
        self.optimizer_mar.zero_grad()
        loss.backward()
        self.optimizer_mar.step()
        return loss.item(), mar

    def evaluate(self, test_loader, ood_loader, num_classes, threshold=None):
        """
        Evaluate classification accuracy and uncertainty quantification:
        - accuracy over all
        - accuracy over confident/uncertain splits
        - AUROC for ID vs. OOD uncertainty separation
        """
        self.model.eval()
        y_all, logits_all, mar_all = [], [], []

        with torch.no_grad():
            start_time = time.time()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits, mar = self.model(data)
                y_all.append(target)
                logits_all.append(logits)
                mar_all.append(mar)

            y_all = torch.cat(y_all)
            logits = F.softmax(torch.cat(logits_all), dim=1)
            mar = torch.cat(mar_all)

            # Uncertainty estimation
            expected_uncertainty = 2 * logits * (1 - logits)
            uncertainty = torch.sum(torch.abs(mar - expected_uncertainty), dim=1)
            confidences, predictions = torch.max(logits, dim=1)
            test_time=(time.time() - start_time) * 1000

            threshold = np.quantile(uncertainty.cpu().numpy(), 0.5)
            acc = (predictions == y_all).float().mean().item()

            confident_mask = (uncertainty < threshold)
            uncertain_mask = ~confident_mask

            acc_confident = (predictions[confident_mask] == y_all[confident_mask]).float().mean().item()
            acc_uncertain = (predictions[uncertain_mask] == y_all[uncertain_mask]).float().mean().item()

            # OOD pass
            logits_ood, mar_ood = [], []
            for data, _ in ood_loader:
                data = data.to(device)
                logits_o, mar_o = self.model(data)
                logits_ood.append(logits_o)
                mar_ood.append(mar_o)

            logits_ood = F.softmax(torch.cat(logits_ood), dim=1)
            mar_ood = torch.cat(mar_ood)
            expected_ood = 2 * logits_ood * (1 - logits_ood)
            uncertainty_ood = torch.sum(torch.abs(mar_ood - expected_ood), dim=1)

            # AUROC for uncertainty discrimination
            bin_labels = torch.cat([
                torch.zeros_like(uncertainty),
                torch.ones_like(uncertainty_ood)
            ])
            all_uncertainty = torch.cat([uncertainty, uncertainty_ood])
            auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), all_uncertainty.cpu().numpy())

        return acc, acc_confident, acc_uncertain, auroc, test_time / len(test_loader)

    def train(self, train_dataset, test_dataset, ood_dataset, num_classes,
              batch_size=128, num_epochs=40, verbose=True, freq=1, joint_training=0):
        """
        Train classifier with SPC-based uncertainty, and evaluate per epoch.
        """
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

        loss_curve, uncertainty_curve = [], []
        acc_curve, acc_conf_curve, acc_unc_curve = [], [], []
        train_times, test_times = [], []

        if joint_training:
            for self.epoch in range(1, num_epochs + 1):
                total_loss, total_unc = 0.0, 0.0
                start_time = time.time()

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss, mar = self.joint_train_step(data, target, num_classes)
                    total_loss += loss

                    if verbose:
                        with torch.no_grad():
                            logits, mar = self.model(data)
                            prob = F.softmax(logits, dim=1)
                            mar_target = 2 * prob * (1 - prob)
                            uncertainty = torch.sum(torch.abs(mar - mar_target), dim=1)
                            total_unc += uncertainty.mean().item()

                train_times.append((time.time() - start_time) * 1000)
                avg_loss = total_loss / len(train_loader)
                avg_uncertainty = total_unc / len(train_loader)

                loss_curve.append(avg_loss)
                uncertainty_curve.append(avg_uncertainty)

                if self.epoch % freq == 0:
                    acc, acc_conf, acc_unc, auroc, test_time = self.evaluate(test_loader, ood_loader, num_classes)
                    acc_curve.append(acc)
                    acc_conf_curve.append(acc_conf)
                    acc_unc_curve.append(acc_unc)
                    test_times.append(test_time)

                    if acc > self.max_acc:
                        self.max_acc = acc
                        self.max_acc_confident = acc_conf
                        self.max_acc_uncertain = acc_unc
                        self.max_auroc = auroc

        else:
            for self.epoch in range(1, num_epochs + 1):
                total_loss = 0.0

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss, mar = self.cls_train_step(data, target)
                    total_loss += loss

                avg_loss = total_loss / len(train_loader)
                loss_curve.append(avg_loss)

            for self.epoch in range(1, num_epochs + 1):
                total_unc = 0.0
                start_time = time.time()

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    loss, mar = self.mar_train_step(data, target, num_classes)

                    if verbose:
                        with torch.no_grad():
                            logits, mar = self.model(data)
                            prob = F.softmax(logits, dim=1)
                            mar_target = 2 * prob * (1 - prob)
                            uncertainty = torch.sum(torch.abs(mar - mar_target), dim=1)
                            total_unc += uncertainty.mean().item()

                train_times.append((time.time() - start_time) * 1000)
                avg_uncertainty = total_unc / len(train_loader)

                uncertainty_curve.append(avg_uncertainty)

                if self.epoch % freq == 0:
                    acc, acc_conf, acc_unc, auroc, test_time = self.evaluate(test_loader, ood_loader, num_classes)
                    acc_curve.append(acc)
                    acc_conf_curve.append(acc_conf)
                    acc_unc_curve.append(acc_unc)
                    test_times.append(test_time)

                    self.max_acc = acc
                    self.max_acc_confident = acc_conf
                    self.max_acc_uncertain = acc_unc
                    self.max_auroc = auroc

        if verbose:
            self._plot_training_curves(loss_curve, uncertainty_curve, acc_curve, acc_conf_curve, acc_unc_curve)

        return (
            self.model, self.max_acc, self.max_acc_confident,
            self.max_acc_uncertain, self.max_auroc,
            np.mean(train_times), np.mean(test_times)
        )

    def _plot_training_curves(self, loss, unc, acc, acc_cer, acc_unc):
        """Plot loss, uncertainty, and accuracy curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Train Loss')
        plt.title('Training Loss')
        plt.grid(True); plt.legend(); plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(unc, label='Train Uncertainty', color='orange')
        plt.title('Training Uncertainty')
        plt.grid(True); plt.legend(); plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(acc, label='Overall Accuracy')
        plt.plot(acc_cer, label='Confident Accuracy')
        plt.plot(acc_unc, label='Uncertain Accuracy')
        plt.title('Test Accuracy Curves')
        plt.grid(True); plt.legend(); plt.show()