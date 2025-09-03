import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def nll_loss(mean, log_var, target):
    """
    Negative log-likelihood loss for Gaussian output
    """
    var = torch.exp(log_var)
    loss = 0.5 * torch.log(2 * np.pi * var) + 0.5 * ((target - mean) ** 2) / var
    return loss.mean()

class NLLRegressionNN(nn.Module):
    """
    Neural network for regression with Gaussian likelihood output
    Outputs mean and log-variance
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)  # Outputs: mean and log_variance

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.hidden(x))
        x = self.fc2(x)
        mean = x[:, :1]
        log_var = x[:, 1:]
        return mean, log_var

class DeepEnsemble:
    """
    Deep Ensemble for probabilistic regression with Gaussian likelihood
    """
    def __init__(self, num_models=5, learning_rate=5e-3):
        torch.manual_seed(42)
        self.models = [NLLRegressionNN() for _ in range(num_models)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]

    def train(self, data, target, num_epochs=5000):
        """
        Train all models independently on the same data
        """
        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers), start=1):
            torch.manual_seed(idx + 42)  # Different seed for each model
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                mean, log_var = model(data)
                loss = nll_loss(mean, log_var, target)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 500 == 0:
                    print(f"Model {idx}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def predict(self, data):
        """
        Return ensemble mean, uncertainty, and prediction interval
        """
        means = []
        variances = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                mean, log_var = model(data)
                means.append(mean.numpy())
                variances.append(torch.exp(log_var).numpy())

        means = np.array(means)  # (num_models, batch, 1)
        variances = np.array(variances)

        # Mean and total predictive variance (mean of model variances)
        mean_ensemble = np.mean(means, axis=0)
        var_ensemble = np.mean(variances, axis=0)

        # Epistemic uncertainty: variance across model means
        epistemic_uncertainty = np.var(means, axis=0)

        std_ensemble = np.sqrt(var_ensemble)
        y_low = mean_ensemble - 2 * std_ensemble
        y_high = mean_ensemble + 2 * std_ensemble

        return (
            mean_ensemble.squeeze(),
            y_high.squeeze(),
            y_low.squeeze(),
            epistemic_uncertainty.squeeze()
        )
