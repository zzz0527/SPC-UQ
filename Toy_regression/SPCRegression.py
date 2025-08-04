import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def pinball_loss(q_pred, target, tau):
    """Standard quantile regression loss."""
    errors = target - q_pred
    loss = torch.max(tau * errors, (tau - 1) * errors)
    return torch.mean(loss)

def cali_loss(y_pred, y_true, q, scale=True):
    """
    Calibration loss for quantile regression.
    Penalizes over- or under-coverage relative to quantile level q.
    """
    diff = y_true - y_pred
    under_mask = (y_true <= y_pred)
    over_mask = ~under_mask

    coverage = torch.mean(under_mask.float())

    if coverage < q:
        loss = torch.mean(diff[over_mask])
    else:
        loss = torch.mean(-diff[under_mask])

    if scale:
        loss *= torch.abs(q - coverage)
    return loss

class SPCRegressionNet(nn.Module):
    """
    Neural network that predicts:
    - point estimate (v)
    - MAR (mean absolute residual)
    - MAR up/down (for epistemic decomposition)
    - QR up/down (for aleatoric decomposition)
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super(SPCRegressionNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        self.output_v = nn.Linear(hidden_dim, 1)
        self.output_uq = nn.Linear(hidden_dim, 5)


    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        v = self.output_v(x)
        x = self.hidden3(x)
        x = self.relu(x)
        output = self.output_uq(x)
        mar, mar_up, mar_down, q_up, q_down = torch.chunk(output, 5, dim=-1)

        q_up = F.softplus(q_up)
        q_down = F.softplus(q_down)
        return v, mar, mar_up, mar_down, q_up, q_down

class SPCregression:
    """
    Trainer and predictor for SPC UQ model.
    Supports joint or stagewise training strategies.
    """
    def __init__(self, learning_rate=5e-3):
        torch.manual_seed(42)
        self.model = SPCRegressionNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer1 = optim.Adam(list(self.model.hidden.parameters()) + list(self.model.hidden2.parameters()) + list(self.model.output_v.parameters()),
                                     lr=learning_rate,weight_decay=1e-4)
        self.optimizer2 = optim.Adam(list(self.model.hidden3.parameters()) + list(self.model.output_uq.parameters()),
                                     lr=learning_rate,weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.criterion2=nn.L1Loss()

    def mar_loss(self, y, predictions, mar, mar_up, mar_down, q_up, q_down):
        """Computes loss for MAR and QR heads."""
        residual = abs(y - predictions)
        diff = (y - predictions.detach())
        loss_mar = self.criterion(mar, residual)

        mask_up = (diff > 0)
        mask_down = (diff < 0)
        loss_mar_up = self.criterion(mar_up[mask_up], (y[mask_up] - predictions[mask_up]))
        loss_mar_down = self.criterion(mar_down[mask_down], (predictions[mask_down] - y[mask_down]))

        loss_q_up = pinball_loss(q_up[mask_up], (y[mask_up] - predictions[mask_up]), 0.95)
        loss_q_down = pinball_loss(q_down[mask_down], (predictions[mask_down] - y[mask_down]), 0.95)
        loss_cali_up = cali_loss(q_up[mask_up], (y[mask_up] - predictions[mask_up]), 0.95)
        loss_cali_down = cali_loss(q_down[mask_down], (predictions[mask_down] - y[mask_down]), 0.95)


        loss = loss_mar + loss_mar_up + loss_mar_down +(loss_cali_up + loss_cali_down)
               # + 0.2 * (loss_q_up + loss_q_down) \
               # + 0.8 * (loss_cali_up + loss_cali_down) \
        return loss

    def train(self, data, target, num_epochs=5000, strategy='stagewise'):
        """
        Train the model using either:
        - 'joint': full loss on all components
        - 'stagewise': first fit task head, then UQ heads
        """
        torch.manual_seed(42)
        self.model.train()

        if strategy== 'joint':
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                predictions, mar, mar_up, mar_down, q_up, q_down = self.model(data)
                loss = self.criterion(predictions, target)+self.mar_loss(target, predictions, mar, mar_up, mar_down, q_up, q_down)
                loss.backward()
                self.optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        if strategy== 'stagewise':
            for epoch in range(num_epochs):
                self.optimizer1.zero_grad()
                predictions, mar, mar_up, mar_down, q_up, q_down = self.model(data)
                loss = self.criterion(predictions, target)
                loss.backward()
                self.optimizer1.step()
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

            for epoch in range(num_epochs):
                self.optimizer2.zero_grad()
                predictions, mar, mar_up, mar_down, q_up, q_down = self.model(data)
                loss = self.mar_loss(target, predictions, mar, mar_up, mar_down, q_up, q_down)
                loss.backward()
                self.optimizer2.step()
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def predict(self, data, calibration=False):
        """Run prediction and return interval bounds and uncertainty estimate."""
        self.model.eval()
        with torch.no_grad():
            predictions, mar, mar_up, mar_down, q_up, q_down = self.model(data)

            v = predictions.numpy()
            mar = mar.detach().numpy()
            mar_up = mar_up.detach().numpy()
            mar_down = mar_down.detach().numpy()
            q_up = q_up.detach().numpy()
            q_down = q_down.detach().numpy()

            if calibration:
                # Calibration adjustment based on Self-consistency
                d_up = (mar * mar_down) / ((2 * mar_down - mar) * mar_up)
                d_down = (mar * mar_up) / ((2 * mar_up - mar) * mar_down)
                d_up = np.clip(d_up, 1, None)
                d_down = np.clip(d_down, 1, None)
                q_up *= d_up
                q_down *= d_down

            high_bound = v + q_up
            low_bound = v - q_down

            # Self-consistency Verification
            uncertainty = (abs(2 * mar_up * mar_down - mar * (mar_up + mar_down)))


        return v.squeeze(), high_bound.squeeze(), low_bound.squeeze(), uncertainty.squeeze()