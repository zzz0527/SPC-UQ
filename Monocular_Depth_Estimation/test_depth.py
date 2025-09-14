import torch
import torch.nn.functional as F
import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum
from pathlib import Path
import time

from tqdm import tqdm
import models
from trainers.evidential import edl_loss
from trainers.utils import picp, picp_up, picp_down, interval_score
from sklearn import metrics

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(Enum):
    GROUND_TRUTH = "GroundTruth"
    DROPOUT = "Dropout"
    ENSEMBLE = "Ensemble"
    EVIDENTIAL = "Evidential"
    QREVIDENTIAL = "QREvidential"
    QROC = "QROC"
    SPC = "SPC"

# Directory settings
save_dir = "pretrained_model_weights"

# Pretrained model paths for each method
trained_models = {
    Model.DROPOUT:     ["dropout.pth", "dropout2.pth", "dropout3.pth", "dropout4.pth", "dropout5.pth"],
    Model.ENSEMBLE:    ["ensemble.pth", "ensemble2.pth", "ensemble3.pth", "ensemble4.pth", "ensemble5.pth"],
    Model.EVIDENTIAL:  ["edl.pth", "edl2.pth", "edl3.pth", "edl4.pth", "edl5.pth"],
    Model.QREVIDENTIAL:["qredl.pth", "qredl2.pth", "qredl3.pth", "qredl4.pth", "qredl5.pth"],
    Model.QROC:        ["qroc.pth", "qroc2.pth", "qroc3.pth", "qroc4.pth", "qroc5.pth"],
    Model.SPC:         ["spc.pth", "spc2.pth", "spc3.pth", "spc4.pth", "spc5.pth"],
}

def compute_predictions(batch_size=50, n_adv=9):
    (x_in, y_in), (x_ood, y_ood) = load_data()
    datasets = [(x_in, y_in, False), (x_ood, y_ood, True)]
    adv_epsilons = np.linspace(0, 0.2, n_adv)

    for method, model_paths in trained_models.items():
        num_trials = len(model_paths)
        metrics_dict = {
            "rmse": np.zeros(num_trials),
            "picp": np.zeros(num_trials),
            "picp_up": np.zeros(num_trials),
            "picp_down": np.zeros(num_trials),
            "winklar_score": np.zeros(num_trials),
            "mpiw": np.zeros(num_trials),
            "speed": np.zeros(num_trials),
            "rmse_cer": np.zeros(num_trials),
            "picp_cer": np.zeros(num_trials),
            "is_cer": np.zeros(num_trials),
            "auroc": np.zeros(num_trials)
        }
        adv_unc_all = np.zeros((num_trials, n_adv))
        adv_rmse_all = np.zeros((num_trials, n_adv))

        for n, model_path in enumerate(model_paths):
            full_path = os.path.join(save_dir, model_path)
            model = models.load_depth_model(method.value, full_path).to(device)
            model.eval()

            # Preallocate arrays for adversarial evaluation
            id_rmse = np.zeros((n_adv, len(x_in)))
            id_unc = np.zeros((n_adv, len(x_in)))
            id_picp = np.zeros((n_adv, len(x_in)))
            id_picp_up = np.zeros((n_adv, len(x_in)))
            id_picp_down = np.zeros((n_adv, len(x_in)))
            id_is = np.zeros((n_adv, len(x_in)))
            id_width = np.zeros((n_adv, len(x_in)))
            speed = np.zeros((n_adv, len(x_in)))

            ood_rmse = np.zeros(len(x_ood))
            ood_unc = np.zeros(len(x_ood))
            ood_picp = np.zeros(len(x_ood))
            ood_is = np.zeros(len(x_ood))

            for x, y, ood in datasets:
                for start_i in tqdm(range(0, len(x), batch_size)):
                    inds = np.arange(start_i, min(start_i + batch_size, len(x)))
                    x_batch = torch.tensor(np.transpose(x[inds] / 255.0, (0, 3, 1, 2)), dtype=torch.float32, device=device)
                    y_batch = torch.tensor(np.transpose(y[inds] / 255.0, (0, 3, 1, 2)), dtype=torch.float32, device=device)

                    if ood:
                        pred_batch, unc_batch, rmse, picp, picp_up, picp_down, interval_score_val, mpiw, batch_time = get_prediction_summary(
                            method, model, x_batch, y_batch
                        )
                        pred_batch = pred_batch.mean(axis=(1, 2, 3))
                        unc_batch = unc_batch.mean(axis=(1, 2, 3))
                        rmse = rmse.mean(axis=(1, 2, 3))
                        ood_rmse[inds] = rmse
                        ood_unc[inds] = unc_batch
                        ood_picp[inds] = picp
                        ood_is[inds] = interval_score_val
                    else:
                        if method == Model.ENSEMBLE:
                            mask_batch = create_adversarial_pattern(method, model.models[0], x_batch, y_batch)
                        else:
                            mask_batch = create_adversarial_pattern(method, model, x_batch, y_batch)

                        for adv_idx, eps in enumerate(adv_epsilons):
                            x_adv = x_batch + (eps * mask_batch)
                            x_adv = torch.clamp(x_adv, 0, 1)
                            pred_batch, unc_batch, rmse, picp, picp_up, picp_down, interval_score_val, mpiw, batch_time = get_prediction_summary(
                                method, model, x_adv, y_batch
                            )
                            pred_batch = pred_batch.mean(axis=(1, 2, 3))
                            unc_batch = unc_batch.mean(axis=(1, 2, 3))
                            rmse = rmse.mean(axis=(1, 2, 3))

                            id_rmse[adv_idx, inds] = rmse
                            id_unc[adv_idx, inds] = unc_batch
                            id_picp[adv_idx, inds] = picp
                            id_picp_up[adv_idx, inds] = picp_up
                            id_picp_down[adv_idx, inds] = picp_down
                            id_is[adv_idx, inds] = interval_score_val
                            id_width[adv_idx, inds] = mpiw
                            speed[adv_idx, inds] = batch_time

            # AUROC calculation for ID/OOD
            id_unc0 = id_unc[0]
            bin_labels = np.zeros(id_unc0.shape[0])
            bin_labels = np.concatenate((bin_labels, np.ones(ood_unc.shape[0])))
            all_unc = np.concatenate((id_unc0, ood_unc))
            auroc = metrics.roc_auc_score(bin_labels, all_unc)

            # Calibration split
            threshold = np.quantile(id_unc0, 0.5)
            cer_indices = (id_unc0 <= threshold)
            unc_indices = ~cer_indices
            id_rmse_cer = id_rmse[0][cer_indices]
            id_rmse_uncer = id_rmse[0][unc_indices]
            id_picp_cer = id_picp[0][cer_indices]
            id_picp_uncer = id_picp[0][unc_indices]
            id_is_cer = id_is[0][cer_indices]
            id_is_uncer = id_is[0][unc_indices]

            # Average metrics over all samples for each run
            metrics_dict["rmse"][n] = id_rmse[0].mean()
            metrics_dict["picp"][n] = id_picp[0].mean()
            metrics_dict["picp_up"][n] = id_picp_up[0].mean()
            metrics_dict["picp_down"][n] = id_picp_down[0].mean()
            metrics_dict["winklar_score"][n] = id_is[0].mean()
            metrics_dict["mpiw"][n] = id_width[0].mean()
            metrics_dict["speed"][n] = speed[0].mean()
            metrics_dict["rmse_cer"][n] = id_rmse_cer.mean()
            metrics_dict["picp_cer"][n] = id_picp_cer.mean()
            metrics_dict["is_cer"][n] = id_is_cer.mean()
            metrics_dict["auroc"][n] = auroc

            adv_rmse_all[n] = id_rmse.mean(axis=1)
            adv_unc_all[n] = id_unc.mean(axis=1)

        # Aggregate and report
        print(f"\n===== {method.value} Results =====")
        for key, arr in metrics_dict.items():
            mean = arr.mean()
            std = arr.std() / np.sqrt(num_trials)
            print(f"{key.upper()}: {mean:.2f} ± {std:.2f}")

        # Plot RMSE under adversarial attack
        mean_rmse = np.mean(adv_rmse_all, axis=0)
        std_rmse = np.std(adv_rmse_all, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(mean_rmse, label='Accuracy (RMSE)', color='red')
        plt.fill_between(range(len(mean_rmse)),
                         mean_rmse - std_rmse,
                         mean_rmse + std_rmse,
                         color='red', alpha=0.3, label="±1 Std Dev")
        plt.xlabel('Adversarial epsilon')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()

        # Plot normalized uncertainty under adversarial attack
        adv_unc_norm = (adv_unc_all - adv_unc_all.min(axis=1, keepdims=True)) / \
                       (adv_unc_all.max(axis=1, keepdims=True) - adv_unc_all.min(axis=1, keepdims=True) + 1e-8)
        mean_unc = np.mean(adv_unc_norm, axis=0)
        std_unc = np.std(adv_unc_norm, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(mean_unc, label='Normalized Uncertainty', color='orange')
        plt.fill_between(range(len(mean_unc)),
                         mean_unc - std_unc,
                         mean_unc + std_unc,
                         color='orange', alpha=0.3, label="±1 Std Dev")
        plt.xlabel('Adversarial epsilon')
        plt.ylabel('Normalized Uncertainty')
        plt.legend()
        plt.show()

def get_prediction_summary(method, model, x_batch, y_batch, vis=False):
    start_time = time.time()
    pred, upper, lower, epistemic = predict(method, model, x_batch)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    rmse = torch.sqrt(F.mse_loss(pred, y_batch, reduction='none')).detach().cpu().numpy()
    picp_val = picp(y_batch, lower, upper).detach().cpu().numpy()
    picp_up_val = picp_up(y_batch, pred, lower, upper)
    picp_down_val = picp_down(y_batch, pred, lower, upper)
    interval_score_val, mpiw = interval_score(y_batch, lower, upper, alpha=0.05)
    interval_score_val = interval_score_val.detach().cpu().numpy()
    mpiw = mpiw.detach().cpu().numpy()

    if vis:
        # Visualization block
        fig, axs = plt.subplots(5, 1, figsize=(6, 20))
        img = x_batch[0].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        axs[0].imshow(img)
        axs[0].axis("off")
        axs[0].set_title("Input Image")
        axs[1].imshow(y_batch[0, 0].detach().cpu().numpy(), cmap='magma')
        axs[1].axis("off")
        axs[1].set_title("Depth Map GT")
        axs[2].imshow(pred[0, 0].detach().cpu().numpy(), cmap='magma')
        axs[2].axis("off")
        axs[2].set_title("Depth Map Pred")
        axs[3].imshow(abs(y_batch[0, 0] - pred[0, 0]).detach().cpu().numpy(), cmap='magma')
        axs[3].axis("off")
        axs[3].set_title("Error Map")
        axs[4].imshow(epistemic[0, 0].detach().cpu().numpy(), cmap='magma')
        axs[4].axis("off")
        axs[4].set_title("Uncertainty Map")
        plt.tight_layout()
        plt.show()

    pred_np = torch.clamp(pred, 0, 1).detach().cpu().numpy()
    unc_np = epistemic.detach().cpu().numpy()
    return pred_np, unc_np, rmse, picp_val, picp_up_val, picp_down_val, interval_score_val, mpiw, elapsed_ms

def load_data():
    def load_depth():
        train = h5py.File("datasets/depth_train.h5", "r")
        test = h5py.File("datasets/depth_test.h5", "r")
        return (train["image"], train["depth"]), (test["image"], test["depth"])

    def load_apollo():
        test = h5py.File("datasets/apolloscape_test.h5", "r")
        return (None, None), (test["image"], test["depth"])

    _, (x_test, y_test) = load_depth()
    _, (x_ood_test, y_ood_test) = load_apollo()
    print("Loaded data shapes:", x_test.shape, x_ood_test.shape)
    return (x_test, y_test), (x_ood_test, y_ood_test)

def predict(method, model, x, n_samples=5):
    with torch.no_grad():
        if method == Model.DROPOUT:
            model.train()
            preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)
            mu, var = torch.mean(preds, dim=0), torch.var(preds, dim=0)
            mean_mu, mean_sigma = torch.chunk(mu, 2, dim=1)
            mean_sigma = torch.sqrt(mean_sigma)
            upper = mean_mu + mean_sigma * 2
            lower = mean_mu - mean_sigma * 2
            epistemic = var
            return mean_mu, upper, lower, epistemic
        elif method == Model.EVIDENTIAL:
            outputs = model(x)
            mu, v, alpha, beta = torch.chunk(outputs, 4, dim=1)
            aleatoric = beta / (alpha - 1)
            epistemic = torch.sqrt(beta / (v * (alpha - 1)))
            # dof = 2.0 * alpha
            # scale = torch.sqrt((1.0 + v) * beta / (alpha * v))
            # alpha = 1.0 - 0.95
            # lower_q = alpha / 2.0
            # upper_q = 1.0 - alpha / 2.0
            # dof_np = dof.detach().cpu().numpy()
            # t_lower = student_t.ppf(lower_q, df=dof_np)
            # t_upper = student_t.ppf(upper_q, df=dof_np)
            # lower_bound = mu + torch.from_numpy(t_lower).to(mu.device) * scale
            # upper_bound = mu + torch.from_numpy(t_upper).to(mu.device) * scale
            lower_bound = mu - aleatoric
            upper_bound = mu + aleatoric
            return mu, upper_bound, lower_bound, epistemic
        elif method == Model.QREVIDENTIAL:
            outputs = model(x)
            mu, v, alpha, beta = torch.chunk(outputs, 4, dim=1)
            mu_low, mu_mid, mu_high = torch.unbind(mu, dim=1)
            mu_low = mu_low.unsqueeze(1)
            mu_mid = mu_mid.unsqueeze(1)
            mu_high = mu_high.unsqueeze(1)
            v_low, v_mid, v_high = torch.unbind(v, dim=1)
            v_low = v_low.unsqueeze(1)
            v_mid = v_mid.unsqueeze(1)
            v_high = v_high.unsqueeze(1)
            alpha_low, alpha_mid, alpha_high = torch.unbind(alpha, dim=1)
            alpha_low = alpha_low.unsqueeze(1)
            alpha_mid = alpha_mid.unsqueeze(1)
            alpha_high = alpha_high.unsqueeze(1)
            beta_low, beta_mid, beta_high = torch.unbind(beta, dim=1)
            beta_low = beta_low.unsqueeze(1)
            beta_mid = beta_mid.unsqueeze(1)
            beta_high = beta_high.unsqueeze(1)
            epistemic = (
                torch.sqrt(beta_low / (v_low * (alpha_low - 1))) +
                torch.sqrt(beta_mid / (v_mid * (alpha_mid - 1))) +
                torch.sqrt(beta_high / (v_high * (alpha_high - 1)))
            )
            lower = mu_low
            upper = mu_high
            return mu_mid, upper, lower, epistemic
        elif method == Model.QROC:
            q_low, q_mid, q_high, certificates = model(x)
            certificates = torch.sigmoid(certificates).pow(2).mean(1, keepdim=True).detach()
            epistemic = -certificates
            lower = q_low
            upper = q_high
            return q_mid, upper, lower, epistemic
        elif method == Model.ENSEMBLE:
            preds = torch.stack([model_i(x) for model_i in model.models], dim=0)
            mu, var = torch.mean(preds, dim=0), torch.var(preds, dim=0)
            mean_mu, mean_sigma = torch.chunk(mu, 2, dim=1)
            mean_sigma = torch.sqrt(mean_sigma)
            upper = mean_mu + mean_sigma * 2
            lower = mean_mu - mean_sigma * 2
            epistemic = var
            return mean_mu, upper, lower, epistemic
        elif method == Model.SPC:
            pred, mar, mar_up, mar_down, q_up, q_down = model(x)
            epistemic = torch.sqrt(abs(2 * mar_up * mar_down - mar * (mar_up + mar_down)))
            # epistemic = abs(2 * mar_up * mar_down - mar * (mar_up + mar_down))
            calibration=False
            if calibration:
                d_up = (mar * mar_down) / ((2 * mar_down - mar) * mar_up)
                d_down = (mar * mar_up) / ((2 * mar_up - mar) * mar_down)
                d_up = torch.clamp(d_up, min=1)
                d_down = torch.clamp(d_down, min=1)
                q_up *= d_up
                q_down *= d_down
            upper_bound = pred+q_up
            lower_bound = pred-q_down
            return pred, upper_bound, lower_bound, epistemic

        else:
            raise ValueError("Unknown model")

def create_adversarial_pattern(method, model, x, y):
    x.requires_grad = True
    model.zero_grad()
    if method in [Model.DROPOUT, Model.ENSEMBLE]:
        pred = model(x)
        mu, sigma = torch.chunk(pred, 2, dim=1)
        loss = F.mse_loss(mu, y)
    elif method == Model.EVIDENTIAL:
        pred = model(x)
        mu, v, alpha, beta = torch.chunk(pred, 4, dim=1)
        loss = F.mse_loss(mu, y)
        # loss = edl_loss(y, mu, v, alpha, beta)
    elif method == Model.QREVIDENTIAL:
        pred = model(x)
        mu, v, alpha, beta = torch.chunk(pred, 4, dim=1)
        mu = mu[:, 0:1, :, :]
        loss = F.mse_loss(mu, y)
        # loss = edl_loss(y, mu, v, alpha, beta)
    elif method == Model.QROC:
        q_low, q_mid, q_high, certificates = model(x)
        loss = F.mse_loss(q_mid, y)
    elif method == Model.SPC:
        pred, _, _, _, _, _ = model(x)
        loss = F.mse_loss(pred, y)
    else:
        raise ValueError("Unknown model in adversarial pattern creation.")

    loss.backward()
    signed_grad = x.grad.sign()
    return signed_grad

if __name__ == "__main__":
    compute_predictions()
