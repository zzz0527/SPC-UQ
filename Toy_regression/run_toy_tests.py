import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from SPCRegression import SPCregression
from DeepEnsembleRegression import DeepEnsemble
from ConformalRegression import ConformalRegressor
from EDLRegression import EDLRegressor
from EDLQuantileRegression import EDLQuantileRegressor
from QROC import QROC
from scipy.stats import binned_statistic, norm


def ece_pi(y_true, pred_lower, pred_upper, num_bins=10):
    """Compute Expected Calibration Error (ECE) based on prediction interval width bins."""
    N = y_true.shape[0]
    in_interval = ((y_true >= pred_lower) & (y_true <= pred_upper)).astype(float)
    widths = pred_upper - pred_lower
    min_w, max_w = np.min(widths), np.max(widths)

    if min_w == max_w:
        return np.abs(in_interval.mean() - 1.0)

    bin_edges = np.linspace(min_w, max_w, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
        count_in_bin = np.sum(bin_mask)
        if count_in_bin == 0:
            continue
        avg_in_interval = in_interval[bin_mask].mean()
        nominal_coverage = 0.95
        calib_error = np.abs(avg_in_interval - nominal_coverage)
        weight = count_in_bin / N
        ece += weight * calib_error
    return ece


def binning(pred_lower, pred_upper, num_bins=10):
    """Group indices into bins based on interval width."""
    widths = pred_upper - pred_lower
    min_w, max_w = np.min(widths), np.max(widths)
    bin_edges = np.linspace(min_w, max_w, num_bins + 1)

    bins = []
    for i in range(num_bins):
        if i == num_bins - 1:
            bin_mask = (widths >= bin_edges[i]) & (widths <= bin_edges[i + 1])
        else:
            bin_mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
        bin_indices = np.where(bin_mask)[0]
        bins.append(bin_indices)
    return bins, bin_edges


def generate_multimodal_data(n_samples=1000):
    """Generate mixture-distribution noise samples."""
    x = np.random.randn(n_samples)
    mask = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3], size=n_samples)
    y = np.where(mask == 0, x + np.random.randn(n_samples) * 1,
                 np.where(mask == 1, x + 40 + np.random.randn(n_samples) * 1,
                          x - 10 + np.random.randn(n_samples) * 1))
    return y


def generate_train_data(n_samples=20, noise='log'):
    """Generate synthetic training data with nonlinear relationship and optional noise."""
    np.random.seed(57)
    x = np.linspace(-4, 4, n_samples)

    if noise == 'log':
        noise = np.random.lognormal(mean=1.5, sigma=1, size=n_samples)
    elif noise == 'tri':
        noise = generate_multimodal_data(n_samples)
    elif noise == 'norm':
        noise = np.random.normal(0, 8, size=n_samples)

    noise = noise - np.mean(noise)
    y = x ** 3 + noise
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return x, y


def generate_test_data(n_samples=100, noise='log'):
    """Generate synthetic test data with extended input range and optional noise."""
    np.random.seed(27)
    x = np.linspace(-6, 6, n_samples)

    if noise == 'log':
        noise = np.random.lognormal(mean=1.5, sigma=1, size=n_samples)
    elif noise == 'tri':
        noise = generate_multimodal_data(n_samples)
    elif noise == 'norm':
        noise = np.random.normal(0, 8, size=n_samples)

    noise = noise - np.mean(noise)
    y = x ** 3 + noise
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return x, y


# Generate training, calibration, and testing datasets
x_train, y_train = generate_train_data(n_samples=2000, noise='log')
x_calib, y_calib = generate_train_data(n_samples=500)
x_test, y_test = generate_test_data(n_samples=1000, noise='log')

# Convert data to torch tensors
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_calib_tensor = torch.from_numpy(x_calib)
y_calib_tensor = torch.from_numpy(y_calib)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

# Training parameters
num_epochs = 5000
num_models = 5
lr = 0.001

# Select UQ model to test
UQ = 'SPCregression'

if UQ == 'SPCregression':
    model = SPCregression(learning_rate=lr)
elif UQ == 'ConformalRegressor':
    model = ConformalRegressor(0.95, learning_rate=lr)
elif UQ == 'DeepEnsemble':
    model = DeepEnsemble(num_models=num_models, learning_rate=lr)
elif UQ == 'EDLRegressor':
    model = EDLRegressor(learning_rate=lr)
elif UQ == 'EDLQuantileRegressor':
    model = EDLQuantileRegressor(tau_low=0.025, tau_high=0.975, learning_rate=lr)
elif UQ == 'QROC':
    model = QROC(tau_low=0.025, tau_high=0.975, learning_rate=lr)

# Train model
model.train(x_train_tensor, y_train_tensor, num_epochs)

# Calibrate if applicable
if UQ == 'ConformalRegressor':
    model.calibrate(x_calib_tensor, y_calib_tensor)

# Predict on train and test sets
mean, upper_bound, lower_bound, uncertainty = model.predict(x_train_tensor)
y_train_np = y_train.flatten()

# Evaluate train metrics
mpi_width = np.mean(upper_bound - lower_bound)
picp = np.mean(((y_train_np >= lower_bound) & (y_train_np <= upper_bound)).astype(float))
rmse = np.sqrt(np.mean((y_train_np - mean) ** 2))
print(f"Train Mean Prediction Interval Width (MPIW): {mpi_width:.4f}")
print(f"Train Prediction Interval Coverage Probability (PICP): {picp:.4f}")
print(f"Train Root Mean Squared Error (RMSE): {rmse:.4f}")

# Predict on test set
mean, upper_bound, lower_bound, uncertainty = model.predict(x_test_tensor)
x_test_np = x_test.flatten()
y_test_np = y_test.flatten()

# In-distribution test subset for interval evaluation
y_low_id = lower_bound[170:830]
y_high_id = upper_bound[170:830]
y_meam_id = mean[170:830]
x_test_id = x_test_np[170:830]
y_test_id = y_test_np[170:830]

mpi_width_id = np.mean(y_high_id - y_low_id)
picp_id = np.mean(((y_test_id >= y_low_id) & (y_test_id <= y_high_id)).astype(float))
rmse_id = np.sqrt(np.mean((y_test_id - y_meam_id) ** 2))

print(f"ID Mean Prediction Interval Width (MPIW): {mpi_width_id:.4f}")
print(f"ID Prediction Interval Coverage Probability (PICP): {picp_id:.4f}")
print(f"ID Root Mean Squared Error (RMSE): {rmse_id:.4f}")

# Compute ECE
ece = ece_pi(y_test_id, y_low_id, y_high_id, num_bins=10)
print(f"Expected Calibration Error (ECE): {ece:.4f}")

threshold = uncertainty[170:830].mean()
print('threshold:', threshold)
cer_count = 0
unc_count = 0
cer_diff = []
unc_diff = []

for i in range(len(uncertainty)):
    unc = uncertainty[i]
    diff_sq = (y_test_np[i] - mean[i])**2
    if unc < threshold:
        cer_count += 1
        cer_diff.append(diff_sq)
    else:
        unc_count += 1
        unc_diff.append(diff_sq)

rmse_certain = np.sqrt(np.mean(cer_diff)) if len(cer_diff)>0 else 0
rmse_uncertain = np.sqrt(np.mean(unc_diff)) if len(unc_diff)>0 else 0
rmse_all = np.sqrt(np.mean(cer_diff + unc_diff))

print('Certain sample:', cer_count,
      'RMSE_certain:', round(rmse_certain,4),
      'Uncertain sample:', unc_count,
      'RMSE_uncertain:', round(rmse_uncertain,4),
      'RMSE_all:', round(rmse_all,4))

print(round(rmse_id,4),
      round(picp_id,4),
      round(mpi_width_id,4),
      round(rmse_certain,4),
      round(rmse_uncertain,4),
      round(rmse_all,4),
      cer_count, unc_count)


def plot_binned_intervals(x_test, y_test, mean, lower_bound, upper_bound, bins, num_bins=1):
    """Plot prediction intervals with color-coded bins based on interval width."""
    x = x_test.squeeze()
    y = y_test.squeeze()

    def quantile_stat(q):
        def func(y_in_bin):
            return np.percentile(y_in_bin, q)
        return func

    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=num_bins)
    q5, _, _ = binned_statistic(x, y, statistic=quantile_stat(5), bins=bin_edges)
    q95, _, _ = binned_statistic(x, y, statistic=quantile_stat(95), bins=bin_edges)

    gt = x ** 3
    y_up = (y - gt)[y > gt]
    y_down = (gt - y)[y < gt]
    gt_up = np.quantile(y_up, 0.95)
    gt_down = np.quantile(y_down, 0.95)

    plt.figure(figsize=(7, 5))
    cmap = cm.get_cmap('viridis', len(bins))
    interval_width = 0.1
    for i, bin_indices in enumerate(bins):
        color = cmap(i)
        for j, idx in enumerate(bin_indices):
            x_val = x_test[idx]
            plt.fill_between(
                [x_val - interval_width / 2, x_val + interval_width / 2],
                [lower_bound[idx], lower_bound[idx]],
                [upper_bound[idx], upper_bound[idx]],
                color=color,
                alpha=0.2,
                label=f'Bin {i + 1}' if j == 0 else None
            )

    plt.plot(x, gt, color='blue', linestyle='--', label='Mean E[y|x]', linewidth=2)
    plt.plot(x, gt - gt_down, color='darkorange', linestyle='--', label='Lower bound GT', linewidth=2)
    plt.plot(x, gt + gt_up, color='purple', linestyle='--', label='Upper bound GT', linewidth=2)
    plt.scatter(x_test, y_test, color='green', s=5, label='Test Data')
    plt.plot(x_test, mean, color='red', label='Point Prediction')
    plt.title("Equal-width Binned PIs")
    plt.legend()
    plt.ylim(-100, 100)
    plt.tight_layout()
    plt.show()

    # Second plot with shaded split intervals
    plt.figure(figsize=(7, 4))
    plt.plot(x, gt, color='blue', linestyle='--', label='Mean E[y|x]', linewidth=2)
    plt.plot(x, gt - gt_down, color='darkorange', linestyle='--', label='Lower bound GT', linewidth=2)
    plt.plot(x, gt + gt_up, color='purple', linestyle='--', label='Upper bound GT', linewidth=2)
    plt.scatter(x_test, y_test, color='green', s=5, label='Test Data')
    plt.plot(x_test, mean, color='red', label='Point Prediction')
    plt.fill_between(x_test.flatten(), lower_bound, mean, color='orange', alpha=0.3, label='Lower Interval')
    plt.fill_between(x_test.flatten(), mean, upper_bound, color='purple', alpha=0.4, label='Upper Interval')
    plt.legend(fontsize=8)
    plt.ylim(-80, 100)
    plt.title('Split-point Prediction Intervals')
    plt.tight_layout()
    plt.show()

def plot_intervals(x_test, y_test, mean, lower_bound, upper_bound, uncertainty):
    """Plot prediction intervals and epistemic uncertainty with ground truth reference."""
    x = x_test.squeeze()
    y = y_test.squeeze()

    gt = x ** 3
    y_up = (y - gt)[y > gt]
    y_down = (gt - y)[y < gt]
    gt_up = np.quantile(y_up, 0.95)
    gt_down = np.quantile(y_down, 0.95)

    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 3])

    ax1 = plt.subplot(gs[0])
    ax1.axvspan(x.min(), -4, facecolor='lightgray', alpha=0.4)
    ax1.axvspan(4, x.max(), facecolor='lightgray', alpha=0.4)
    ax1.plot(x, gt, color='blue', linestyle='--', label='Mean E[y|x]', linewidth=2)
    ax1.plot(x, gt - gt_down, color='darkorange', linestyle='--', label='Lower bound GT', linewidth=2)
    ax1.plot(x, gt + gt_up, color='purple', linestyle='--', label='Upper bound GT', linewidth=2)
    ax1.scatter(x_test, y_test, color='green', s=5, label='Test Data')
    ax1.plot(x_test, mean, color='red', label='Point Prediction')
    ax1.fill_between(x_test.flatten(), lower_bound, mean, color='orange', alpha=0.3, label='Lower Interval')
    ax1.fill_between(x_test.flatten(), mean, upper_bound, color='purple', alpha=0.4, label='Upper Interval')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-150, 150)

    ax2 = plt.subplot(gs[1])
    ax2.axvspan(x.min(), -4, facecolor='lightgray', alpha=0.4, label='OOD Region')
    ax2.axvspan(4, x.max(), facecolor='lightgray', alpha=0.4)
    ax2.plot(x_test, uncertainty, label='Epistemic uncertainty', color='dodgerblue')
    ax2.fill_between(x_test.squeeze(), uncertainty.squeeze(), alpha=0.3, color='dodgerblue')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# Call visualizations after metric evaluations
bins, bin_edges = binning(y_low_id, y_high_id, num_bins=5)
plot_binned_intervals(x_test_id, y_test_id, y_meam_id, y_low_id, y_high_id, bins)
plot_intervals(x_test, y_test, mean, lower_bound, upper_bound, uncertainty)
