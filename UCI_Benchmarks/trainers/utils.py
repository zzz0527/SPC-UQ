import numpy as np

def interval_score(y_true, pred, pred_lower, pred_upper, alpha=0.05):
    width = pred_upper - pred_lower

    # Penalties
    below = y_true < pred_lower
    above = y_true > pred_upper

    penalty = (
        (2 / alpha) * (pred_lower - y_true) * below +
        (2 / alpha) * (y_true - pred_upper) * above
    )

    score = width + penalty
    return score.mean()

def ece_pi(y_true, pred_lower, pred_upper, num_bins=10):
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
