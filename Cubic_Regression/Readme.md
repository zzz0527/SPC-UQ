# Toy Example: SPC-UQ on Cubic Regression

This directory provides an implementation of several uncertainty quantification (UQ) methods on a synthetic cubic regression problem.

The task is defined by

    y = x³ + ε(x) - E[ε(x)]

- Training set: 2,000 samples
- Test set: 1,000 samples
- In-distribution: x ∈ [-4, 4]
- Out-of-distribution: x ∈ [-6, -4) ∪ (4, 6]

## Running `run_cubic_tests.py`

`run_cubic_tests.py` trains and evaluates different UQ models, and visualizes results.

### Basic usage

```bash
python run_cubic_tests.py --num-epochs 5000 --data-noise log --UQ-model SPCregression
```

### Arguments

- `--num-epochs`: Number of training epochs (default: 5000).
- `--data-noise {norm, tri, log}`: Type of noise added to the data. `norm` is Gaussian, `tri` is a mixture distribution, and `log` is log-normal (default).
- `--UQ-model {SPCregression, DeepEnsemble, EDLRegressor, EDLQuantileRegressor, QROC, ConformalRegressor}`: UQ model to run.

After execution, the script prints RMSE, PICP and related metrics, and produces plots of prediction intervals and uncertainty.
