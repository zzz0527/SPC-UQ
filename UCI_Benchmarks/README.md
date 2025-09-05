# UCI Regression Benchmarks

This directory trains several uncertainty quantification (UQ) models on a collection of small UCI regression datasets. Each dataset is split into 90% train / 10% test and results are averaged across multiple random seeds.

## Datasets
The script supports the following datasets located under `UCI_Benchmarks/data/uci`:

- `boston`
- `concrete`
- `energy`
- `kin8nm`
- `naval`
- `power`
- `protein`
- `wine`
- `yacht`

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- pandas
- h5py

## Usage
Run the benchmark script from this directory:

```bash
python3 run_uci_dataset_tests.py [--num-trials N] [--num-epochs E] [--datasets d1 d2 ...] [--noise {tri,log}]
```

### Arguments
- `--num-trials` *(default: 20)* — number of random train/test splits.
- `--num-epochs` *(default: 400)* — training epochs per trial.
- `--datasets` — subset of datasets to evaluate; default is all of them.
- `--noise` — optionally augment targets with `tri`-modal or `log`-normal noise.

The script reports RMSE, coverage, calibration error and other statistics for each dataset/model combination.
