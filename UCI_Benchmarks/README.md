# UCI Regression Benchmarks

This directory trains several uncertainty quantification (UQ) models on a collection of small UCI regression datasets. Each dataset is split into 90% train / 10% test, and results are averaged across multiple random seeds.

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

## Usage
Run the benchmark script from this directory:

```bash
python run_uci_dataset_tests.py [--num-trials N] [--num-epochs E] [--datasets d1 d2 ...] [--noise {tri,log}]
```

The benchmarks are lightweight and can run on **CPU-only devices**.  

### Arguments
- `--num-trials` *(default: 20)* — number of random train/test splits.
- `--num-epochs` *(default: 400)* — training epochs per trial.
- `--datasets` — subset of datasets to evaluate; default is all of them.
- `--noise` — optionally augment targets with tri-modal or log-normal noise.

The script reports RMSE, coverage, calibration error and other statistics for each dataset/model combination.

### Pretrained model weights
We also provide pretrained model weights to facilitate **reproducibility** of our experiments.  

Please download the weights from our [Hugging Face directory](https://huggingface.co/zzz0527/SPC-UQ/tree/main/SPC-UQ/UCI_Benchmarks) and place them in the `pretrained_model_weights/` directory. Once downloaded, you can directly run the evaluation without training.  
