# Monocular Depth Estimation

This directory demonstrates monocular depth estimation with uncertainty quantification using various models.

## Datasets
- **In-distribution** training and test sets are stored as HDF5 files `depth_train.h5` and `depth_test.h5` under `datasets/`.
- **Out-of-distribution** evaluation uses `apolloscape_test.h5` placed in the same `datasets/` directory.

Please download the datasets from our [Hugging Face directory](https://huggingface.co/zzz0527/SPC-UQ/tree/main/SPC-UQ/Monocular_Depth_Estimation).

## Training
Train a depth model by selecting a method and other hyperparameters:

```bash
python3 train_depth.py --model spc --batch-size 32 --iters 60000 --learning-rate 1e-4
```

### Arguments
- `--model` *(default: spc)* — one of `evidential`, `qrevidential`, `dropout`, `ensemble`, `qroc`, `spc`.
- `--batch-size` *(default: 32)* — training batch size.
- `--iters` *(default: 60000)* — number of optimization iterations.
- `--learning-rate` *(default: 1e-4)* — optimizer step size.

### Pretrained model weights
We provide several pretrained model weights to facilitate **reproducibility** of our experiments.

Please download the weights from our [Hugging Face directory](https://huggingface.co/zzz0527/SPC-UQ/tree/main/SPC-UQ/Monocular_Depth_Estimation) and place them in the `pretrained_model_weights/` directory. Once downloaded, you can directly run the provided evaluation scripts without additional training.  

## Evaluation
After training or using the provided checkpoints in `pretrained_model_weights/`, evaluate models with:

```bash
python3 test_depth.py
```

The script runs all trained models and reports RMSE, prediction interval coverage (PICP), interval scores, AUROC between in- and out-of-distribution data, and plots accuracy and uncertainty under adversarial perturbations.
