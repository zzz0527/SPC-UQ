# Monocular Depth Estimation

This directory demonstrates monocular depth estimation with uncertainty quantification using various models.

## Dataset
- **In-distribution** training and test sets are stored as HDF5 files `depth_train.h5` and `depth_test.h5` under `data/`.
- **Out-of-distribution** evaluation uses `apolloscape_test.h5` placed in the same `data/` directory.

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

## Evaluation
After training or using provided checkpoints in `save/`, evaluate models with:

```bash
python3 test_depth.py [--load-pkl]
```

`--load-pkl` loads cached predictions from pickle files if available instead of recomputing them.

The script reports RMSE, prediction interval coverage (PICP), interval scores, AUROC between in- and out-of-distribution data, and plots accuracy and uncertainty under adversarial perturbations.
