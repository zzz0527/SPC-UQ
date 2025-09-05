# Image Classification

This directory implements deterministic uncertainty methods for image classification and out-of-distribution (OoD) detection on datasets such as CIFAR-10/100, SVHN, TinyImageNet, and ImageNet.

## Dependencies

Python 3 with

- PyTorch
- Torchvision
- NumPy
- SciPy
- Matplotlib
- seaborn
- scikit-learn
- tensorboard
- tqdm

## Training

Use `train.py` to train a single model.

```bash
python train.py --seed 1 --dataset cifar10 --model wide_resnet -sn -mod --coeff 3.0
```

Key arguments:

- `--dataset {cifar10,cifar100,svhn,imagenet,tinyimagenet}` – dataset for training.
- `--dataset-root PATH` – root directory for datasets.
- `--model {lenet,resnet18,resnet50,wide_resnet,vgg16,...}` – network architecture.
- `-sn` / `--coeff` – enable spectral normalization and set its coefficient.
- `-mod` – use architectural modifications (leaky ReLU + average pooling in skip connections).
- `-b` – batch size.
- `-e` – number of training epochs.
- `--lr`, `--mom`, `--opt` – optimizer settings.
- `--save-path PATH` – where to save checkpoints.

## Evaluation

`evaluate.py` loads saved checkpoints and reports classification accuracy and OoD metrics.

```bash
python evaluate.py --seed 1 --dataset cifar10 --ood_dataset svhn --model wide_resnet --model-type gmm --load-path path/to/models/
```

Important options:

- `--ood_dataset` – dataset used as out-of-distribution data.
- `--load-path PATH` – directory containing saved models.
- `--model-type {softmax,ensemble,gmm,spc,edl,oc,joint}` – evaluation method.
- `--runs` – number of runs to aggregate.
- `--ensemble` – number of models per run for ensembles.
- `--val_size` – fraction of the training data held out for validation.

Additional scripts include `train_ensemble.py` for training ensembles and `evaluate_laplace.py` for Laplace approximation evaluation.
