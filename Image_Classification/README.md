# Deep Deterministic Uncertainty

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2102.11582)
[![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/omegafragger/DDU/blob/main/LICENSE)

This repository contains the code for [*Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty*](https://arxiv.org/abs/2102.11582).

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{mukhoti2021deterministic,
  title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
  author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
  journal={arXiv preprint arXiv:2102.11582},
  year={2021}
}
```

## Dependencies

The code is based on PyTorch and requires a few further dependencies, listed in [environment.yml](environment.yml). It should work with newer versions as well.


## OoD Detection

### Datasets

For OoD detection, you can train on [*CIFAR-10/100*](https://www.cs.toronto.edu/~kriz/cifar.html). You can also train on [*Dirty-MNIST*](https://blackhc.github.io/ddu_dirty_mnist/) by downloading *Ambiguous-MNIST* (```amnist_labels.pt``` and ```amnist_samples.pt```) from [here](https://github.com/BlackHC/ddu_dirty_mnist/releases/tag/data-v0.6.0) and using the following training instructions.

### Training

In order to train a model for the OoD detection task, use the [train.py](train.py) script. Following are the main parameters for training:
```
--seed: seed for initialization
--dataset: dataset used for training (cifar10/cifar100/dirty_mnist)
--dataset-root: /path/to/amnist_labels.pt and amnist_samples.pt/ (if training on dirty-mnist)
--model: model to train (wide_resnet/vgg16/resnet18/resnet50/lenet)
-sn: whether to use spectral normalization (available for wide_resnet, vgg16 and resnets)
--coeff: Coefficient for spectral normalization
-mod: whether to use architectural modifications (leaky ReLU + average pooling in skip connections)
--save-path: path/for/saving/model/
```

As an example, in order to train a Wide-ResNet-28-10 with spectral normalization and architectural modifications on CIFAR-10, use the following:
```
python train.py \
       --seed 1 \
       --dataset cifar10 \
       --model wide_resnet \
       -sn -mod \
       --coeff 3.0 
```
Similarly, to train a ResNet-18 with spectral normalization on Dirty-MNIST, use:
```
python train.py \
       --seed 1 \
       --dataset dirty-mnist \
       --dataset-root /home/user/amnist/ \
       --model resnet18 \
       -sn \
       --coeff 3.0
```

### Evaluation

To evaluate trained models, use [evaluate.py](evaluate.py). This script can evaluate and aggregate results over multiple experimental runs. For example, if the pretrained models are stored in a directory path ```/home/user/models```, store them using the following directory structure:
```
models
├── Run1
│   └── wide_resnet_1_350.model
├── Run2
│   └── wide_resnet_2_350.model
├── Run3
│   └── wide_resnet_3_350.model
├── Run4
│   └── wide_resnet_4_350.model
└── Run5
    └── wide_resnet_5_350.model
```
For an ensemble of models, store the models using the following directory structure:
```
model_ensemble
├── Run1
│   ├── wide_resnet_1_350.model
│   ├── wide_resnet_2_350.model
│   ├── wide_resnet_3_350.model
│   ├── wide_resnet_4_350.model
│   └── wide_resnet_5_350.model
├── Run2
│   ├── wide_resnet_10_350.model
│   ├── wide_resnet_6_350.model
│   ├── wide_resnet_7_350.model
│   ├── wide_resnet_8_350.model
│   └── wide_resnet_9_350.model
├── Run3
│   ├── wide_resnet_11_350.model
│   ├── wide_resnet_12_350.model
│   ├── wide_resnet_13_350.model
│   ├── wide_resnet_14_350.model
│   └── wide_resnet_15_350.model
├── Run4
│   ├── wide_resnet_16_350.model
│   ├── wide_resnet_17_350.model
│   ├── wide_resnet_18_350.model
│   ├── wide_resnet_19_350.model
│   └── wide_resnet_20_350.model
└── Run5
    ├── wide_resnet_21_350.model
    ├── wide_resnet_22_350.model
    ├── wide_resnet_23_350.model
    ├── wide_resnet_24_350.model
    └── wide_resnet_25_350.model
```
Following are the main parameters for evaluation:
```
--seed: seed used for initializing the first trained model
--dataset: dataset used for training (cifar10/cifar100)
--ood_dataset: OoD dataset to compute AUROC
--load-path: /path/to/pretrained/models/
--model: model architecture to load (wide_resnet/vgg16)
--runs: number of experimental runs
-sn: whether the model was trained using spectral normalization
--coeff: Coefficient for spectral normalization
-mod: whether the model was trained using architectural modifications
--ensemble: number of models in the ensemble
--model-type: type of model to load for evaluation (softmax/ensemble/gmm)
```
As an example, in order to evaluate a Wide-ResNet-28-10 with spectral normalization and architectural modifications on CIFAR-10 with OoD dataset as SVHN, use the following:
```
python evaluate.py \
       --seed 1 \
       --dataset cifar10 \
       --ood_dataset svhn \
       --load-path /path/to/pretrained/models/ \
       --model wide_resnet \
       --runs 5 \
       -sn -mod \
       --coeff 3.0 \
       --model-type softmax
```
Similarly, to evaluate the above model using feature density, set ```--model-type gmm```. The evaluation script assumes that the seeds of models trained in consecutive runs differ by 1. The script stores the results in a json file with the following structure: 
```
{
    "mean": {
        "accuracy": mean accuracy,
        "ece": mean ECE,
        "m1_auroc": mean AUROC using log density / MI for ensembles,
        "m1_auprc": mean AUPRC using log density / MI for ensembles,
        "m2_auroc": mean AUROC using entropy / PE for ensembles,
        "m2_auprc": mean AUPRC using entropy / PE for ensembles,
        "t_ece": mean ECE (post temp scaling)
        "t_m1_auroc": mean AUROC using log density / MI for ensembles (post temp scaling),
        "t_m1_auprc": mean AUPRC using log density / MI for ensembles (post temp scaling),
        "t_m2_auroc": mean AUROC using entropy / PE for ensembles (post temp scaling),
        "t_m2_auprc": mean AUPRC using entropy / PE for ensembles (post temp scaling)
    },
    "std": {
        "accuracy": std error accuracy,
        "ece": std error ECE,
        "m1_auroc": std error AUROC using log density / MI for ensembles,
        "m1_auprc": std error AUPRC using log density / MI for ensembles,
        "m2_auroc": std error AUROC using entropy / PE for ensembles,
        "m2_auprc": std error AUPRC using entropy / PE for ensembles,
        "t_ece": std error ECE (post temp scaling),
        "t_m1_auroc": std error AUROC using log density / MI for ensembles (post temp scaling),
        "t_m1_auprc": std error AUPRC using log density / MI for ensembles (post temp scaling),
        "t_m2_auroc": std error AUROC using entropy / PE for ensembles (post temp scaling),
        "t_m2_auprc": std error AUPRC using entropy / PE for ensembles (post temp scaling)
    },
    "values": {
        "accuracy": accuracy list,
        "ece": ece list,
        "m1_auroc": AUROC list using log density / MI for ensembles,
        "m2_auroc": AUROC list using entropy / PE for ensembles,
        "t_ece": ece list (post temp scaling),
        "t_m1_auroc": AUROC list using log density / MI for ensembles (post temp scaling),
        "t_m1_auprc": AUPRC list using log density / MI for ensembles (post temp scaling),
        "t_m2_auroc": AUROC list using entropy / PE for ensembles (post temp scaling),
        "t_m2_auprc": AUPRC list using entropy / PE for ensembles (post temp scaling)
    },
    "info": {dictionary of args}
}
```


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

