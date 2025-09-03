# SPC-UQ: A Post-hoc, Efficient, and Unified Uncertainty Quantification Framework

This repository contains the official code for **SPC-UQ** (Split-Point Consistency for Uncertainty Quantification), a post-hoc framework that jointly quantifies aleatoric and epistemic uncertainty with a single forward pass.

It accompanies the paper:

**"Post-Hoc Split-Point Self-Consistency Verification for Efficient, Unified Quantification of Aleatoric and Epistemic Uncertainty in Deep Learning."**

## Key Features

- **Post-hoc** – augment any pre-trained network without architectural changes.
- **Unified** – works for regression, classification and structured prediction.
- **Efficient** – one forward pass yields both aleatoric and epistemic estimates.

## Repository Structure

```
Cubic_Regression/           # Toy cubic regression with multiple UQ baselines.
MNIST_Classification/       # Handwritten digit classification experiments.
UCI_Benchmarks/             # Standard UCI regression datasets.
Image_classification/       # CIFAR/Dirty-MNIST classification and OoD detection.
Monocular_Depth_Estimation/ # Monocular depth estimation experiments.
Multimodal_classification/  # LUMA multimodal dataset utilities.
```

Each directory provides scripts to reproduce the corresponding experiments.

## Installation

SPC-UQ requires Python 3 and [PyTorch](https://pytorch.org/). Install the core dependencies with:

```
pip install torch torchvision numpy matplotlib
```

Some tasks need additional packages such as `h5py` (depth estimation) or the dependencies listed in `Image_classification/environment.yml` and `Multimodal_classification/code/requirements.txt`.

## Usage

Example entry points:

```
# Synthetic cubic regression
python Cubic_Regression/run_cubic_tests.py

# UCI regression benchmarks
python UCI_Benchmarks/run_uci_dataset_tests.py --dataset energy

# MNIST classification
python MNIST_Classification/run_cls_tests.py
```

For CIFAR or Dirty-MNIST experiments, refer to `Image_classification/README.md`.  
Depth estimation experiments can be started with `python Monocular_Depth_Estimation/train_depth.py`.  
The multimodal LUMA utilities are in `Multimodal_classification/code/`.  
See the documentation in each subdirectory for dataset preparation and additional options.

## Citation

If you use SPC-UQ in your research, please cite our paper:

```
@article{spc_uq_2024,
  title={Post-Hoc Split-Point Self-Consistency Verification for Efficient, Unified Quantification of Aleatoric and Epistemic Uncertainty in Deep Learning},
  year={2024}
}
```
