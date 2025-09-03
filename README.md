# SPC-UQ: A Post-hoc, Efficient, and Unified Uncertainty Quantification Framework

This repository contains the official code for **SPC-UQ** (Split-Point Consistency for Uncertainty Quantification), a post-hoc framework that jointly quantifies aleatoric and epistemic uncertainty with a single forward pass.

It accompanies the paper:

**"Post-Hoc Split-Point Self-Consistency Verification for Efficient, Unified Quantification of Aleatoric and Epistemic Uncertainty in Deep Learning."**

## Key Features

- **Post-hoc** – augments pre-trained network without architectural changes and retraining.
- **Unified** – supports both regression and classification tasks in deep learning.  
- **Efficient** – produces aleatoric and epistemic uncertainty estimates in one forward pass.
- **Calibration** – provides mechanisms to calibrate aleatoric uncertainty and improve predictive reliability. 

## Repository Structure

```
Cubic_Regression/             # Toy cubic regression for fast demonstration.
MNIST_Classification/         # Digit classification for fast demonstration.
UCI_Benchmarks/               # Standard UCI regression datasets for scalar regression evaluation.
Monocular_Depth_Estimation/   # Monocular end-to-end image depth estimation for high-dimensional regression.
Image_Classification/         # CIFAR-10/100, ImageNet-1K for large-scale image classification.
Multimodal_Classification/    # LUMA multimodal benchmark (image/audio/text) for multimodal classification tasks.
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
