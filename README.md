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

We recommend using [conda](https://docs.conda.io/en/latest/) to manage dependencies.  
All required packages and versions are specified in `environment.yml`.

### Step 1: Clone the repository
```bash
git clone https://github.com/zzz0527/SPC-UQ.git
cd SPC-UQ
```
### Step 2: Create and activate the environment
```bash
conda env create -f environment.yml
conda activate spc_uq
```

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
