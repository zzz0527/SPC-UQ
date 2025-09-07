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
All required packages and versions except Multimodal_Classification are specified in `environment.yml`.

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
### Step 3: Download pretrained weights and datasets
Pretrained weights and datasets are available on https://huggingface.co/zzz0527/SPC-UQ

## Usage

Each subdirectory corresponds to a specific benchmark.  
To run an experiment, navigate into the corresponding folder and follow the instructions provided in its `README.md`.  

### Quick Start
For a fast verification, we provide two lightweight benchmark tasks:

```bash
# Synthetic cubic regression
python Cubic_Regression/run_cubic_tests.py

# MNIST classification
python MNIST_Classification/run_cls_tests.py
```

See the documentation in each subdirectory for details on dataset preparation, configuration options, and advanced usage.

## Citation

If you use SPC-UQ in your research, please cite our paper:

```
@article{zhao2025spc,
  title   = {Post-Hoc Split-Point Self-Consistency Verification for Efficient, Unified Quantification of Aleatoric and Epistemic Uncertainty in Deep Learning},
  author  = {Zhao, ZZ and Chen, Ke},
  year    = {2025}
}
```
