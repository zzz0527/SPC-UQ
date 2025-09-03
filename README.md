# SPC-UQ: A Post-hoc, Efficient, and Unified Uncertainty Quantification Framework

This repository provides the official implementation and evaluation code for **SPC-UQ** (Split-Point Consistency for Uncertainty Quantification), a post-hoc, single-forward-pass framework for jointly quantifying **aleatoric** and **epistemic** uncertainty across diverse deep learning tasks.

This work is associated with the paper:  
**"Post-Hoc Split-Point Self-Consistency Verification for Efficient, Unified Quantification of Aleatoric and Epistemic Uncertainty in Deep Learning"**  
(*To appear in IEEE TPAMI / Under Review*)
---

## 📌 Key Features

- **Post-hoc**: Works with pretrained models without architectural modification.
- **Unified**: Applicable to both regression and classification tasks.
- **Efficient**: Requires only a single forward pass for UQ inference.
- **Decomposed Uncertainty**: Separates aleatoric and epistemic signals via self-consistency constraints.

---

## 📊 Benchmarks

SPC-UQ is evaluated on the following tasks:

### ✅ Regression
- **Scalar regression**: Standard UCI datasets (e.g., Boston Housing, Energy).
- **High-dimensional multinomial regression**: Custom synthetic or multimodal datasets.

### ✅ Classification
- **Image classification**: CIFAR-10 / CIFAR-100 using CNN backbones.
- **Multimodal classification**: LUMA dataset with vision, audio, and text inputs.

---

## 🔬 Results Summary

Compared to representative UQ baselines (e.g., MC Dropout, Deep Ensembles, Evidential Models), SPC-UQ achieves the **best trade-off** among:
- **Predictive accuracy**
- **Computational efficiency**
- **General applicability** across tasks and architectures

---

## 📁 Project Structure

```bash
SPC-UQ/
│
├── models/                  # Backbone models and UQ heads
├── trainers/                # Training and evaluation logic
├── data/                    # Dataloaders for all benchmarks
├── configs/                 # Experiment configurations
├── results/                 # Evaluation results and plots
├── utils/                   # Miscellaneous utilities
└── main.py                  # Entry point for running experiments
