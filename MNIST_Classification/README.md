# MNIST Classification

This directory provides a simple experiment that trains a model on the MNIST handwritten digit dataset and measures uncertainty on out-of-distribution inputs.

## Dataset
- **In-distribution**: MNIST training and test sets.
- **Out-of-distribution**: Fashion-MNIST test set for uncertainty evaluation.
- The loader expects the MNIST dataset under `MNIST_Classification/data`. If the dataset is not present, download it from our [Hugging Face directory](https://huggingface.co/zzz0527/SPC-UQ/tree/main/SPC-UQ/MNIST_Classification)

## Usage
Run the classification script from this directory:

```bash
python run_cls_tests.py [--num-trials N] [--num-epochs E]
```
The benchmarks are lightweight and can run on **CPU-only devices** (GPU is optional for faster training).  

### Arguments
- `--num-trials` *(default: 5)* — number of training repetitions for statistical averaging.
- `--num-epochs` *(default: 40)* — training epochs per trial.

The script reports accuracy, confident accuracy, uncertain accuracy, AUROC, and training/evaluation times for each setting.
