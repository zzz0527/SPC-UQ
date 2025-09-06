# Multimodal Classification

This directory contains experiments for **multimodal classification** on the multimodal UQ benchmark: **LUMA** (Learning from Uncertain and Multimodal Data)

The implementation is based on [LUMA](https://github.com/omegafragger/DDU), with the implementation of SPC-UQ.

## Overview

LUMA is a multimodal dataset designed for benchmarking multimodal learning and multimodal uncertainty quantification. This dataset includes audio, text, and image modalities, enabling researchers to study uncertainty quantification in multimodal classification settings.


## Dataset Summary

LUMA consists of:
- **Audio Modality**: `wav` files of people pronouncing the class labels of the selected 50 classes.
- **Text Modality**: Short text passages about the class labels, generated using large language models.
- **Image Modality**: Images from a 50-class subset from CIFAR-10/100 datasets, as well as generated images from the same distribution.

The dataset allows controlled injection of uncertainties, facilitating the study of uncertainty quantification in multimodal data.

## Getting Started

### Prerequisites

- Anaconda / Miniconda
- Git

### Installation
Copy the data folder into this repository:

```bash
cp -r ../data .
```
Install and activate the conda enviroment
```bash
conda env create -f environment.yml
conda activate luma_env
```


### Usage
The provided Python tool allows compiling different versions of the dataset with various amounts and types of uncertainties.

To compile the dataset with specified uncertainties, create or edit the configuration file similar to the files in `cfg` directory, and run:
```
python compile_dataset.py -c <your_yaml_config_file>
```

### Usage in Deep Learning models
After compiling the dataset, you can use the `LUMADataset` class from the `dataset.py` file. Example of the usage can be found in `run_baselines.py` file.

### Unprocessed & Unaigned data
If you want to get all the data (without sampling or noise) without alignment (to perform your own alignment, or use the data without alignment for other tasks) you can run the following command:

```
python get_unprocessed_data.py
```
