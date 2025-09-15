# Image Classification

This directory contains experiments for **image classification** on standard benchmarks including **CIFAR-10**, **CIFAR-100**, **SVHN**, **TinyImageNet**, and **ImageNet**.  

The implementation is based on [DDU](https://github.com/omegafragger/DDU), with the following extensions:
- Additional **baseline methods** (e.g., Laplace Approximation, Evidential models, Orthonormal Certificates).  
- The OOD detection task is **extended** to include:
  - Misclassification detection  
  - Adversarial sample detection  
  - Classical OOD detection


##  Dataset Preparation
- **CIFAR-10**, **CIFAR-100**, and **SVHN**  
  These datasets will be downloaded automatically by the code.  

- **TinyImageNet** and **ImageNet-1K**  
  Due to their large size, users need to **manually download** them. Official links:  
  - [TinyImageNet (CS231n version)](http://cs231n.stanford.edu/tiny-imagenet-200.zip)  
  - [ImageNet-1K](https://www.image-net.org/)  

- **ImageNet-O/A**  
  Please follow the download guide provided here:  
  [hendrycks/natural-adv-examples](https://github.com/hendrycks/natural-adv-examples?tab=readme-ov-file)  

**Note:** All datasets should be placed in the [data](data) directory.


##  Additional Environment Requirements
Since the application of Laplace approximation depends on the [Laplace library](https://github.com/aleximmer/Laplace). It requires extra dependencies beyond the default `environment.yml`. 

If you want to run the Laplace approximation baseline, please install the following libraries manually:
```
pip install laplace-torch
pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl
```


## Pretrained model weights
We provide several pretrained model weights to facilitate **reproducibility** of our experiments.

All pretrained model directories are prefixed with **'model_'**. Please download the weights from our [Hugging Face directory](https://huggingface.co/zzz0527/SPC-UQ/tree/main/SPC-UQ/Image_Classification) and place them **directly under this directory**. Once downloaded, you can directly run the provided evaluation scripts without additional training.  


## Training
In order to train a model, use the [train.py](train.py) script. 

Key arguments:
- `--dataset {cifar10,cifar100,svhn,imagenet,tinyimagenet}` – dataset for training.
- `--dataset-root PATH` – root directory for datasets.
- `--model {lenet,resnet18,resnet50,wide_resnet,vgg16,...}` – network architecture.
- `-sn` / `--coeff` – enable spectral normalization (available for wide_resnet, vgg16 and resnets).
- `-mod` – use architectural modifications (leaky ReLU + average pooling in skip connections).
- `-b` – batch size.
- `-e` – number of training epochs.
- `--lr`, `--mom`, `--opt` – optimizer settings.
- `--save-path PATH` – where to save checkpoints.

As an example, in order to train a Wide-ResNet-28-10 with spectral normalization and architectural modifications on CIFAR-10, use the following:
```
python train.py \
       --seed 1 \
       --dataset cifar10 \
       --model wide_resnet \
       -sn -mod \
       --coeff 3.0 
```
Similarly, to train a VGG-16 without spectral normalization on CIFAR-100, use:
```
python train.py \
       --seed 1 \
       --dataset cifar100 \
       --model vgg16
```

## Evaluation

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

Following are the main parameters for evaluation:
```
--seed: seed used for initializing the first trained model
--dataset: dataset used for training (cifar10/cifar100/imagenet)
--ood_dataset: OOD dataset (ood_union/svhn/tinyimagenet/imagenet_o)
--load-path: /path/to/pretrained/models/
--model: model architecture to load (wide_resnet/vgg16)
--runs: number of experimental runs
-sn: whether the model was trained using spectral normalization
--coeff: Coefficient for spectral normalization
-mod: whether the model was trained using architectural modifications
--ensemble: number of models in the ensemble
--model-type: type of model to load for evaluation (softmax/ensemble/edl/gmm/oc/spc)
```

As an example, in order to evaluate a Wide-ResNet-28-10 with spectral normalization and architectural modifications on CIFAR-10 with the union OOD dataset, use the following:
```
python evaluate.py \
    --seed 1 \
    --dataset cifar10 \
    --ood_dataset ood_union \
    --load-path model_vgg_10 \
    --model vgg16 \
    --runs 25 \
    --model-type spc
```

In order to evaluate a Wide-ResNet-28-10 with spectral normalization and architectural modifications on CIFAR-10 with OOD dataset as SVHN, use the following:
```
python evaluate.py \
    --seed 1 \
    --dataset cifar10 \
    --ood_dataset svhn \
    --load-path model_wide_10 \
    --model wide_resnet \
    --runs 25 \
    -sn -mod \
    --coeff 3.0 \
    --model-type spc
```

Additional script `evaluate_laplace.py` for Laplace approximation evaluation.

We provide bash scripts for all baseline methods used in our experiments. If you download the datasets and pre-trained models into this directory, you can directly execute the corresponding script to reproduce results.
