"""
Create train, valid, test iterators for Tiny-ImageNet.
Train set size: 90000 (450 per class)
Val set size: 10000 (50 per class)
Test set size: 10000 (no labels)
"""

import torch
import numpy as np
import os
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def get_train_valid_loader(batch_size, augment, val_seed, imagesize=128, val_size=0.1, num_workers=1, pin_memory=False, **kwargs):
    """
    Load and return train and valid iterators over the Tiny-ImageNet dataset.

    Params:
    ------
    - data_dir: path to Tiny-ImageNet dataset directory.
    - batch_size: number of samples per batch.
    - augment: whether to apply data augmentation.
    - val_seed: random seed for reproducibility.
    - val_size: fraction of the training set used for validation (0 to 1).
    - num_workers: number of subprocesses for data loading.
    - pin_memory: set to True if using GPU.

    Returns:
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    assert 0 <= val_size <= 1, "[!] val_size should be in the range [0, 1]."

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Define transforms
    valid_transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(256, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = valid_transform

    # Load dataset
    data_dir = "./data/tinyimagenet"
    train_dir = os.path.join(data_dir, "train")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = ImageFolder(root=train_dir, transform=valid_transform)  # Same dataset, different transform

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False
    )

    return train_loader, valid_loader


def get_test_loader(batch_size, imagesize=128, num_workers=1, pin_memory=False, **kwargs):
    """
    Load and return a test iterator over the Tiny-ImageNet dataset.

    Params:
    ------
    - data_dir: path to Tiny-ImageNet dataset directory.
    - batch_size: number of samples per batch.
    - num_workers: number of subprocesses for data loading.
    - pin_memory: set to True if using GPU.

    Returns:
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
        normalize
    ])

    data_dir = "./data/tinyimagenet"
    test_dir = os.path.join(data_dir, "test")
    dataset = ImageFolder(root=test_dir, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader
