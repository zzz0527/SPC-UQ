"""
Create train, valid, test iterators for ImageNet.
Train set size: user-defined
Val set size: user-defined
Test set size: user-defined (if available)
"""

import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_train_valid_loader(batch_size, augment, val_seed, imagesize=224, val_size=0.1, num_workers=1, pin_memory=False, **kwargs):
    assert 0 <= val_size <= 1, "[!] val_size should be in the range [0, 1]."

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imagesize = 224
    # Define transformations
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        normalize
    ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = valid_transform

    data_dir = "./data/Imagenet1K"
    # Load the dataset
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )

    return train_loader, valid_loader


def get_test_loader(batch_size, imagesize=224, num_workers=1, pin_memory=False, **kwargs):

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    data_dir = "./data/Imagenet1K"
    dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
