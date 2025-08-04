import os
import torch
import numpy as np
from torch.utils.data import Subset, ConcatDataset, DataLoader
import random

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def get_svhn_test_loader(batch_size, imagesize=128, num_workers=1, pin_memory=False, **kwargs):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transform
    transform = transforms.Compose([transforms.Resize(imagesize), transforms.ToTensor(), normalize,])


    data_dir = "./data"
    dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=transform,)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_tinyimagenet_test_loader(batch_size, imagesize=128, num_workers=1, pin_memory=False, **kwargs):

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


def get_cifar10_test_loader(batch_size, imagesize=128, num_workers=1, pin_memory=False, **kwargs):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transform
    transform = transforms.Compose([transforms.Resize(imagesize), transforms.ToTensor(), normalize,])

    data_dir = "./data"
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform,)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_cifar100_test_loader(batch_size, imagesize=128, num_workers=1, pin_memory=False, **kwargs):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transform
    transform = transforms.Compose([transforms.Resize(imagesize), transforms.ToTensor(), normalize,])

    data_dir = "./data"
    dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform,)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader



def get_combined_ood_test_loader(batch_size, sample_seed, imagesize=128, num_workers=1, pin_memory=False, sample_size=10000, **kwargs):
    svhn_ds = get_svhn_test_loader(batch_size=1, imagesize=imagesize).dataset
    tiny_ds = get_tinyimagenet_test_loader(batch_size=1, imagesize=imagesize).dataset
    # cifar10_ds = get_cifar10_test_loader(batch_size=1, imagesize=imagesize).dataset
    # cifar100_ds = get_cifar100_test_loader(batch_size=1, imagesize=imagesize).dataset

    combined_dataset = ConcatDataset([
        svhn_ds,
        tiny_ds,
        # cifar10_ds,
        # cifar100_ds
    ])

    # print(len(combined_dataset))

    random.seed(sample_seed)
    if sample_size is not None and sample_size < len(combined_dataset):
        indices = random.sample(range(len(combined_dataset)), sample_size)
        combined_dataset = Subset(combined_dataset, indices)

    data_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader