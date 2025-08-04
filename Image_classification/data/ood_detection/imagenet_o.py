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



def get_test_loader(batch_size, imagesize=224, num_workers=1, pin_memory=False,  **kwargs):
    """
    加载 ImageNet-O 数据集作为无标签的图像列表（可用于OOD评估）
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    root = "./data/imagenet-o"
    dataset = datasets.ImageFolder(
        root=root,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader
