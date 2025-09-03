"""
IO module for train/test regression datasets
"""
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


def generate_cubic(x, noise=False):
    x = x.astype(np.float32)
    y = x**3

    if noise:
        sigma = 3 * np.ones_like(x)
    else:
        sigma = np.zeros_like(x)
    r = np.random.normal(0, sigma).astype(np.float32)
    return y+r, sigma


#####################################
# individual data files             #
#####################################
vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/uci")

def _load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    input_shape=[28,28,1]
    num_classes=10
    return train_dataset, mnist_test, fashion_mnist_test, input_shape, num_classes

def load_dataset(name):
    # load full dataset
    load_funs = { "mnist"        : _load_mnist}

    train_dataset, test_dataset, ood_dataset, input_shape, num_classes = load_funs[name]()


    print("Done loading dataset {}".format(name))
    return train_dataset, test_dataset, ood_dataset, input_shape, num_classes