"""
IO module for train/test regression datasets
"""
import os
import numpy as np
from torchvision import datasets, transforms

def _load_mnist():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(current_dir, 'data')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    input_shape=[28,28,1]
    num_classes=10
    return train_dataset, mnist_test, fashion_mnist_test, input_shape, num_classes

def load_dataset(name):
    # load full dataset
    load_funs = { "mnist": _load_mnist}
    train_dataset, test_dataset, ood_dataset, input_shape, num_classes = load_funs[name]()


    print("Done loading dataset {}".format(name))
    return train_dataset, test_dataset, ood_dataset, input_shape, num_classes