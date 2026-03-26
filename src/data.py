from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_classifier_train_transform():
    """Build train-time image augmentations for supervised classifier training."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def get_eval_transform():
    """Build deterministic preprocessing transform for evaluation and embedding extraction."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def get_cifar10_train(root, transform=None):
    """Returns the CIFAR-10 training dataset"""
    return datasets.CIFAR10(root=root, train=True, download=True, transform=transform)


def get_cifar10_test(root, transform=None):
    "Returns CIFAR-10 test dataset"
    return datasets.CIFAR10(root=root, train=False, download=True, transform=transform)


def make_subset_loader(dataset, indices: Sequence[int], batch_size, shuffle, num_workers):
    """Create a DataLoader over a subset of dataset indices."""
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )