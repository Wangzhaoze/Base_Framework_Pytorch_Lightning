#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : mnist_datamodule.py
# @IDE     : vscode

"""
Implement MNIST DataModule for PyTorch Lightning.
"""

import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


class MNIST_DataModule(pl.LightningDataModule):
    """
    DataModule class for MNIST dataset in PyTorch Lightning.

    This class defines the data loading and processing steps for the MNIST dataset.
    """
    def __init__(self) -> None:
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        

    def prepare_data(self) -> None:
        """
        Implement data download or preprocessing steps here (optional).
        This method is called only once and on a single GPU.
        """
        # Download or load MNIST dataset
        self.dataset = datasets.MNIST(root='./data',
                                      train=True,
                                      transform=self.transform,
                                      download=True)

    def setup(self, stage: str = None) -> None:
        """
        Load and split the dataset into training and validation sets.

        Args:
            stage (str): One of 'fit' (train), 'validate', 'test', or None.
        """
        dataset_size = len(self.dataset)

        train_size = int(0.8 * dataset_size)
        val_size = int(0.2 * dataset_size)

        # Randomly split the dataset based on the split ratios
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """
        return DataLoader(dataset=self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
        return DataLoader(dataset=self.val_dataset, batch_size=64, shuffle=False)
    