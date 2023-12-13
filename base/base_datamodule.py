#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : base_datamodule.py
# @IDE     : vscode

"""
Implement Base PyTorch Lightning DataModule Class
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for PyTorch Lightning data modules.

    This class provides a foundation for custom data modules in PyTorch Lightning.
    Derived classes can override its methods to customize the data loading and processing behavior.
    """

    def __init__(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def prepare_data(self):
        """
        Implement data download or preprocessing steps here (optional).
        This method is called only once and on a single GPU.
        """

    def setup(self, stage: str = None):
        """
        Load and split your dataset here.

        Args:
            stage (str): One of 'fit' (train), 'validate', 'test', or None.
        """

    def train_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for training.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """

    def val_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for validation.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
