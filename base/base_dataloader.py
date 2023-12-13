#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : base_dataloader.py
# @IDE     : vscode

"""
Implement Base Dataloader Class
"""

from torch.utils.data import DataLoader, Dataset


class BaseDataLoader(DataLoader):
    """
    Base class for data loader, inheriting from torch.utils.data.DataLoader.

    This class provides a foundation for custom data loaders and extends the functionality
    of the torch.utils.data.DataLoader class. Derived classes can override its methods
    to customize the data loading behavior.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """
        Initializes a new instance of the BaseDataLoader class.

        Args:
            dataset (Dataset): The custom Dataset object to be used for data loading.
            batch_size (int): Number of samples in a batch.
            shuffle (bool): Whether to shuffle at the beginning of each epoch. Default is True.
            num_workers (int): Number of worker threads used for data loading. Default is 0.
        """
        # Call the constructor of the parent DataLoader class
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
