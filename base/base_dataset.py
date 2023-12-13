#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : base_dataset.py
# @IDE     : vscode

"""
Implement Base Dataset Class
"""

from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for custom datasets, inheriting from the torch.utils.data.Dataset class.

    This class provides the basic structure that derived dataset classes should follow.
    Inherited classes are required to implement the __len__() and __getitem__() methods.
    """

    def __len__(self) -> int:
        """
        Abstract method that must be implemented by inherited classes.

        Returns:
            int: The total number of samples in the dataset.

        Raises:
            NotImplementedError: If not overridden by the derived class.
        """
        raise NotImplementedError(
            'Must implement __len__() method in the derived class.'
        )

    def __getitem__(self, idx: int) -> Any:
        """
        Abstract method that must be implemented by inherited classes.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Any: The sample data at the given index.

        Raises:
            NotImplementedError: If not overridden by the derived class.
        """
        raise NotImplementedError(
            'Must implement __getitem__() method in the derived class.'
        )
