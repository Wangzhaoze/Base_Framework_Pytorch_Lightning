#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : base_model.py
# @IDE     : vscode

"""
Implement Base Model Class
"""

from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):
    """
    Base class for PyTorch Lightning models.

    This class provides a foundation for custom models in PyTorch Lightning.
    Derived classes can override its methods to define the model's architecture,
    training loop, and optimization logic.
    """

    def __init__(self):
        """
        Initializes a new instance of the BaseModel class.
        """
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Define the forward pass of the model.

        Args:
            x (Any): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

    def training_step(
        self, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Define a single training step on the training set.

        Returns:
            Union[torch.Tensor, Dict[str, Any]]: Training loss or a dictionary containing
            the training loss and other optional items.
        """

    def validation_step(
        self, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Define a single validation step on the validation set.

        Returns:
            Union[torch.Tensor, Dict[str, Any]]: Validation loss or a dictionary containing
            the validation loss and other optional items.
        """

    def test_step(
        self, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Define a single testing step on the test set.

        Returns:
            Union[torch.Tensor, Dict[str, Any]]: Test result or a dictionary containing
            the test result and other optional items.
        """

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Define the optimizer and learning rate scheduler.

        Returns:
            torch.optim.Optimizer:
            Optimizer or a tuple containing the optimizer and learning rate scheduler.
        """
