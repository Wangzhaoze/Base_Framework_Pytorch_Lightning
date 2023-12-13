#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : base_callback.py
# @IDE     : vscode

"""
Implement Base Callback
"""

import pytorch_lightning as pl


class BaseCallback(pl.Callback):
    """
    Base class for custom callbacks in PyTorch Lightning.

    This class provides hooks that can be overridden in derived classes to customize
    the behavior during different phases of the training process.

    Attributes:
        None

    """

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when a validation epoch begins.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer object.
            pl_module (pl.LightningModule): The PyTorch Lightning module being trained.

        Returns:
            None
        """

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when a validation epoch ends.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer object.
            pl_module (pl.LightningModule): The PyTorch Lightning module being trained.

        Returns:
            None
        """

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation loop begins.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer object.
            pl_module (pl.LightningModule): The PyTorch Lightning module being trained.

        Returns:
            None
        """

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation loop ends.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer object.
            pl_module (pl.LightningModule): The PyTorch Lightning module being trained.

        Returns:
            None
        """
