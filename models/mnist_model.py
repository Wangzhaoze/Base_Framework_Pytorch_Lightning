#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-20
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : mnist_model.py
# @IDE     : vscode

"""
Implement CLF for MNIST Dataset
"""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer


class MNISTModel(pl.LightningModule):
    """
    MNISTModel:
    A PyTorch Lightning module for MNIST digit classification.
    Inherits from: pytorch_lightning.LightningModule
    """

    def __init__(self):
        super().__init__()

        # Define the architecture of the neural network
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1792),
            nn.ReLU(),
            nn.Linear(1792, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.nll_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.nll_loss(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
