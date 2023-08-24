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

from torch import nn
from torch import optim
import pytorch_lightning as pl

class MNISTModel(pl.LightningModule):
    """
    MNISTModel:
        A PyTorch Lightning module for MNIST digit classification.
        Inherits from: pytorch_lightning.LightningModule
    """
    def __init__(self):
        super().__init__()  # Used Python 3 style super() without arguments

        # Define the architecture of the neural network
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1792),
            nn.ReLU(),
            nn.Linear(1792, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.nll_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)