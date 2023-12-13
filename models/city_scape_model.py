import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from typing import Any, List, Tuple
from models.backbones import UNet

class CityScapeModel(pl.LightningModule):
    """
    MNISTModel:
    A PyTorch Lightning module for MNIST digit classification.
    Inherits from: pytorch_lightning.LightningModule
    """
    def __init__(self):
        super().__init__()

        # Define the architecture of the neural network
        self.backbone = UNet(3, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self) -> Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer