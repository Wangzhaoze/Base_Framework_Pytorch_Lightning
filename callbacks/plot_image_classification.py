#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-08-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/BNFPL
# @File    : image_classification_visualization.py
# @IDE     : vscode

"""
Implement Callback for Image Visualization and Label Prediction
"""
import random

import pytorch_lightning as pl
import torch


class PlotImageClassification(pl.Callback):
    """
    Callback for visualizing image classification results during validation step.

    This callback
        - randomly selects a subset of images from the validation dataset,
        - makes predictions using the trained model, and
        - visualizes the images and preded label ussing the experiment logger.

    Args:
        num_images (int): Number of random images to visualize in each validation epoch.
    """

    def __init__(self, num_images: int = 10):
        """
        Initializes the ImagesClassificationVisualization callback.

        Args:
            num_images (int): Number of random images to visualize in each validation epoch.
        """
        self.num_images = num_images

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Method called at the end of each validation epoch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer object.
            pl_module (LightningModule): The Lightning module being trained.
        """
        # Get the validation dataset
        self.val_dataset = trainer.datamodule.val_dataset

        # Randomly select `num_images` images for visualization
        random_indices = random.sample(range(len(self.val_dataset)), self.num_images)

        selected_images = [self.val_dataset[i][0] for i in random_indices]
        selected_labels = [self.val_dataset[i][1] for i in random_indices]

        # Stack selected images and move them to the appropriate device
        selected_images = torch.stack(selected_images)
        selected_labels = torch.tensor(selected_labels)
        selected_images = selected_images.to(pl_module.device)

        # Make predictions on selected images
        with torch.no_grad():
            predicted = pl_module(selected_images)
            _, preds = torch.max(predicted, dim=1)

        logger = trainer.logger.experiment
        for i in range(self.num_images):
            title = f'label: {selected_labels[i]}, prediction:{preds[i]}'
            logger.add_images(
                title, selected_images[i].unsqueeze(0), pl_module.current_epoch
            )
