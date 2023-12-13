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
import numpy as np
import os
import cv2
import os
from typing import Any
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset



import random
import torch
import torchvision

import torchvision.transforms.functional as F
from PIL import Image


class CityScapeDataset(Dataset):
    """
    Base class for segmentation datasets.

    Args:
        image_dir (str): Path to the directory containing input images.
        label_dir (str): Path to the directory containing corresponding label images.
        transform (callable, optional): A function/transform to apply to each image and label.

    Attributes:
        image_filenames (list): List of image filenames in the dataset.
        label_filenames (list): List of label filenames in the dataset.
    """
    def __init__(self, image_dir: str, label_dir: str, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_filenames = os.listdir(image_dir)
        self.label_filenames = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Get the images and labels for a batch of indices.

        Args:
            idx (int): List of indices for the batch.

        Returns:
            images (torch.Tensor): Batch of loaded input images.
            labels (torch.Tensor): Batch of loaded label images.
        """

        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = cv2.imread(image_path, cv2.COLOR_RGB2GRAY)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


class SegmentationTransform():
    def __init__(self):
        super().__init__()
        self.transforms = [
            scale,
            to_tensor,
            to_chw
        ]
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> tuple:
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label
    

def to_tensor(image, label):
    return torch.from_numpy(image).float(), torch.from_numpy(label).float()

def to_chw(image, label):
    return image.permute(2, 0, 1), label.permute(2, 0, 1)

def scale(image, label):
    return image / np.max(image), label / np.max(label)

def random_rotate(img, mask, angle_range=(-90, 90)):
    angle = random.uniform(angle_range[0], angle_range[1])
    img = F.rotate(img, angle, resample=Image.BILINEAR)
    mask = F.rotate(mask, angle, resample=Image.NEAREST)
    return img, mask




class CityScapeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, class_name: str,  batch_size:int, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.augmentation = SegmentationTransform()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.class_name = class_name

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            self.train_dataset = CityScapeDataset(
                image_dir=os.path.join(self.data_dir, self.class_name, 'images'),
                label_dir=os.path.join(self.data_dir, self.class_name, 'labels'),
                transform=self.augmentation
            )

            self.val_dataset = CityScapeDataset(
                image_dir=os.path.join(self.data_dir, self.class_name, 'images'),
                label_dir=os.path.join(self.data_dir, self.class_name, 'labels'),
                transform=self.augmentation
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )