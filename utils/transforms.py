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


def random_rotate(img, mask, angle_range=(-90, 90)):
    angle = random.uniform(angle_range[0], angle_range[1])
    img = F.rotate(img, angle, resample=Image.BILINEAR)
    mask = F.rotate(mask, angle, resample=Image.NEAREST)
    return img, mask


def random_affine(img, mask, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)):
    angle = random.uniform(-degrees, degrees)
    translate_x = random.uniform(-translate[0], translate[0])
    translate_y = random.uniform(-translate[1], translate[1])
    scale = random.uniform(scale[0], scale[1])

    img = F.affine(img, angle, (translate_x, translate_y), scale, resample=Image.BILINEAR, shear=0)
    mask = F.affine(mask, angle, (translate_x, translate_y), scale, resample=Image.NEAREST, shear=0)

    return img, mask


# 定义随机翻转函数
def random_flip(img, mask):
    if random.random() < 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)

    if random.random() < 0.5:
        img = np.flip(img, axis=2)
        mask = np.flip(mask, axis=2)

    return img, mask


def random_color_jitter(img, mask, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    transform = torchvision.transforms.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    img = transform(img)

    return img, mask