"""
file        : transforms.py
create date : October 15, 2024
author      : truong.manh.le@aioz.io
description : Redefine some transform functions
Aim to support transform the target image if it exists
"""

import random

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size: int, fill: int = 0):
    """Pads the image if its dimensions are smaller than the specified size.

    Args:
        img (_type_): The input image, either a PIL Image or a torch.Tensor.
        size (int): Target size for the smallest dimension.
        fill (int, optional): Padding value, can be an int or a tuple. Defaults to 0.

    Returns:
        _type_: The padded image or the original image if no padding is needed.
    """

    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        pad_h = size - oh if oh < size else 0
        pad_w = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, pad_w, pad_h), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        if target is not None:
            target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
            return image, target
        else:
            return image


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
                return image, target
            else:
                return image
        else:
            if target is not None:
                return image, target
            else:
                return image


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
            return image, target
        else:
            return image


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
            return image, target
        else:
            return image


class PILToTensor:
    def __call__(self, image, target=None):
        image = F.pil_to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            return image, target
        else:
            return image


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target=None):
        if target is not None:
            if not self.scale:
                return image.to(dtype=self.dtype), target
            image = F.convert_image_dtype(image, self.dtype)
            return image, target
        else:
            if not self.scale:
                return image.to(dtype=self.dtype)
            image = F.convert_image_dtype(image, self.dtype)
            return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            return image, target
        else:
            return image


class CVResize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image, target=None):
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if target is not None:
            target = cv2.resize(target, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            return image, target
        else:
            return image


class NumpyToTensor:
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            return image, target
        else:
            return image
