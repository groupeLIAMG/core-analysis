# -*- coding: utf-8 -*-

"""Geometrical and data augmentation utilities."""

import numpy as np
from scipy.interpolate import griddata
from imgaug.augmenters import (
    Sequential,
    Affine,
    Fliplr,
    Flipud,
    # Rot90,
    # AdditiveGaussianNoise as Noise,
)


def adjust_rgb(img, perc_init=5, perc_final=95):
    dim = img.shape
    adjusted_img = np.zeros(dim)

    for n in range(dim[-1]):
        channel = img[:, :, n]
        perc_i = np.percentile(channel, perc_init)
        perc_f = np.percentile(channel, perc_final)
        channel = np.clip(channel, perc_i, perc_f)
        channel = normalize(channel)
        adjusted_img[:, :, n] = channel

    return adjusted_img


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def min_dist(x_n, y_n, pairs):
    min_distance = np.inf
    for x_f, y_f in pairs:
        distance = np.sqrt((x_n - x_f) ** 2 + (y_n - y_f) ** 2)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def undersample(image, mask=None, undersample_by=2):
    image = image[::undersample_by, ::undersample_by]
    if mask is not None:
        mask = mask[::undersample_by, ::undersample_by]
    return image, mask


def upsample(arr, dest_shape):
    y0 = np.arange(arr.shape[0])
    x0 = np.arange(arr.shape[1])
    x0, y0 = np.meshgrid(x0, y0)

    y = np.linspace(y0.min(), y0.max(), dest_shape[0])
    x = np.linspace(x0.min(), x0.max(), dest_shape[1])
    x, y = np.meshgrid(x, y)

    arr = griddata(
        (x0.ravel(), y0.ravel()),
        arr[:, :, 0].ravel(),
        (x, y),
        method="nearest",
    )

    return arr


augment = Sequential(
    [
        Affine(scale=[0.5, 1.0, 2.0]),
        Fliplr(p=0.5),
        Flipud(p=0.5),
        # Rot90(k=(0, 1, 2, 3)),
        # Noise(scale=(0.0, 0.05)),
    ]
)
