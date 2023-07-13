# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import griddata
from imgaug.augmenters import (
    Sequential,
    Fliplr,
    Flipud,
    Rot90,
    AdditiveGaussianNoise as Noise,
)


def normalize(x, lim=255.0):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) * lim


def adjust_rgb(img, perc_init=5, perc_final=95, nchannels=3):
    dim = img.shape
    adjusted_img = np.zeros((dim))

    if dim[-1] == nchannels:
        for n in range(nchannels):
            channel = img[:, :, n]
            perc_i = np.nanpercentile(channel, perc_init)
            perc_f = np.nanpercentile(channel, perc_final)
            channel = np.clip(channel, perc_i, perc_f)
            channel = normalize(channel, 1.0)
            adjusted_img[:, :, n] = channel

    else:
        raise ValueError(f"The shape should be (M, N, {nchannels}).")

    return adjusted_img


def min_dist(x_n, y_n, pairs):
    distances = [999999.0]
    if len(pairs) > 0:
        distances = []
        for pair in pairs:
            x_f, y_f = pair
            distances.append(np.sqrt(np.power(x_n - x_f, 2) + np.power(y_n - y_f, 2)))

    return np.min(distances)


def hillshade(array, azimuth, angle_altitude):
    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
        slope
    ) * np.cos(azimuthrad - aspect)
    hillshade_array = 255 * (shaded + 1) / 2

    return hillshade_array


def standardize_data(grid, data_format="channels_last"):
    dim = grid.shape
    std_grid = np.zeros(dim)

    for n in range(np.min(dim)):
        if data_format == "channels_last":
            data = (grid[:, :, n] - np.nanmean(grid[:, :, n])) / np.nanstd(
                grid[:, :, n]
            )
            std_grid[:, :, n] = (data - np.nanmin(data)) / (
                np.nanmax(data) - np.nanmin(data)
            )

        if data_format == "channels_first":
            data = (grid[n] - np.nanmean(grid[n])) / np.nanstd(grid[n])
            std_grid[n] = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    return std_grid


def undersample(image, mask=None, undersample_by=2):
    yy = np.arange(0, image.shape[0], undersample_by)
    xx = np.arange(0, image.shape[1], undersample_by)

    idx, idy = np.meshgrid(xx, yy)

    ny = idy.shape[0]
    nx = idy.shape[1]

    image = image[idy.ravel(), idx.ravel(), :]
    image = image.reshape((ny, nx, 3))

    if mask is not None:
        mask = mask[idy.ravel(), idx.ravel(), :]
        mask = mask.reshape((ny, nx, mask.shape[-1]))

    return image, mask


def upsample(image, original_image, result):
    y0 = np.arange(image.shape[0])
    x0 = np.arange(image.shape[1])

    x0, y0 = np.meshgrid(x0, y0)

    y = np.linspace(y0.min(), y0.max(), original_image.shape[0])
    x = np.linspace(x0.min(), x0.max(), original_image.shape[1])

    x, y = np.meshgrid(x, y)

    interp_result = griddata(
        (x0.ravel(), y0.ravel()),
        result[:, :, 0].ravel(),
        (x, y),
        method="nearest",
    )

    return interp_result


def return_zeroed(mask, crf_mask):
    _, _, nz = mask.shape
    for z in range(nz):
        summ = np.sum(mask[:, :, z])
        if summ == 0:
            crf_mask[:, :, z] = 0.0
    return crf_mask


augment = Sequential(
    [
        Fliplr(p=0.5),
        Flipud(p=0.5),
        # Rot90(k=(0, 1, 2, 3)),
        # Noise(scale=(0.0, 0.05)),
    ]
)
