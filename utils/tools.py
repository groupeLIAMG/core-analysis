#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:28:22 2023

@author: silva
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def normalize(x, lim=255.):
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))*lim

def adjust_rgb(img, perc_init=5, perc_final=95, nchannels=3):
    
    dim = img.shape
    adjusted_img = np.zeros((dim))
    
    if dim[-1] == nchannels:
        
        for n in range(nchannels):
            channel = img[:, :, n]
            perc_i = np.nanpercentile(channel, perc_init)
            perc_f = np.nanpercentile(channel, perc_final)
            channel = np.clip(channel, perc_i, perc_f)
            channel = normalize(channel, 1.)
            adjusted_img[:, :, n] = channel
        
    else:
        raise ValueError(f'The shape should be (M, N, {nchannels}).')
        
        
    return adjusted_img

def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) \
             + np.cos(altituderad) * np.cos(slope) \
             * np.cos(azimuthrad - aspect)
    hillshade_array = 255 * (shaded + 1) / 2

    return hillshade_array

def standardize_data(grid, data_format='channels_last'):
    
    dim = grid.shape
    std_grid = np.zeros(dim)
    
    
    for n in range(np.min(dim)):
        
        if data_format == 'channels_last':
             data = (grid[:, :, n]-np.nanmean(grid[:, :, n]))/np.nanstd(grid[:, :, n])
             std_grid[:, :, n] = (data - np.nanmin(data))/(np.nanmax(data) - np.nanmin(data))
             
        if data_format == 'channels_first':
            data = (grid[n]-np.nanmean(grid[n]))/np.nanstd(grid[n])
            std_grid[n] = (data - np.nanmin(data))/(np.nanmax(data) - np.nanmin(data))
            
    return std_grid


def undersample(image, mask, undersample_by):
    yy = np.arange(0, image.shape[0], undersample_by)
    xx = np.arange(0, image.shape[1], undersample_by)

    idx, idy = np.meshgrid(xx, yy)

    ny = idy.shape[0]
    nx = idy.shape[1]

    resampled_image = image[idy.ravel(), idx.ravel(), :].reshape((ny, nx, 3))
    #resampled_mask = mask[idy.ravel(), idx.ravel(), :].reshape((ny, nx, mask.shape[-1]))
    
    return resampled_image

class data_augmentation:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        

    def rotation(self, nrot=[0, 1, 2, 3], perc=1.):

        from numpy import rot90    

        Xaug = []
        Yaug = []
        
        for n in nrot:
            Xaug.append(np.rot90(self.X, n, axes=(1, 2)))
            Yaug.append(np.rot90(self.Y, n, axes=(1, 2)))            
        
        n_generated_samples = int(self.X.shape[0] + perc*self.X.shape[0])
        Xaug = np.concatenate(Xaug)[:n_generated_samples]
        Yaug = np.concatenate(Yaug)[:n_generated_samples]
        size = Xaug.shape[0]
        
        shuffle = np.random.choice(np.arange(0, size, 1, dtype=np.int16), size=size, replace=False)                
                
        self.X = Xaug[shuffle]
        self.Y = Yaug[shuffle]

        return self.X, self.Y    
    
    
    
    def noise(self, var=0.05):

        self.X = self.X + np.random.normal(np.mean(self.X), var, size=self.X.shape)

        return self.X    