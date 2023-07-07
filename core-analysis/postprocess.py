#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:06:54 2023

@author: silva
"""

import cv2
import numpy as np

class predict_tiles:
    
    def __init__(self, model, merge_func=np.mean, add_padding=False, reflect=False):
        
        self.model = model
        self.add_padding = add_padding
        self.reflect = reflect
        self.merge_func = merge_func
        
        
    def create_batches(self, data, dim, step, n_classes):
        
        if self.add_padding:
            data = cv2.copyMakeBorder(data, dim[0], dim[1], dim[0], dim[1], cv2.BORDER_REFLECT)
            
        (self.y_max, self.x_max, _) = data.shape
        sy = self.y_max//step; sx = self.x_max//step
        batch             = np.zeros((sy*sx, *dim))
        self.dim          = dim
        self.step         = step
        self.n_classes    = n_classes
        
        if self.reflect:
            batch  = np.zeros((4*sy*sx, *dim))            

        n = 0
        for y in range(dim[1]//2, self.y_max-dim[1]//2, self.step):
            for x in range(dim[0]//2, self.x_max-dim[0]//2, self.step):
                batch[n] = data[y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2, :]
                n += 1
                
        if self.reflect:
            m = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2, -self.step):
                    batch[n+m] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    m += 1 
                
            j = 0
            for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2, -self.step):
                    batch[n+m+j] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    j += 1 
                    
                    
            k = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                    batch[n+m+j+k] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    k += 1 
                    
        self.batches = batch
        self.num = n
        del batch
        
        if self.reflect:
            self.num = n+m+j+k
        
               
        
    def predict(self, batches_num, coords_channels=False):
        
        results = []
        
        for n in range(0, self.num, batches_num):
            
            if coords_channels:
                p = self.model.predict([self.batches[:batches_num, :, :, :coords_channels], 
                                    self.batches[:batches_num, :, :, coords_channels:]])
                
            else: 
                p = self.model.predict(self.batches[:batches_num])
                

            results.append(p)
            self.batches = self.batches[batches_num:]
            
        self.results = np.concatenate(results)
        del self.batches
        del results
          
        
    def merge(self):
        
        # reserve memory
        grid = np.zeros((1, self.y_max, self.x_max, self.n_classes))

        n = 0
        for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
            for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                grid[:, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                self.merge_func((grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                self.results[n]), axis=0)
                n += 1     
        
        if self.reflect:
            m = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.merge_func((grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    self.results[n+m]), axis=0)
                    m += 1 
                        
            j = 0
            for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.merge_func((grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    self.results[n+m+j]), axis=0)
                    j += 1 
                        
                        
            k = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.merge_func((grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    self.results[n+m+j+k]), axis=0)
                    k += 1 
                    
        if self.add_padding:
            return grid[0, self.dim[1]:-self.dim[1], self.dim[0]:-self.dim[0], :]
            
        else: return grid[0]
        
        