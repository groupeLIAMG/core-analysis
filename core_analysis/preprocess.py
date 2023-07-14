# -*- coding: utf-8 -*-

from os.path import join
from sys import stdout
import cv2
import numpy as np
from numpy.random import choice
from PIL import Image, ImageOps
from scipy.stats import mode
from tqdm.notebook import tqdm

from core_analysis import postprocess
from core_analysis.architecture import dense_crf
from core_analysis.utils.tools import undersample, upsample, return_zeroed, min_dist


def get_path(data, img_id):
    for dict_ in data["images"]:
        if dict_["id"] == img_id:
            file_name = dict_["file_name"]

    return file_name


def get_image(coco, image_id, cat_ids, folder=""):
    # Get all annotations for a given image.

    mask_grid = []
    annotations = []
    for cid in cat_ids:
        annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=cid)
        anns_ = coco.loadAnns(annotation_ids)
        file_name = coco.imgs[image_id]["file_name"]
        subfolder = file_name.split(" ")[0]
        image = Image.open(join(folder, subfolder, file_name))
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        ny, nx = image.shape[:2]

        if anns_:
            mask = np.zeros((ny, nx))
            for i in range(len(anns_)):
                mask += coco.annToMask(anns_[i])

            mask_grid.append(mask > 0)
        else:
            mask_grid.append(np.zeros((ny, nx)))

        annotations += anns_

    if mask_grid:
        mask_grid = np.stack(mask_grid, -1)
    else:
        mask_grid = np.zeros((ny, nx, 3))

    return image, mask_grid, annotations


def preprocess_batches(X, Y, fill_with_local_mean=False, pred_model=True):
    
    n = 0
    for im_i, m_i in tqdm(zip(X, Y)):
        fill_mean = np.mean(mode(im_i, keepdims=True)[0])
        idy, idx = np.where(im_i != fill_mean)[:2]
        local_mean = np.mean(im_i[idy, idx])
        iy, ix, _ = np.where(im_i == fill_mean)
        # fill background with the local mean
        if fill_with_local_mean:
            im_i = np.where(im_i == fill_mean, local_mean, im_i)
        # fill the background with zeros
        else:
            im_i = np.where(im_i == fill_mean, 0.0, im_i)

        m_i[iy, ix] = 0.0

        bilat_img = np.float32(
            cv2.bilateralFilter(np.float32(im_i), d=3, sigmaColor=15, sigmaSpace=25)
        )
        if np.isnan(bilat_img).any():
            bilat_img = np.nan_to_num(bilat_img, nan=np.nanmean(bilat_img))

        crf_mask = dense_crf(im_i, m_i, gw=5, bw=7, n_iterations=1)
        crf_mask[iy, ix] = 0.0
        X[n] = bilat_img
        Y[n] = return_zeroed(m_i, crf_mask)
        n += 1

    return X, Y


def unbox(model, original_image, dim, batches_num=1000, ths=0.6):
    image, _ = undersample(original_image, mask=None, undersample_by=5)
    image = cv2.bilateralFilter(np.float32(image), d=15, sigmaColor=55, sigmaSpace=35)
    pred_tile = postprocess.predict_tiles(model, merge_func=np.max, reflect=True)
    pred_tile.create_batches(image, (dim[0], dim[1], 3), step=int(dim[0]), n_classes=1)
    pred_tile.predict(batches_num=batches_num, coords_channels=False)
    result = pred_tile.merge()

    interp_result = upsample(image, original_image, result)

    idy, idx = np.where(interp_result <= ths)[:2]
    original_image[idy, idx] = np.nanmean(image)

    return original_image

class Gen_datasets:
    def __init__(self, coco_data, image_ids, use_ids, dim, n_samples, undersample_by=[1, 2]):
        self.coco_data = coco_data
        self.image_ids = image_ids
        self.use_ids = use_ids
        self.dim = dim
        self.n_samples = n_samples
        self.undersample_by = undersample_by
        self.X = []
        self.masks = []
        self.y = []
        
    def patchify(self, image, mask, dim, patch_num, norm=True, min_dist_to_sample=0, max_it=1e4):
        
        '''
        image: input image
        mask: input mask
        dim: tuple of dimensions
        patch_num: number of patches
        norm: if True, normalize
        clip_mask - uses mask to limit central-point selection
        perc - minimal percentage of pixels with class == 1.
        '''

        # select only labelled pixels
        size = int(patch_num) 
        # create grids to store values
        X = np.zeros((size, *dim))
        Ym = np.zeros((size, dim[0], dim[1], mask.shape[-1]))
        y = []
        
        # count images
        i = 0

        pairs = []
        # select pairs at random
        idy, idx = np.where(mask > 0)[:2]
        elems = np.arange(0, mask[mask > 0].shape[0], 1, dtype=int)

        count = 0 
        iteration = 0
        while count < size:

            # create batches
            e = np.random.choice(elems, size=1, replace=False)
            # create subset
            iy, ix = int(idy[e]), int(idx[e])
            # submask
            msk = mask[iy-dim[0]//2:iy+dim[0]//2, ix-dim[1]//2:ix+dim[1]//2]

            iteration += 1
            # check pc and if y-position was repeated
            if min_dist(ix, iy, pairs) >= min_dist_to_sample:
                img = image[iy-dim[0]//2:iy+dim[0]//2, ix-dim[1]//2:ix+dim[1]//2, :]
                dimm = img.shape

                if(dimm == dim):
                    X[i] = img
                    Ym[i, :, :, :] = msk
                    ny, nx, nz = msk.shape
                    summ = np.sum(msk.reshape((ny*nx, nz)), 0)
                    y += [np.argmax(summ)] 
                    pairs.append((ix, iy))
                    count += 1
                    i += 1

            # to avoid infinity loop
            if iteration > max_it:
                break

        if norm:
            X/=255.

        return X[:i], Ym[:i], y[:i]
    
    
    def generate_batches(self, n_classes=3):
        
        # counter
        self.image_ids = self.image_ids*10 # extend the number of times an image will be opened
        counts = np.unique(np.arange(n_classes), return_counts=True)[1]
        
        iteration = 0
        while (counts.min() < self.n_samples):
    
            m = np.min(counts)
            stdout.write(f"\r iteration: {iteration} / img-id {self.image_ids[iteration]} /{m*100/self.n_samples:.2f}%")

             # ==================  generate and store batches =================
            us = np.random.choice(self.undersample_by)
            image, mask, anns = get_image(self.coco_data, self.image_ids[iteration], self.use_ids)
            image, mask = undersample(image, mask, undersample_by=us)

            Xi, mi, yi  = self.patchify(image, mask, self.dim, patch_num=len(anns), norm=False, min_dist_to_sample=self.dim[0]//10)
            # append
            self.X.append(Xi)
            self.masks.append(mi)
            self.y.append(yi)
            counts = np.unique(np.concatenate(self.y), return_counts=True)[1]
            iteration += 1
            
        # concat data    
        X = np.concatenate(self.X, axis=0)
        m = np.concatenate(self.masks, axis=0)
        y = np.concatenate(self.y)
        output = (X, m, y)
        
        return output
    
