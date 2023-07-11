# -*- coding: utf-8 -*-

from os.path import join

import cv2
import numpy as np
from numpy.random import choice
from PIL import Image, ImageOps
from scipy.stats import mode
from tqdm.notebook import tqdm

from core_analysis import postprocess
from core_analysis.architecture import dense_crf
from core_analysis.utils.transform import undersample, upsample, return_zeroed, min_dist


def get_image(coco, image_id, cat_ids, folder=""):
    # Get all fracture annotations for a given image.

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


def generate_batches(
    image, mask, dim, patch_num, norm=True, clip_mask=False, min_dist_to_sample=0
):
    """
    image - input array data
    mask - labelled array data
    dim - 3D-dimensions
    patch_num - number of samples
    norm - if True normalize RGB images
    clip_mask - uses mask to limit central-point selection
    """

    # Select only labelled pixels.
    size = int(patch_num)
    # Create grids to store values.
    X = np.zeros((size, *dim))
    Ym = np.zeros((size, dim[0], dim[1], mask.shape[-1]))
    y = []

    pairs = []
    # Select pairs at random.
    idy, idx = np.where(mask > 0)[:2]

    # Use mask information to limit point selection.
    if clip_mask:
        ny, nx = mask.shape
        idy = idy[(idy > dim[0] // 2) & (idy < ny - dim[0] // 2)]
        idx = idx[(idx > dim[1] // 2) & (idx < nx - dim[1] // 2)]

    elems = np.arange(0, mask[mask > 0].shape[0], 1, dtype=int)

    i = 0
    iteration = 0
    while i < size:
        # Create batches.
        e = choice(elems, size=1, replace=False)
        # Create subset.
        iy, ix = int(idy[e]), int(idx[e])

        # Submask.
        msk = mask[
            iy - dim[0] // 2 : iy + dim[0] // 2, ix - dim[1] // 2 : ix + dim[1] // 2
        ]

        iteration += 1

        # Check pc and if y-position was repeated.
        if min_dist(ix, iy, pairs) >= min_dist_to_sample:
            img = image[
                iy - dim[0] // 2 : iy + dim[0] // 2,
                ix - dim[1] // 2 : ix + dim[1] // 2,
                :,
            ]
            dimm = img.shape

            if dimm == dim:
                X[i] = img
                Ym[i, :, :, :] = msk
                ny, nx, nz = msk.shape
                summ = np.sum(msk.reshape((ny * nx, nz)), 0)
                y += [np.argmax(summ)]
                pairs.append((ix, iy))
                i += 1
            else:
                pass

        if iteration > 100:
            # Force stop.
            break

    if norm:
        X /= 255.0

    return X[:i], Ym[:i], y[:i]


def preprocess_batches(X, Y, fill_with_local_mean=False, pred_model=True):
    n = 0
    for im_i, m_i in tqdm(zip(X, Y)):
        fill_mean = np.mean(mode(im_i, keepdims=True)[0])
        idy, idx, _ = np.where(im_i != fill_mean)
        local_mean = np.mean(im_i[idy, idx])
        iy, ix, _ = np.where(im_i == fill_mean)
        if fill_with_local_mean:
            im_i = np.where(im_i == fill_mean, local_mean, im_i)
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
    image = np.float32(
        cv2.bilateralFilter(np.float32(image), d=15, sigmaColor=55, sigmaSpace=35)
    )
    pred_tile = postprocess.predict_tiles(model, merge_func=np.max, reflect=True)
    pred_tile.create_batches(image, (dim[0], dim[1], 3), step=int(dim[0]), n_classes=1)
    pred_tile.predict(batches_num=batches_num, coords_channels=False)
    result = pred_tile.merge()

    interp_result = upsample(image, original_image, result)

    idy, idx = np.where(interp_result <= ths)[:2]
    original_image[idy, idx] = np.nanmean(image)

    return original_image
