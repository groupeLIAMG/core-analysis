# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
import pickle as pkl

import cv2
import numpy as np
from numpy.random import choice
import PIL
from scipy.stats import mode
from tqdm.notebook import tqdm
import segmentation_models as sm

from core_analysis.architecture import Model, dense_crf
from core_analysis.utils.constants import BATCH_SIZE, IMAGE_DIR
from core_analysis.utils.transform import augment, undersample, return_zeroed, min_dist
from core_analysis.utils.visualize import plot_inputs
from core_analysis.preprocess import preprocess_batches, preprocess_input

preprocess_input = sm.get_preprocessing(Model.BACKBONE)


def get_image(coco, image_id, cat_ids, folder=""):
    # Get all fracture annotations for a given image.

    mask_grid = []
    annotations = []
    for cid in cat_ids:
        annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=cid)
        anns_ = coco.loadAnns(annotation_ids)
        file_name = coco.imgs[image_id]["file_name"]
        subfolder = file_name.split(" ")[0]
        image = PIL.Image.open(join(folder, subfolder, file_name))
        image = PIL.ImageOps.exif_transpose(image)
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


def prepare_inputs(do_augment=False, do_plot=False):
    with open(
        join("data", "dataset", "dataset_forages_128x128_20230705.pickle"),
        "rb",
    ) as f:
        dataset = pkl.load(f)

    X_train, Y_train, y_train = (
        dataset["X_train"],
        dataset["Y_train"],
        dataset["y_train"],
    )
    X_test, Y_test, _ = dataset["X_test"], dataset["Y_test"], dataset["y_test"]
    n_classes = Y_train.shape[-1]

    counts = np.unique(y_train, return_counts=True)[1]
    n_samples = np.min(counts)

    indexes = []
    for i in range(n_classes):
        class_idx = np.where(y_train == i)[0]
        indexes.append(np.random.choice(class_idx, size=n_samples, replace=False))
    indexes = np.concatenate(indexes)
    np.random.shuffle(indexes)

    X_train, Y_train = X_train[indexes], Y_train[indexes]

    for i in range(0, X_train.shape[0], BATCH_SIZE):
        (
            X_train[i : i + BATCH_SIZE],
            Y_train[i : i + BATCH_SIZE],
        ) = preprocess_batches(X_train[i : i + BATCH_SIZE], Y_train[i : i + BATCH_SIZE])

    if do_plot:
        plot_inputs(X_train, Y_train, qty=5)

    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    if do_augment:
        X_train, Y_train = augment(
            images=X_train,
            heatmaps=Y_train.astype(np.float32),
        )

    return X_train, Y_train, X_test, Y_test


def prepare_test_inputs():
    image_list = []

    # Walk through all files in the folder and load images.
    for filename in listdir(IMAGE_DIR):
        if filename.endswith(".JPG") or filename.endswith(".jpeg"):
            # Load image and add it to the list.
            img_path = join(IMAGE_DIR, filename)
            img = PIL.Image.open(img_path)
            img = PIL.ImageOps.exif_transpose(img)
            image_list.append(np.array(img))

    ii = np.random.choice(len(image_list), size=1)[0]
    image, _ = undersample(image_list[ii], undersample_by=1)
    XX = np.float32(
        cv2.bilateralFilter(np.float32(image), d=5, sigmaColor=35, sigmaSpace=35)
    )
    XX = preprocess_input(XX)
    mask = (image == 0).all(axis=-1)
    XX[mask] = 0.0

    return XX, mask
