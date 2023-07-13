# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
import pickle as pkl

import cv2
import numpy as np
from PIL import Image, ImageOps

from core_analysis.utils.constants import BATCH_SIZE, IMAGE_DIR
from core_analysis.utils.transform import augment, undersample
from core_analysis.utils.visualize import plot_inputs
from core_analysis.preprocess import preprocess_batches, preprocess_input


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
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            image_list.append(np.array(img))

    ii = np.random.choice(len(image_list), size=1)[0]
    image, _ = undersample(image_list[ii], undersample_by=1)
    XX = np.float32(
        cv2.bilateralFilter(np.float32(image), d=5, sigmaColor=35, sigmaSpace=35)
    )
    XX = preprocess_input(XX)
    median_pixel_value = np.median(image[:100, :100])
    mask = (image == median_pixel_value)[..., 0]
    XX[mask] = 0.0

    return XX, mask
