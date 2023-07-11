# -*- coding: utf-8 -*-

import os
from os.path import join
import pickle as pkl
from argparse import ArgumentParser

os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks
import segmentation_models as sm

from core_analysis.architecture import masked_loss
from core_analysis.preprocess import preprocess_batches
from core_analysis.postprocess import predict_tiles
from core_analysis.utils.transform import data_augmentation, adjust_rgb, undersample
from core_analysis.utils.visualize import plot_inputs, plot_loss, plot_predictions
from core_analysis.utils.constants import (
    TODAY,
    BATCH_SIZE,
    BACKBONE,
    IMAGE_DIR,
    CHECKPOINT_DIR,
    LOAD_FILENAME,
    LR,
    BATCH_SIZE,
    DIM,
)

# Check the number of available GPUs.
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)


parser = ArgumentParser()
parser.add_argument("train", type=bool, action="store_true")
parser.add_argument("test", type=bool, action="store_true")
parser.add_argument("plot", type=bool, action="store_true")
parser.add_argument("w", "weights-filename", type=str, default=None)


def main(args):
    if args.weights_filename is not None:
        model = tf.keras.models.load_model(
            join(CHECKPOINT_DIR, args.weights), compile=False
        )
    else:
        model = sm.Linknet(
            BACKBONE,
            classes=classes,
            activation="softmax",
            encoder_weights="imagenet",
            encoder_freeze=False,
        )

    loss = masked_loss(DIM, ths=0.5, hold_out=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss=loss.contrastive_loss,
        metrics=["acc"],
    )

    if args.train:
        # Prepare inputs.

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
        X_test, Y_test, y_test = dataset["X_test"], dataset["Y_test"], dataset["y_test"]
        classes = Y_train.shape[-1]
        print(X_train.shape)

        counts = np.unique(y_train, return_counts=True)[1]
        n_samples = np.min(counts)

        indexes = []
        for ii in range(classes):
            class_idx = np.where(y_train == ii)[0]
            indexes.append(np.random.choice(class_idx, size=n_samples, replace=False))
        indexes = np.concatenate(indexes)

        X_train, Y_train = X_train[indexes], Y_train[indexes]

        for i in range(0, X_train.shape[0], BATCH_SIZE):
            print(i)
            (
                X_train[i : i + BATCH_SIZE],
                Y_train[i : i + BATCH_SIZE],
            ) = preprocess_batches(
                X_train[i : i + BATCH_SIZE], Y_train[i : i + BATCH_SIZE]
            )

        plot_inputs(X_train, Y_train, qty=5)
        preprocess_input = sm.get_preprocessing(BACKBONE)
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)

        if args.do_augment:
            augdata = data_augmentation(X_train, Y_train)
            X_train, Y_train = augdata.rotation(nrot=[0, 2], perc=1.0)
            print(X_train.shape)

        # Train.

        checkpoint_filename = f"linket_{BACKBONE}_weights_{TODAY}.h5"
        checkpointer = callbacks.ModelCheckpoint(
            filepath=join(CHECKPOINT_DIR, checkpoint_filename),
            monitor="loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=10e-4,
            patience=50,
        )
        history = model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, Y_test),
            callbacks=[checkpointer, early_stopping],
            epochs=250,
        )
        plot_loss(history)

    plot_predictions(model, X_test, Y_test, begin=600, end=610)

    if args.test:
        image_list = []

        # Walk through all files in the folder and load images.
        for filename in os.listdir(FOLDER_PATH):
            if filename.endswith(".JPG") or filename.endswith(".jpeg"):
                # Load image and add it to the list.
                img_path = join(FOLDER_PATH, filename)
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
                image_list.append(np.array(img))

        ii = np.random.choice(len(image_list), size=1)[0]
        image, _ = undersample(image_list[ii], undersample_by=1)
        dim = X_train.shape[1:]
        XX = np.float32(
            cv2.bilateralFilter(np.float32(image), d=5, sigmaColor=35, sigmaSpace=35)
        )
        XX = preprocess_input(XX)
        median_pixel_value = np.median(image[:100, :100])
        imy, imx = np.where(image == median_pixel_value)[:2]
        XX[imy, imx] = 0.0

        pred_tile = predict_tiles(model, merge_func=np.max, reflect=True)
        pred_tile.create_batches(
            XX, (dim[0], dim[1], 3), step=int(dim[0]), n_classes=classes
        )
        pred_tile.predict(batches_num=1500, coords_channels=False)
        result = pred_tile.merge()
        result[imy, imx] = 0.0

        plot_test_results(XX, results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
