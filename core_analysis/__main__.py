# -*- coding: utf-8 -*-

import os
from os.path import join
from argparse import ArgumentParser

os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
import numpy as np
from keras import callbacks
import segmentation_models as sm

from core_analysis.architecture import masked_loss
from core_analysis.dataset import prepare_inputs, prepare_test_inputs
from core_analysis.postprocess import predict_tiles
from core_analysis.utils.visualize import plot_loss, plot_predictions, plot_test_results
from core_analysis.utils.constants import (
    TODAY,
    BATCH_SIZE,
    BACKBONE,
    CHECKPOINT_DIR,
    LOAD_FILENAME,
    LR,
    BATCH_SIZE,
    DIM,
    N_CLASSES,
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
parser.add_argument("w", "weights-filename", type=str, default=LOAD_FILENAME)


def main(args):
    if args.weights_filename is not None:
        model = tf.keras.models.load_model(
            join(CHECKPOINT_DIR, args.weights), compile=False
        )
    else:
        model = sm.Linknet(
            BACKBONE,
            classes=N_CLASSES,
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
        X_train, Y_train, X_test, Y_test = prepare_inputs(args.do_augment)

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
        if args.plot:
            plot_loss(history)

    if args.plot:
        plot_predictions(model, X_test, Y_test, begin=600, end=610)

    if args.test:
        XX, mask = prepare_test_inputs()
        pred_tile = predict_tiles(model, merge_func=np.max, reflect=True)
        pred_tile.create_batches(XX, DIM, step=int(DIM[0]), n_classes=N_CLASSES)
        pred_tile.predict(batches_num=1500, coords_channels=False)
        results = pred_tile.merge()
        results[mask] = 0.0

        if args.plot:
            plot_test_results(XX, results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
