from os.path import join

import numpy as np
import tensorflow as tf
from keras import callbacks
from keras import backend as K
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    create_pairwise_gaussian,
    create_pairwise_bilateral,
    unary_from_softmax,
)
import segmentation_models as sm

from core_analysis.postprocess import predict_tiles
from core_analysis.utils.constants import (
    CHECKPOINT_DIR,
    DIM,
    N_CLASSES,
    LR,
    TODAY,
)


class Model:
    BACKBONE = "efficientnetb7"
    BATCH_SIZE = 16

    def __init__(self, weights_filename=None, args=None):
        if weights_filename is not None:
            self.model = tf.keras.models.load_model(
                join(CHECKPOINT_DIR, args.weights),
                compile=False,
            )
        else:
            self.model = sm.Linknet(
                self.BACKBONE,
                classes=N_CLASSES,
                activation="softmax",
                encoder_weights="imagenet",
                encoder_freeze=False,
            )

        loss = masked_loss(DIM, ths=0.5, hold_out=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.model.compile(
            optimizer=optimizer,
            loss=loss.contrastive_loss,
            metrics=["acc"],
        )

    def train(self, X_train, Y_train, X_test, Y_test):
        checkpoint_filename = f"linknet_{self.BACKBONE}_weights_{TODAY}.h5"
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
        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=self.BATCH_SIZE,
            validation_data=(X_test, Y_test),
            callbacks=[checkpointer, early_stopping],
            epochs=250,
        )
        return history

    def test(self, images):
        pred_tile = predict_tiles(self.model, merge_func=np.max, reflect=True)
        pred_tile.create_batches(images, DIM, step=int(DIM[0]), n_classes=N_CLASSES)
        pred_tile.predict(batches_num=1500, coords_channels=False)
        results = pred_tile.merge()
        return results


def dense_crf(image, final_probabilities, gw=11, bw=3, n_iterations=5):
    """
    gw - pairwise gaussian window size: enforces more spatially consistent segmentations.
    bw - pairwise bilateral window size: uses local color features to refine predictions.
    """

    ny = image.shape[0]
    nx = image.shape[1]
    n_classes = final_probabilities.shape[-1]
    softmax = final_probabilities.squeeze()
    softmax = softmax.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values.
    # Look up the definition of the unary_from_softmax for more information.
    unary = unary_from_softmax(softmax, scale=None, clip=1e-5)

    # The inputs should be C-continious -- we are using Cython wrapper.
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(ny * nx, n_classes)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforce more spatially consistent segmentations.
    feats = create_pairwise_gaussian(sdims=(gw, gw), shape=(ny, nx))

    d.addPairwiseEnergy(
        feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them.
    feats = create_pairwise_bilateral(
        sdims=(bw, bw), schan=(7, 7, 7), img=image, chdim=2
    )

    d.addPairwiseEnergy(
        feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    Q = d.inference(n_iterations)
    probs = np.array(Q, dtype=np.float32).reshape((n_classes, ny, nx))
    probs = np.around(probs, 4)
    # res = np.argmax(Q, axis=0).reshape((ny, nx))

    return probs.swapaxes(1, 0).swapaxes(1, 2)


class masked_loss:
    def __init__(self, dim, ths, wmatrix=0, use_weights=False, hold_out=0.1):
        self.dim = dim
        self.ths = ths
        self.wmatrix = wmatrix
        self.use_weights = use_weights
        self.hold_out = hold_out

    def masked_rmse(self, y_true, y_pred):
        # Distance between the predictions and simulation probabilities.
        squared_diff = K.square(y_true - y_pred)

        # Give different weights by class.
        if self.use_weights:
            squared_diff *= self.wmatrix

        # Calculate the loss only where there are samples.
        mask = tf.where(y_true >= self.ths, 1.0, 0.0)

        # Take some of the training points out at random.
        if self.hold_out > 0:
            mask *= tf.where(
                tf.random.uniform(
                    shape=(1, *squared_diff.shape[1:]), minval=0.0, maxval=1.0
                )
                > self.hold_out,
                1.0,
                0.0,
            )

        denominator = K.sum(mask)  # Number of pixels.
        if self.use_weights:
            denominator = K.sum(mask * self.wmatrix)

        # Sum of squared differences at sampled locations,
        summ = K.sum(squared_diff * mask)
        # Compute error,
        rmse = K.sqrt(summ / denominator)

        return rmse

    def dice_loss(self, y_true, y_pred):
        # Dice coefficient loss.
        y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        return dice_loss

    def contrastive_loss(self, y_true, y_pred):
        rmse = self.masked_rmse(y_true, y_pred)
        dice_loss = self.dice_loss(y_true, y_pred)

        loss = rmse + dice_loss

        return loss
