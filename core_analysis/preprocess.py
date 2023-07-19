# -*- coding: utf-8 -*-

from os.path import join

import cv2
import numpy as np
from keras.models import load_model
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    create_pairwise_gaussian,
    create_pairwise_bilateral,
    unary_from_softmax,
)

from core_analysis.postprocess import predict_tiles
from core_analysis.utils.transform import undersample, upsample

UNBOX_DIR = join("data", "models", "background_seg")
UNBOX_FILENAME = "resnet_unet_weights_rm_bkground_20230607.h5"
model = load_model(join(UNBOX_DIR, UNBOX_FILENAME), compile=False)


def unbox(original_image, dim, batches_num=1000, threshold=0.6):
    image, _ = undersample(original_image, mask=None, undersample_by=5)
    image = cv2.bilateralFilter(image, d=15, sigmaColor=55, sigmaSpace=35)

    pred_tile = predict_tiles(model, merge_func=np.max, reflect=True)
    pred_tile.create_batches(image, dim, step=int(dim[0]), n_classes=1)
    pred_tile.predict(batches_num=batches_num, coords_channels=False)
    result = pred_tile.merge()

    result = upsample(result, original_image.shape)
    result = result < threshold

    return result


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
