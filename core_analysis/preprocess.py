# -*- coding: utf-8 -*-

from os.path import join

import cv2
import numpy as np
from tensorflow.keras.saving import load_model

from core_analysis.postprocess import predict_tiles
from core_analysis.utils.transform import undersample, upsample
from core_analysis.utils.constants import DIM

UNBOX_DIR = join("data", "models", "background_seg")
UNBOX_FILENAME = "resnet_unet_weights_rm_bkground_20230607.h5"
model = load_model(join(UNBOX_DIR, UNBOX_FILENAME), compile=False)


def unbox(original_image, batches_num=1000, threshold=0.6):
    image, _ = undersample(original_image, mask=None, undersample_by=5)
    image = cv2.bilateralFilter(image, d=15, sigmaColor=55, sigmaSpace=35)

    pred_tile = predict_tiles(model, merge_func=np.max, reflect=True)
    pred_tile.create_batches(image, DIM, step=int(DIM[0]), n_classes=1)
    pred_tile.predict(batches_num=batches_num, coords_channels=False)
    result = pred_tile.merge()

    result = upsample(result, original_image.shape)
    result = result < threshold

    return result
