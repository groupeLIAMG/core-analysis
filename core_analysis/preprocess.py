# -*- coding: utf-8 -*-

import cv2
import numpy as np

from core_analysis import postprocess
from core_analysis.utils.transform import undersample, upsample


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
