# -*- coding: utf-8 -*-

from tabnanny import verbose
import cv2
import numpy as np
from tqdm import tqdm


class predict_tiles:
    def __init__(self, model, merge_func=np.mean, add_padding=False, reflect=False):
        self.model = model
        self.add_padding = add_padding
        self.reflect = reflect
        self.merge_func = merge_func

    def overlap(self, im1, im2):
        result = self.merge_func([im1, im2], 0)
        return result

    def create_batches(self, data, dim, overlap_ratio, n_classes):
        if self.add_padding:
            data = cv2.copyMakeBorder(
                data, dim[0], dim[1], dim[0], dim[1], cv2.BORDER_REFLECT
            )

        step = int((1 - overlap_ratio) * dim[0])
        (self.y_max, self.x_max, _) = data.shape
        sy = self.y_max // step
        sx = self.x_max // step
        batch = np.zeros((sy * sx, *dim))
        self.dim = dim
        self.step = step
        self.n_classes = n_classes

        if self.reflect:
            batch = np.zeros((4 * sy * sx, *dim))

        n = 0
        for y in range(dim[1] // 2, self.y_max - dim[1] // 2, self.step):
            for x in range(dim[0] // 2, self.x_max - dim[0] // 2, self.step):
                batch[n] = data[
                    y - self.dim[1] // 2 : y + self.dim[1] // 2,
                    x - self.dim[0] // 2 : x + self.dim[0] // 2,
                    :,
                ]
                n += 1

        if self.reflect:
            m = 0
            for y in range(self.y_max - self.dim[1] // 2, self.dim[1] // 2, -self.step):
                for x in range(
                    self.x_max - self.dim[0] // 2, self.dim[0] // 2, -self.step
                ):
                    batch[n + m] = data[
                        y - self.dim[1] // 2 : y + dim[1] // 2,
                        x - dim[0] // 2 : x + dim[0] // 2,
                        :,
                    ]
                    m += 1

            j = 0
            for y in range(self.dim[1] // 2, self.y_max - self.dim[1] // 2, self.step):
                for x in range(
                    self.x_max - self.dim[0] // 2, self.dim[0] // 2, -self.step
                ):
                    batch[n + m + j] = data[
                        y - self.dim[1] // 2 : y + dim[1] // 2,
                        x - dim[0] // 2 : x + dim[0] // 2,
                        :,
                    ]
                    j += 1

            k = 0
            for y in range(self.y_max - self.dim[1] // 2, self.dim[1] // 2, -self.step):
                for x in range(
                    self.dim[0] // 2, self.x_max - self.dim[0] // 2, self.step
                ):
                    batch[n + m + j + k] = data[
                        y - self.dim[1] // 2 : y + dim[1] // 2,
                        x - dim[0] // 2 : x + dim[0] // 2,
                        :,
                    ]
                    k += 1

        self.batches = batch
        self.num = n
        del batch

        if self.reflect:
            self.num = n + m + j + k

    def predict(self, batches_num, extra_channels=0, output=0, pad=3):
        results = []

        for n in tqdm(range(0, self.num, batches_num)):
            p = self.model.predict(self.batches[:batches_num], verbose=0)

            # drop border
            p[:, :pad, :] = np.nan
            p[:, -pad:, :] = np.nan
            p[:, -pad:, :] = np.nan
            p[:, :, -pad:] = np.nan
            results.append(p)
            self.batches = self.batches[batches_num:]

        self.results = np.concatenate(results)
        del self.batches
        del results

    def reconstruct(self, results):
        # Preallocate memory.
        grid = np.zeros((1, self.y_max, self.x_max, self.n_classes))

        n = 0
        for y in range(self.dim[1] // 2, self.y_max - self.dim[1] // 2, self.step):
            for x in range(self.dim[0] // 2, self.x_max - self.dim[0] // 2, self.step):
                grid[
                    :,
                    y - self.dim[1] // 2 : y + self.dim[1] // 2,
                    x - self.dim[0] // 2 : x + self.dim[0] // 2,
                ] = self.overlap(
                    grid[
                        0,
                        y - self.dim[1] // 2 : y + self.dim[1] // 2,
                        x - self.dim[0] // 2 : x + self.dim[0] // 2,
                    ],
                    results[n],
                )
                n += 1

        if self.reflect:
            m = 0
            for y in range(self.y_max - self.dim[1] // 2, self.dim[1] // 2, -self.step):
                for x in range(
                    self.x_max - self.dim[0] // 2, self.dim[0] // 2, -self.step
                ):
                    grid[
                        0,
                        y - self.dim[1] // 2 : y + self.dim[1] // 2,
                        x - self.dim[0] // 2 : x + self.dim[0] // 2,
                    ] = self.overlap(
                        grid[
                            0,
                            y - self.dim[1] // 2 : y + self.dim[1] // 2,
                            x - self.dim[0] // 2 : x + self.dim[0] // 2,
                        ],
                        results[n + m],
                    )
                    m += 1

            j = 0
            for y in range(self.dim[1] // 2, self.y_max - self.dim[1] // 2, self.step):
                for x in range(
                    self.x_max - self.dim[0] // 2, self.dim[0] // 2, -self.step
                ):
                    grid[
                        0,
                        y - self.dim[1] // 2 : y + self.dim[1] // 2,
                        x - self.dim[0] // 2 : x + self.dim[0] // 2,
                    ] = self.overlap(
                        grid[
                            0,
                            y - self.dim[1] // 2 : y + self.dim[1] // 2,
                            x - self.dim[0] // 2 : x + self.dim[0] // 2,
                        ],
                        results[n + m + j],
                    )
                    j += 1

            k = 0
            for y in range(self.y_max - self.dim[1] // 2, self.dim[1] // 2, -self.step):
                for x in range(
                    self.dim[0] // 2, self.x_max - self.dim[0] // 2, self.step
                ):
                    grid[
                        0,
                        y - self.dim[1] // 2 : y + self.dim[1] // 2,
                        x - self.dim[0] // 2 : x + self.dim[0] // 2,
                    ] = self.overlap(
                        grid[
                            0,
                            y - self.dim[1] // 2 : y + self.dim[1] // 2,
                            x - self.dim[0] // 2 : x + self.dim[0] // 2,
                        ],
                        results[n + m + j + k],
                    )
                    k += 1

        if self.add_padding:
            return grid[0, self.dim[1] : -self.dim[1], self.dim[0] : -self.dim[0], :]
        else:
            return grid[0]

    def merge(self):
        output = self.reconstruct(self.results)

        return output
