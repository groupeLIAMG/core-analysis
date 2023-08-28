# -*- coding: utf-8 -*-

"""Load and manage data."""

import os
from os.path import join, split
from copy import copy
from json import dump

os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
from numpy.random import choice, permutation, seed
from PIL.Image import open
from PIL.ImageOps import exif_transpose
import segmentation_models as sm
from pycocotools.coco import COCO
from scipy.ndimage import maximum_filter

from core_analysis.architecture import Model
from core_analysis.preprocess import unbox
from core_analysis.utils.constants import IMAGE_DIR, DIM
from core_analysis.utils.transform import augment
from core_analysis.utils.processing import stored_property, saved_array_property

preprocess_input = sm.get_preprocessing(Model.BACKBONE)


class Dataset(COCO):
    CAT_IDS = [1, 2, 3]
    CAT_NAMES = ["FRACTURES", "VEINS", "REALGAR"]
    VAL_PERCENT = 0.1
    DO_AUGMENT = True
    BATCH_SIZE = 16
    N_PATCHES = 8600

    def __init__(self, label_path):
        super().__init__(label_path)
        self.imgs = {
            img_id: Image(img_id, self, info) for img_id, info in self.imgs.items()
        }

    def subset(self, mode):
        subset = copy(self)

        if mode in ["train", "val"]:
            subset_ids = [image.id for image in subset.imgs.values() if image.is_train]
            val_size = int(len(subset_ids) * self.VAL_PERCENT)
            seed(0)
            val_ids = choice(subset_ids, val_size)
            if mode == "train":
                subset_ids = [img_id for img_id in subset_ids if img_id not in val_ids]
            else:
                subset_ids = val_ids
        elif mode == "test":
            subset_ids = {
                image.id for image in subset.imgs.values() if not image.is_train
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        subset.imgs = {img_id: subset.imgs[img_id] for img_id in subset_ids}
        return subset

    def __iter__(self):
        batches_idx = permutation(self.all_patches)
        # TODO: Split into balanced batches.
        n_batches = len(batches_idx) // self.BATCH_SIZE
        batches_idx = batches_idx[: n_batches * self.BATCH_SIZE]
        batches_idx = np.split(batches_idx, n_batches)
        batches_idx = np.array(batches_idx)
        for batch_idx in batches_idx:
            patches = np.empty([len(batch_idx), *DIM], dtype=np.float32)
            masks = np.empty(
                [len(batch_idx), *DIM[:2], len(self.CAT_IDS)], dtype=np.float32
            )
            for i, (img_id, patch_id) in enumerate(batch_idx):
                patch = self.imgs[img_id].get_patch(patch_id)
                patch, mask = self.preprocess(patch)
                patches[i] = patch.astype(np.float32)
                masks[i] = mask.astype(np.float32)
            if self.DO_AUGMENT:
                patches, masks = augment(
                    images=patches,
                    heatmaps=masks,
                )
            yield patches, masks

    @stored_property
    def all_patches(self):
        return np.array(
            [(i, j) for i, img in self.imgs.items() for j in range(len(img.patches_ij))]
        )

    def preprocess(self, patch):
        patch = patch.without_background()
        data = patch.data[:]
        masks = patch.masks[:]
        data = preprocess_input(data)
        return data, masks


class Image:
    def __init__(self, path, dataset, info=None):
        if isinstance(path, int):
            path = self.convert_id_to_path(path, dataset)
        self.dir, self.filename = split(path)
        self.path = path
        self.dataset = dataset
        self.info = info

    @classmethod
    def open(cls, path, dataset, info=None):
        return cls(path, dataset, info)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.info[idx]
        else:
            return self.data[idx]

    def __repr__(self):
        return str(self.info)

    @saved_array_property
    def data(self):
        data = open(self.path)
        data = exif_transpose(data)
        return np.array(data)

    @property
    def shape(self):
        return self["height"], self["width"], 3

    def convert_id_to_path(self, id, dataset=None):
        if dataset is None:
            dataset = self.dataset
        file_name = dataset.imgs[id]["file_name"]
        subfolder = file_name.split(" ")[0]
        path = join(IMAGE_DIR, subfolder, file_name)
        return path

    @stored_property
    def id(self):
        filenames = [img.filename for img in self.dataset.imgs.values()]
        file_idx = filenames.index(self.filename)
        return list(self.dataset.imgs.keys())[file_idx]

    @stored_property
    def is_train(self):
        if self.dataset.getAnnIds(imgIds=self.id, catIds=self.dataset.CAT_IDS):
            return True
        else:
            return False

    @saved_array_property
    def background(self):
        return unbox(self)

    @saved_array_property
    def masks(self):
        masks, _ = self.get_annotations()
        return masks

    def get_annotations(self):
        masks = np.zeros([*self.shape[:2], len(self.dataset.CAT_IDS)], dtype=bool)
        annotations = []
        for i, cid in enumerate(self.dataset.CAT_IDS):
            annotation_ids = self.dataset.getAnnIds(imgIds=self.id, catIds=cid)
            annotations = self.dataset.loadAnns(annotation_ids)
            for annotation in annotations:
                mask = self.dataset.annToMask(annotation)
                mask = mask.astype(bool)
                masks[mask, i] = True
            annotations += annotations

        return masks, annotations

    def without_background(self):
        # TODO: Apply to masks as well.
        return View(self, get_op=lambda view: np.where(view.background, 0, view))

    @saved_array_property
    def patches_ij(self):
        patches_ij = np.meshgrid(
            np.arange(self.shape[0] - DIM[0]),
            np.arange(self.shape[1] - DIM[1]),
            indexing="ij",
        )
        patches_ij = np.moveaxis(patches_ij, 0, -1)

        has_label = self.masks.any(axis=-1)
        has_label = maximum_filter(has_label, size=DIM[:2], output=bool)
        is_valid = ~self.background & has_label
        is_valid = is_valid[
            DIM[0] // 2 : -DIM[0] // 2 - DIM[0] % 2,
            DIM[1] // 2 : -DIM[1] // 2 - DIM[0] % 2,
        ]
        patches_ij = patches_ij[is_valid]

        return patches_ij

    def get_patch(self, idx):
        i, j = self.patches_ij[idx]
        classes = self.masks[i : i + DIM[0], j : j + DIM[1]]
        return Patch(self, i, j, classes)


class View:
    def __init__(
        self, image, get_op=lambda view: view, set_op=lambda idx, value: (idx, value)
    ):
        self.image = image
        self.get_op = get_op
        self.set_op = set_op

    def __getitem__(self, idx):
        view = self.get_op(self.image)
        view = view[idx]
        return view

    def __setitem__(self, idx, value):
        idx, value = self.set_op(idx, value)
        self.image[idx] = value

    def __setattr__(self, name, value):
        if name == "image" and hasattr(self, "image"):
            raise AttributeError(
                "Cannot change reference image implicitly. Use `view[:] = value` instead."
            )
        else:
            super().__setattr__(name, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def without_background(self):
        return View(
            self,
            get_op=lambda view: np.where(
                self.background[..., None], [0] * DIM[-1], view
            ),
        )

    @property
    def background(self):
        background = self.image.background
        background = self.get_op(background)
        return background

    @property
    def masks(self):
        masks = self.image.masks
        masks = self.get_op(masks)
        return masks

    @property
    def data(self):
        data = self.image.data
        data = self.get_op(data)
        return data


class Patch(View):
    def __init__(self, image, i, j, classes):
        self.classes = classes
        self.i, self.j = i, j
        di, dj, _ = DIM
        super().__init__(
            image,
            get_op=lambda view: view[i : i + di, j : j + dj],
            set_op=lambda idx, value: ((idx[0] + i, idx[1] + j), value),
        )
