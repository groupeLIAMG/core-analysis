# -*- coding: utf-8 -*-

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

from core_analysis.architecture import Model
from core_analysis.preprocess import unbox
from core_analysis.utils.constants import BATCH_SIZE, IMAGE_DIR, DIM
from core_analysis.utils.transform import augment
from core_analysis.utils.visualize import plot_inputs
from core_analysis.utils.processing import stored_property, saved_array_property

preprocess_input = sm.get_preprocessing(Model.BACKBONE)


class Dataset(COCO):
    CAT_IDS = [1, 2, 3]
    CAT_NAMES = ["FRACTURES", "VEINS", "REALGAR"]
    VAL_PERCENT = 0.1

    def __init__(self, label_path):
        super().__init__(label_path)
        self.imgs = {
            img_id: Image(img_id, self, info) for img_id, info in self.imgs.items()
        }
        plot_inputs(self.imgs)

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

    # def save(self):
    #     for image in self.images.values():
    #         image.save_info()

    def __iter__(self):
        batches_idx = permutation(self.all_patches)
        # TODO: Split into balanced batches.
        batches_idx = np.array_split(
            self.all_patches, len(self.all_patches) // BATCH_SIZE
        )
        for batch_idx in batches_idx:
            patches = []
            for i, j in batch_idx:
                patches.append(self.imgs[i].get_patch(j))
            patches = np.array(patches)
            patches = self.preprocess(patches)
            yield patches

    # def __getitem__(self):
    #     counts = np.unique(np.concatenate([[0, 1, 2]]), return_counts=True)[1]

    #     return [Image(id) for id in batch_idx]

    @stored_property
    def all_patches(self):
        return np.array(
            [(i, j) for i, img in self.imgs.items() for j in range(len(img.patches))]
        )

    def preprocess(patch, do_augment=True):
        patch = patch.without_background()
        patch = preprocess_input(patch)
        if do_augment:
            patch.image[:], patch.masks = augment(
                images=patch,
                heatmaps=patch.masks,
            )

        return patch


class Image:
    def __init__(self, path, dataset, info=None):
        if isinstance(path, int):
            path = self.convert_id_to_path(path, dataset)
        self.data = np.array(self._open(path))
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

    def _open(self, path):
        data = open(path)
        data = exif_transpose(data)
        return data

    @property
    def shape(self):
        return self.data.shape

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

    @stored_property
    def patches(self):
        patches = []
        for i in range(self.shape[0] - DIM[0]):
            for j in range(self.shape[1] - DIM[1]):
                if not self.background[i + DIM[0] // 2, j + DIM[1] // 2]:
                    classes = self.masks[i : i + DIM[0], j : j + DIM[1]]
                    patch = Patch(self, i, j, classes)
                    patches.append(patch)
        return patches


class View:
    def __init__(
        self, image, get_op=lambda view: view, set_op=lambda idx, value: (idx, value)
    ):
        self.image = image
        self.get_op = get_op
        self.set_op = set_op

    def __getitem__(self, idx):
        view = self.image[idx]
        view = self.get_op(view)
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
