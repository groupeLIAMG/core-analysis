# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from core_analysis.utils.transform import adjust_rgb
from core_analysis.utils.constants import TODAY, PLOT_DIR


class Figure:
    DIR = PLOT_DIR
    SAVE_DPI = 200
    SHOW_DPI = 200

    def __init__(self, *, filename=None, subplots):
        self.filename = f"{filename}.pdf" if filename is not None else None
        self.subplots = np.array(subplots)
        if self.subplots.ndim < 2:
            self.subplots = self.subplots.reshape(1, -1)
        self.nrows, self.ncols = self.subplots.shape
        self.fig, self.axs = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            squeeze=False,
            dpi=self.SHOW_DPI,
        )
        self.plot()
        self.show()
        if self.filename is not None:
            self.save()

    @property
    def filepath(self):
        return join(self.DIR, self.filename)

    def generate(self, gpus):
        with self.Metadata(gpus) as data:
            self.plot(data)

    def save(self, show=True):
        self.fig.savefig(self.filepath, transparent=True, dpi=self.SAVE_DPI)

    def plot(self):
        for subplot, ax in zip(self.subplots.flatten(), self.axs.flatten()):
            subplot.plot(ax)

    def format(self):
        for subplot, ax in zip(self.subplots.flatten(), self.axs.flatten()):
            subplot.format(ax)

    def show(self):
        plt.show()


class Subplot:
    def __init__(self, *items):
        self.items = items

    def plot(self, ax):
        raise NotImplementedError

    def format(self, ax):
        pass


class Image(Subplot):
    def __init__(self, image, mask=None, adjust_rgb=True, draw_boxes=False):
        self.image = image
        self.mask = mask
        self.do_adjust_rgb = adjust_rgb
        self.do_draw_boxes = draw_boxes

    def plot(self, ax):
        image = self.image.data
        if self.do_adjust_rgb:
            image = adjust_rgb(image, 2, 98)
        ax.imshow(image, vmin=0, vmax=1)
        if self.mask is not None:
            ax.imshow(np.where(self.mask > 0, 1, np.nan), cmap="viridis", alpha=0.5)
        if self.do_draw_boxes:
            self.draw_boxes()

    def format(self):
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

    def draw_boxes(self):
        _, anns = self.image.get_annotations()
        for ann in anns:
            box = ann["bbox"]
            bb = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
            self.ax.add_patch(bb)


class Mask(Subplot):
    def __init__(self, mask):
        self.mask = mask

    def plot(self, ax):
        ax.imshow(self.mask, cmap="jet", interpolation="spline16")

    def format(self):
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")
        self.ax.set_title(self.cat_name)


class Loss(Subplot):
    def __init__(self, history):
        self.history = history

    def plot(self, ax):
        ax.plot(self.history.history["loss"])
        ax.plot(self.history.history["val_loss"])
        ax.set_title("Loss")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        ax.legend(["train", "test"], loc="upper left")

    def format(self):
        pass


def turn_plot_off():
    Figure.show = lambda *args, **kwargs: None
    Figure.plot = lambda *args, **kwargs: None
