# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from core_analysis.utils.transform import adjust_rgb
from core_analysis.utils.constants import TODAY, PLOT_DIR


def plot_masks(images, masks, cat_names):
    for i in range(3):
        _, axs = plt.subplots(1, 4, figsize=(8, 4))

        axs[0].axis("off")
        axs[0].imshow(images[i], vmin=0, vmax=1)
        for j in range(3):
            axs[j + 1].imshow(masks[i, :, :, j], cmap="jet", interpolation="spline16")
            axs[j + 1].set_title(cat_names[j])
            axs[j + 1].axis("off")
        plt.savefig(
            join(PLOT_DIR, f"image_tiles_masks_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_image_and_mask(image):
    print("Image ID:", image.id)

    _, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Draw boxes and add label to each box.
    for ann in image.annotations:
        box = ann["bbox"]
        bb = patches.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
        axs[0].add_patch(bb)

    axs[0].imshow(adjust_rgb(image, 2, 98))
    axs[0].set_aspect(1)
    axs[0].axis("off")
    axs[0].set_title("Image", fontsize=12)

    axs[1].imshow(np.argmax(image.masks[..., 2], -1), cmap="Dark2")
    axs[1].set_aspect(1)
    axs[1].axis("off")
    axs[1].set_title("Masque", fontsize=12)

    plt.savefig(join(PLOT_DIR, "image_masque.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_image_with_mask(image, mask):
    plt.figure(figsize=(12, 12))
    plt.imshow(adjust_rgb(image, 2, 98))
    plt.imshow(np.where(mask > 0, 1, np.nan), cmap="viridis", alpha=0.5)
    plt.axis("scaled")
    plt.axis("off")
    plt.show()


def plot_inputs(images, qty=1):
    ids = list(images.keys())
    ids = np.random.choice(ids, qty)
    for id in ids:
        image = images[id]
        _, axs = plt.subplots(1, 4, figsize=(12, 4))
        axs[0].imshow(adjust_rgb(image, 2, 98))
        axs[0].axis("off")
        for i in range(3):
            axs[i + 1].imshow(image.masks[:, :, i])
            axs[i + 1].axis("off")
        plt.show()


def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(
        join(PLOT_DIR, f"graph_losses_{TODAY}.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_predictions(model, images, labels, begin=None, end=None):
    pred_probs = model.predict(images[begin:end])

    for n, i in enumerate(range(begin, end)):
        _, axs = plt.subplots(1, 5, figsize=(15, 6))

        axs[0].imshow(adjust_rgb(images[i], 10, 90))
        axs[0].axis("off")
        axs[1].imshow(labels[i, :, :, 1], cmap="plasma", vmin=0, vmax=1)
        axs[1].axis("off")
        for i in range(3):
            axs[i + 2].imshow(pred_probs[n, :, :, i], cmap="plasma", vmin=0, vmax=1)
            axs[i + 2].axis("off")
        plt.show()


def plot_test_results(images, results):
    y = np.arange(results.shape[0])
    x = np.arange(results.shape[1])
    x, y = np.meshgrid(x, y)

    for c in range(results.shape[-1]):
        image = image.without_background()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(adjust_rgb(images, 5, 99), zorder=0)
        ax.pcolormesh(
            x,
            y,
            np.where(results[:, :, c] > 0.9, 1.0, np.nan),
            cmap="plasma",
            vmin=0.3,
            vmax=1.0,
            alpha=0.7,
            zorder=1,
        )
        plt.xlim(70, 2300)
        plt.ylim(200, 1800)
        plt.axis("off")
        plt.savefig(join(PLOT_DIR, f"pred_{c}.png"), dpi=300, bbox_inches="tight")
        plt.show()
