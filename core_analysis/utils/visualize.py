# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from core_analysis.preprocess import get_image
from core_analysis.utils.transform import adjust_rgb
from core_analysis.utils.constants import TODAY

IMAGE_FOLDER = "images"


def plot_masks(images, masks, cat_names):
    for i in range(3):
        fig, axs = plt.subplots(1, 4, figsize=(8, 4))

        axs[0].axis("off")
        axs[0].imshow(images[i], vmin=0, vmax=1)
        for j in range(3):
            axs[j + 1].imshow(masks[i, :, :, j], cmap="jet", interpolation="spline16")
            axs[j + 1].set_title(cat_names[j])
            axs[j + 1].axis("off")
        plt.savefig(
            join("data", "plots", f"image_tiles_masks_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_image_and_mask(coco, cat_ids, image_ids):
    img_id = np.random.choice(image_ids, size=1)[0]
    image, mask, anns = get_image(coco, img_id, cat_ids=cat_ids, folder=IMAGE_FOLDER)
    print("Image ID:", img_id)

    _, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Draw boxes and add label to each box.
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
        axs[0].add_patch(bb)

    axs[0].imshow(adjust_rgb(image, 2, 98))
    axs[0].set_aspect(1)
    axs[0].axis("off")
    axs[0].set_title("Image", fontsize=12)

    axs[1].imshow(np.argmax(mask, -1), cmap="Dark2")
    axs[1].set_aspect(1)
    axs[1].axis("off")
    axs[1].set_title("Masque", fontsize=12)

    plt.savefig(join("data", "plots", "image_masque.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return image, mask


def plot_image_with_mask(image, mask):
    plt.figure(figsize=(12, 12))
    plt.imshow(adjust_rgb(image, 2, 98))
    plt.imshow(np.where(mask > 0, 1, np.nan), cmap="viridis", alpha=0.5)
    plt.axis("scaled")
    plt.axis("off")
    plt.show()


def plot_inputs(images, masks, qty=1):
    for _ in range(qty):
        _, axs = plt.subplots(1, 4, figsize=(12, 4))
        ii = np.random.choice(np.arange(0, images.shape[0], 1, dtype=int))
        axs[0].imshow(adjust_rgb(images[ii], 2, 98))
        axs[0].axis("off")
        for i in range(3):
            axs[i + 1].imshow(masks[ii, :, :, i])
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
        join("data", "plots", f"graph_losses_{TODAY}.png"), dpi=300, bbox_inches="tight"
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
        plt.savefig(
            join("data", "plots", f"pred_{c}.png"), dpi=300, bbox_inches="tight"
        )
        plt.show()
