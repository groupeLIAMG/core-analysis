# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import tensorflow as tf

from core_analysis.dataset import Dataset
from core_analysis.architecture import Model
from core_analysis.utils.visualize import (
    turn_plot_off,
    Figure,
    Image,
    Mask,
    Loss,
    wait_for_figures,
)
from core_analysis.utils.constants import MODEL_FILENAME, LABELS_PATH, TODAY


# Check the number of available GPUs.
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)


parser = ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-w", "--weights-filename", default=MODEL_FILENAME)
parser.add_argument("-a", "--do_augment", action="store_true")


def main(args):
    if not args.plot:
        turn_plot_off()

    model = Model(args.weights_filename)
    dataset = Dataset(LABELS_PATH)

    if args.train:
        train_subset = dataset.subset("train")
        val_subset = dataset.subset("val")

        image = next(iter(train_subset.imgs.values()))
        Figure(
            filename="image_masks",
            subplots=[
                Image(image, draw_boxes=True),
                *(Mask(image.masks[..., i]) for i in range(3)),
            ],
        )
        Figure(subplots=[Image(image=image, mask=image.masks[..., 1], draw_boxes=True)])
        tile = next(iter(train_subset))[0]
        Figure("tiles", [Image(tile), *(Mask(tile.masks[..., i]) for i in range(3))])

        # history = model.train(train_subset, val_subset)

        # Figure(f"graph_losses_{TODAY}", [Loss(history)])

    if args.test:
        results = model.test(dataset.subset("test"))

        image = next(iter(dataset.subset("test").imgs.values()))
        pred = model.predict([image])
        Figure(
            "predictions",
            [
                Image(image),
                Mask(image.masks[..., 1]),
                *(Mask(pred[..., i]) for i in range(3)),
            ],
        )
        Figure(
            "predictions_with_images",
            [Image(image.without_background(), mask=pred[..., i]) for i in range(3)],
        )

    wait_for_figures()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
