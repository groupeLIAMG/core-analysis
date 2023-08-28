# -*- coding: utf-8 -*-

"""Core analysis by deep learning.

Exploring the capabilities of *Transformer*-based neural network
"""

from argparse import ArgumentParser

import tensorflow as tf
import neptune

from core_analysis.dataset import Dataset
from core_analysis.architecture import Model
from core_analysis.utils.visualize import (
    report_figures,
    turn_plot_off,
    Figure,
    Image,
    Mask,
    Loss,
)
from core_analysis.utils.constants import (
    MODEL_FILENAME,
    LABELS_PATH,
    TODAY,
    NEPTUNE_NAME,
    NEPTUNE_PROJECT,
    NEPTUNE_API_TOKEN,
)

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
parser.add_argument("-a", "--do-augment", action="store_true")
parser.add_argument("-e", "--run-eagerly", action="store_true")


def main(args):
    run = neptune.init_run(
        name=NEPTUNE_NAME,
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_TOKEN,
    )
    report_figures(run)
    for key, value in args.__dict__.items():
        run[key] = value

    if not args.plot:
        turn_plot_off()

    model = Model(args.weights_filename, args.run_eagerly)
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
        patches, masks = next(iter(train_subset))
        Figure(
            filename="tiles",
            subplots=[Image(patches[0]), *(Mask(masks[0, ..., i]) for i in range(3))],
        )

        history = model.train(train_subset, val_subset)

        Figure(filename=f"graph_losses_{TODAY}", subplots=[Loss(history)])

    if args.test:
        results = model.test(dataset.subset("test"))

        image = next(iter(dataset.subset("test").imgs.values()))
        pred = model.predict([image])
        Figure(
            filename="predictions",
            subplots=[
                Image(image),
                Mask(image.masks[..., 1]),
                *(Mask(pred[..., i]) for i in range(3)),
            ],
        )
        Figure(
            filename="predictions_with_images",
            subplots=[
                Image(image.without_background(), mask=pred[..., i]) for i in range(3)
            ],
        )

    run.stop()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
