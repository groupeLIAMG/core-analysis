# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import tensorflow as tf

from core_analysis.dataset import Dataset
from core_analysis.architecture import Model
from core_analysis.utils.visualize import plot_loss, plot_predictions, plot_test_results
from core_analysis.utils.constants import MODEL_FILENAME, LABELS_PATH


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
    model = Model()
    dataset = Dataset(LABELS_PATH)

    if args.train:
        history = model.train(
            dataset.subset("train"), dataset.subset("val"), args.weights_filename
        )
        if args.plot:
            plot_loss(history)
            plot_predictions(model, dataset.subset("val"), begin=600, end=610)

    if args.test:
        results = model.test(dataset.subset("test"))
        if args.plot:
            plot_test_results(results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
