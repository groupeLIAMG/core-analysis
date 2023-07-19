# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser

os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf

from core_analysis.architecture import Model
from core_analysis.dataset import prepare_inputs, prepare_test_inputs
from core_analysis.utils.visualize import plot_loss, plot_predictions, plot_test_results
from core_analysis.utils.constants import LOAD_FILENAME, LABELS_PATH


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
parser.add_argument("-w", "--weights-filename", default=LOAD_FILENAME)
parser.add_argument("-a", "--do_augment", action="store_true")


def main(args):
    model = Model()

    if args.train:
        dataset = Dataset(LABELS_PATH)
        X_train, Y_train, X_test, Y_test = prepare_inputs(args.do_augment, args.do_plot)
        history = model.train(X_train, Y_train, X_test, Y_test)
        if args.plot:
            plot_loss(history)
            plot_predictions(model, X_test, Y_test, begin=600, end=610)

    if args.test:
        images, mask = prepare_test_inputs()
        results = model.test(images)
        results[mask] = 0.0
        if args.plot:
            plot_test_results(images, results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
