# -*- coding: utf-8 -*-

from os.path import join
from datetime import date

TODAY = str(date.today()).replace("-", "_")
BATCH_SIZE = 500
IMAGE_DIR = join("data", "images")
PLOT_DIR = join("data", "plots")
CHECKPOINT_DIR = join("data", "models", "save_models")
MODEL_FILENAME = "linknet_efficientnetb7_weights_2023_07_05.h5"
LABELS_FILENAME = "labels_20230714.json"
LABELS_DIR = join("data", "labels")
LABELS_PATH = join(LABELS_DIR, LABELS_FILENAME)
LR = 1.5e-5
DIM = (128, 128, 3)
N_CLASSES = 3
