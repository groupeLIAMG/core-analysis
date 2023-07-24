# -*- coding: utf-8 -*-

from os.path import join
from datetime import date

TODAY = str(date.today()).replace("-", "_")
BATCH_SIZE = 500
IMAGE_DIR = "images"
DATA_DIR = "data"
PLOT_DIR = join(DATA_DIR, "plots")
MODEL_DIR = join(DATA_DIR, "models", "save_models")
MODEL_FILENAME = "linknet_efficientnetb7_weights_2023_07_05.h5"
MODEL_PATH = join(MODEL_DIR, MODEL_FILENAME)
LABELS_FILENAME = "labels_20230714.json"
LABELS_DIR = join(DATA_DIR, "json_files")
LABELS_PATH = join(LABELS_DIR, LABELS_FILENAME)
LR = 1.5e-5
DIM = (128, 128, 3)
N_CLASSES = 3
