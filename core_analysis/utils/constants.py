# -*- coding: utf-8 -*-

from os.path import join
from datetime import date

TODAY = str(date.today()).replace("-", "_")
BATCH_SIZE = 500
IMAGE_DIR = join("data", "images")
CHECKPOINT_DIR = join("data", "models", "save_models")
LOAD_FILENAME = "linknet_efficientnetb7_weights_2023_07_05.h5"
LR = 1.5e-5
DIM = (128, 128, 3)
N_CLASSES = 3
