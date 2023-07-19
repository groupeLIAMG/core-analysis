# -*- coding: utf-8 -*-

from os import makedirs
from os.path import join, exists

import numpy as np

from core_analysis.utils.constants import BACKGROUND_DIR


class stored_property(property):
    def __get__(self, obj, objtype=None):
        name = "_" + self.fget.__name__
        if not hasattr(obj, name):
            setattr(obj, name, self.fget(obj))
        return getattr(obj, name)


def save_array_property(dir):
    class saved_property(property):
        def __init__(self, fget=None):
            self._fget = fget
            super().__init__(fget=self.fget)

        def fget(self, obj):
            path = join(dir, f"{obj.filename}.npy")

            if exists(path):
                return self.load(path)
            else:
                self = self._fget()
                self.save(path, obj)
                return self

        def load(self, path):
            return np.load(path)

        def save(self, path, obj):
            np.save(path, obj)

    return saved_property


def automatically_makedirs(function):
    def wrapper(*args, **kwargs):
        path = args[0]
        if not exists(path):
            makedirs(path)
        return function(*args, **kwargs)

    return wrapper


np.save = automatically_makedirs(np.save)
