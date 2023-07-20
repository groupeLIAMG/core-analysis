# -*- coding: utf-8 -*-

from os import makedirs
from os.path import join, split, exists, splitext

import numpy as np
from h5py import File

from core_analysis.utils.constants import DATA_DIR


class stored_property(property):
    def __get__(self, obj, objtype=None):
        name = "_" + self.fget.__name__
        if not hasattr(obj, name):
            setattr(obj, name, self.fget(obj))
        return getattr(obj, name)


class saved_array_property(stored_property):
    def __init__(self, fget=None):
        self._fget = fget
        super().__init__(fget=self.fget)
        self.filename = f"{fget.__name__}.h5"
        self.archive = File(join(DATA_DIR, self.filename), "a")

    def fget(self, obj):
        key = obj.filename

        if key in self.archive.keys():
            return self.archive[key][:]
        else:
            obj = self._fget(obj)
            self.archive[key] = obj
            return obj


def replace_ext(path, ext):
    ext = ext.lstrip(".")
    return f"{splitext(path)[0]}.{ext}"


def automatically_makedirs(function):
    def wrapper(*args, **kwargs):
        path = args[0]
        if "." in path:
            path, _ = split(path)
        if path and not exists(path):
            makedirs(path)
        return function(*args, **kwargs)

    return wrapper


np.save = automatically_makedirs(np.save)
