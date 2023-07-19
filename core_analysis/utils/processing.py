# -*- coding: utf-8 -*-


class stored_property(property):
    def __get__(self, obj, objtype=None):
        name = "_" + self.fget.__name__
        if not hasattr(obj, name):
            setattr(obj, name, self.fget(obj))
        return getattr(obj, name)
