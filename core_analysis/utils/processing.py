# -*- coding: utf-8 -*-


class stored_property(property):
    def __get__(self, obj):
        name = "_" + self._name
        if not hasattr(obj, name):
            setattr(obj, name, self.fget())
        return getattr(obj, name)
