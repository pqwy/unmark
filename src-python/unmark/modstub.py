# Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
# See LICENSE.md

class stub (object):
    def __init__ (self, module, fun):
        self.m, self.f = module, fun
    def __call__ (self, *a, **kw):
        raise Exception (
            "'%s' is unavailable, because module '%s' failed to load. :(" % (
                self.f, self.m))

def requires (*mods):
    try:
        for m in mods:
            __import__ (m)
        return (lambda f: f)
    except ModuleNotFoundError as exn:
        module = exn.name
        return (lambda f: stub (module, f.__name__))
