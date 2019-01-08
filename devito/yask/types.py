from devito.logger import yask as log
import devito.functions.basic as basic

from devito.yask.utils import namespace
from devito.yask.wrappers import contexts

__all__ = ['CacheManager', 'YaskGridObject', 'YaskSolnObject']


basic.Basic.from_YASK = False
basic.Basic.is_YaskGridObject = False
basic.Array.from_YASK = True


class YaskGridObject(basic.Object):

    is_YaskGridObject = True

    dtype = namespace['type-grid']
    value = None

    def __init__(self, mapped_function_name):
        self.mapped_function_name = mapped_function_name
        self.name = namespace['code-grid-name'](mapped_function_name)

    # Pickling support
    _pickle_args = ['mapped_function_name']
    _pickle_kwargs = []


class YaskSolnObject(basic.Object):

    dtype = namespace['type-solution']
    value = None

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = []


class CacheManager(basic.CacheManager):

    @classmethod
    def clear(cls):
        log("Dumping contexts and symbol caches")
        contexts.dump()
        super(CacheManager, cls).clear()
