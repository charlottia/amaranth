import contextlib
import functools
import warnings
import linecache
import re
from collections import OrderedDict
from collections.abc import Iterable

from .utils import *


__all__ = ["flatten", "union" , "log2_int", "bits_for", "memoize", "final", "deprecated",
           "get_linter_options", "get_linter_option",
           "ConformableMeta", "redirect_subclasses"]


def flatten(i):
    for e in i:
        if isinstance(e, str) or not isinstance(e, Iterable):
            yield e
        else:
            yield from flatten(e)


def union(i, start=None):
    r = start
    for e in i:
        if r is None:
            r = e
        else:
            r |= e
    return r


def memoize(f):
    memo = OrderedDict()
    @functools.wraps(f)
    def g(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]
    return g


def final(cls):
    def init_subclass():
        raise TypeError("Subclassing {}.{} is not supported"
                        .format(cls.__module__, cls.__name__))
    cls.__init_subclass__ = init_subclass
    return cls


def deprecated(message, stacklevel=2):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def _ignore_deprecated(f=None):
    if f is None:
        @contextlib.contextmanager
        def context_like():
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=DeprecationWarning)
                yield
        return context_like()
    else:
        @functools.wraps(f)
        def decorator_like(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=DeprecationWarning)
                return f(*args, **kwargs)
        return decorator_like


def extend(cls):
    def decorator(f):
        if isinstance(f, property):
            name = f.fget.__name__
        else:
            name = f.__name__
        setattr(cls, name, f)
    return decorator


def get_linter_options(filename):
    first_line = linecache.getline(filename, 1)
    if first_line:
        match = re.match(r"^#\s*amaranth:\s*((?:\w+=\w+\s*)(?:,\s*\w+=\w+\s*)*)\n$", first_line)
        if match:
            return dict(map(lambda s: s.strip().split("=", 2), match.group(1).split(",")))
    return dict()


def get_linter_option(filename, name, type, default):
    options = get_linter_options(filename)
    if name not in options:
        return default

    option = options[name]
    if type is bool:
        if option in ("1", "yes", "enable"):
            return True
        if option in ("0", "no", "disable"):
            return False
        return default
    if type is int:
        try:
            return int(option, 0)
        except ValueError:
            return default
    assert False


class ConformableMeta(type):
    """
    Abstract metaclass base for creating a conformable type.

    Subclass ``ConformableMeta`` to define the properties of a conformable type:
    what instances and classes are considered conforming.

    Classes created with such a metaclass are conformable classes. When
    ``isinstance`` and ``issubclass`` are called with the conformable class in
    the second argument, the conformable type determines the result. Default
    Python behavior is used for subclasses of the conformable class, permitting
    extension of the conformable type with regular inheritance.
    """

    __top = None

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if not bases:
            assert mcls.__top is None, (f"conformable type object '{mcls.__name__}' already "
                                        f"has a conformable class '{mcls.__top.__name__}'")
            mcls.__top = cls
        return cls

    def __init_subclass__(cls):
        for absmeth in ("conformable_isinstance", "conformable_issubclass"):
            if absmeth not in vars(cls):
                raise TypeError(f"type object '{cls.__name__}' deriving from " # :nocov:
                                f"`ConformableMeta` must override the `{absmeth}` method")

    def conformable_isinstance(cls, instance):
        """Determine if an instance should be considered an instance of a
        conformable class.

        Arguments
        ---------
        instance : any
            The object being tested.

        Returns
        -------
        :class:`bool` or NotImplemented
            True or false will be returned to the ``isinstance`` caller
            directly. NotImplemented will fall back to the default Python
            implementation.
        """
        raise NotImplementedError # :nocov:

    def conformable_issubclass(cls, subclass):
        """Determine if a class should be considered a subclass of a conformable
        class.

        Arguments
        ---------
        subclass : class
            The class being tested.

        Returns
        -------
        :class:`bool` or NotImplemented
            True or false will be returned to the ``issubclass`` caller
            directly. NotImplemented will fall back to the default Python
            implementation.
        """
        raise NotImplementedError # :nocov:

    def __instancecheck__(cls, instance):
        if cls is not cls.__top:
            return super().__instancecheck__(instance)

        result = cls.conformable_isinstance(instance)
        if result is not NotImplemented:
            return result
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is not cls.__top:
            return super().__subclasscheck__(subclass)

        result = cls.conformable_issubclass(subclass)
        if result is not NotImplemented:
            return result
        return super().__subclasscheck__(subclass)


def redirect_subclasses(target_mcls, from_cls, to_cls):
    """Redirect further subclassing of a class to a different class.

    Used with conformable types in order to allow differentiating true subclass
    instances from those that only conform to the conformable type.

    Arguments
    ---------
    target_mcls : type
        The metaclass where the redirection is to occur.
    from_cls : class
        The class to redirect subclassing attempts away from.
    to_cls : class
        The class which will be subclassed instead.
    """
    assert target_mcls in type(from_cls).__mro__, (
        f"type object '{target_mcls.__name__}' cannot redirect subclassing "
        f"of '{from_cls.__name__}'")

    to_mcls_new = type(to_cls).__new__
    def __new__(mcls, name, bases, namespace, /, **kwargs):
        bases = tuple(
            (to_cls if base is from_cls else base)
            for base in bases
        )
        return to_mcls_new(type(to_cls), name, bases, namespace, **kwargs)
    target_mcls.__new__ = __new__
