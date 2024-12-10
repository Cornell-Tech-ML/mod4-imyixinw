"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply `x` by `y`"""
    return x * y


def id(x: float) -> float:
    """Return input `x` unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add `x` and `y`"""
    return x + y


def neg(x: float) -> float:
    """Negate `x`"""
    return -x


def lt(x: float, y: float) -> float:
    """Return true if `x` is less than `y`, else false"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return true if `x` is equal to `y`, else false"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of `x` and `y`"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return true if `x` is close to `y`, else false"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of `x`"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the relu function of `x`"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of `x`"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function of `x`"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of `x`"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Compute the derivative of log of `x` times `y`"""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of the reciprocal of `x` times `y`"""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of the relu of `x` times `y`"""
    return y if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order-map

    Args:
    ----
        fn: A function that takes a float and returns a float

    Returns:
    -------
        A function that takes an iterable of floats, applies `fn' to each element, and returns a new iterable

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith

    Args:
    ----
        fn: A function that combines two floats into one

    Returns:
    -------
        A function that takes two iterables of floats, combines them element-wise using `fn', and returns a new iterable

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce

    Args:
    ----
        fn: A function that combines two floats into one
        start: The starting value $x_0$

    Returns:
    -------
        A function that takes an iterable of floats, reduces them to a single value using `fn', and returns the result

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add elements of `ls1` and `ls2` using `zipWith` and `add`"""
    return zipWith(add)(ls1, ls2)


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map' and `neg' to negate all elements in 'ls'"""
    return map(neg)(ls)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add'"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul'"""
    return reduce(mul, 1.0)(ls)
