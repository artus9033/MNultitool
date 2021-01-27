from typing import Tuple, Union

import numpy as np

from .utils import isType


def exponential(x: Union[int, float], n: int) -> float:
    """
    Computes an approximation of e^x

    :param x: x
    :param k: k

    :return: the approximated value
    """

    acc = 0

    try:
        if isType(x, [int, float]) and isType(n, int) and n >= 0:
            for i in range(0, n):
                acc += pow(x, i) / np.math.factorial(i)

            return acc
        else:
            return np.nan
    except OverflowError:
        return acc


def coskx1(k: int, x: Union[int, float]) -> float:
    """
    Computes an approximation of cos(kx) using the formula:

    .. math::
        cos((m+1)x) = 2cos(x) * cos(mx) - cos((m-1)x)

    :param x: x
    :param k: k

    :return: the approximated value
    """

    try:
        if isType(k, int) and k >= 0 and isType(x, [int, float]):
            m1 = k - 1
            m2 = m1 - 1

            if m1 >= 1:
                if m1 == 1:
                    e1 = np.math.cos(x)
                else:
                    e1 = coskx1(m1, x)
            else:
                e1 = 1

            if m2 >= 1:
                if m2 == 1:
                    e2 = np.math.cos(x)
                else:
                    e2 = coskx1(m2, x)
            else:
                e2 = 1

            if k == 0:
                return np.math.cos(0)
            elif k == 1:
                return np.math.cos(x)
            else:
                return 2 * np.math.cos(x) * e1 - e2
        else:
            return np.nan
    except:
        return np.nan


def cossinkx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """
    Computes an approximation of cos(kx) & sin(kx) using the formulas:

    .. math::
        cos(mx) = cosx \\cdot cos(m-1)x - sinx \\cdot sin(m-1)x
    .. math::
        sin(mx) = sinx \\cdot cos(m-1)x + cosx \\cdot sin(m-1)x

    :param x: x
    :param k: k

    :return: approximated cos(kx) and sin(kx) values
    """

    try:
        if isType(k, int) and k >= 0 and isType(x, [int, float]):
            if k == 0:
                return (np.math.cos(0), np.math.sin(0))
            elif k == 1:
                return (np.math.cos(x), np.math.sin(x))
            else:
                (cosKLess1, sinKLess1) = cossinkx2((k - 1), x)
                return (np.math.cos(x) * cosKLess1 - np.math.sin(x) * sinKLess1, np.math.sin(x) * cosKLess1 + np.math.cos(x) * sinKLess1)
        else:
            return np.nan, np.nan
    except:
        return np.nan, np.nan


def pi(n: int) -> float:
    """Computes an approximate value of Pi, using the formula:

    .. math::
        \\sum_{n=1}^{\\infty} \\cfrac{1}{n^2} = \\cfrac{1}{6} \\pi^2


    :param n: the amount of sum elements

    :return: the approximated value of pi
    """

    if isType(n, int) and n > 0:
        acc = 0

        for i in range(1, n + 1):
            acc += 1 / pow(i, 2)

        return np.math.sqrt(acc * 6)
    else:
        return np.nan
