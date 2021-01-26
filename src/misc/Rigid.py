from math import pi

import numpy as np


def cylinderArea(r: float, h: float):
    """
    Computes the total area of a cylinder

    :param r: cylinder base radius
    :param h: cylinder height

    :return: cylinder area
    :rtype: float
    """

    if r > 0 and h > 0:
        return pi * pow(r, 2) * 2 + 2 * pi * r * h
    else:
        return np.nan


def fibonacci(n: int):
    """
    Computes the first n Fibonacci's sequence elements

    Parameters:
    :param n: the amount of elements

    :return: n first Fibonacci's sequence elements
    :rtype: np.ndarray
    """

    if n <= 0 or isinstance(n, float):
        return None
    else:
        arr = np.ndarray(shape=(1, n), dtype=int)

        for i in range(0, n):
            if i == 0 or i == 1:
                arr[0][i] = 1
            else:
                arr[0][i] = arr[0][i - 1] + arr[0][i - 2]

        return arr
