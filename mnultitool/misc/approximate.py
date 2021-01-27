from typing import Callable, Tuple, Union

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


def approxZeroBisection(a: Union[int, float], b: Union[int, float], f: Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    """
    Approximates the solution to f(x) = 0 in range [a, b] using bisection method

    :param a: left range bound
    :param b: right range bound
    :param f: callable function of variable x
    :param epsilon: the desired precision (stop condition)
    :param iteration: maximum iterations count limit (to prevent an infinite loop)

    :raises ValueError: when input types are wrong or f has same signs on both range bounds

    :return: a tuple containing (in order): an approximate solution x, iterations made count 
    """

    if not isType(a, [int, float]) or not isType(b, [int, float]) or not isType(f, Callable) or not isType(epsilon, float) or not isType(iteration, int) or epsilon < 0 or iteration < 0 or b < a:
        raise ValueError("Wrong input types")

    if np.sign(f(a)) * np.sign(f(b)) >= 0:
        raise ValueError(
            "Function f has to be of different signs at the edges of the range [a, b]")

    i = 0
    c = (a + b) / 2
    while np.abs(f(c)) > epsilon and i < iteration:
        c = (a + b) / 2

        if np.sign(f(a)) != np.sign(f(c)):
            b = c
        else:
            a = c

        i += 1

    return c, i - 1


def approxZeroSecant(a: float, b: float, f: Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    """
    Approximates the solution to f(x) = 0 in range [a, b] using secant method

    :param a: left range bound
    :param b: right range bound
    :param f: callable function of variable x
    :param epsilon: the desired precision (stop condition)
    :param iteration: maximum iterations count limit (to prevent an infinite loop)

    :raises ValueError: when input types are wrong or f has same signs on both range bounds

    :return: a tuple containing (in order): an approximate solution x, iterations made count 
    """

    if not isType(a, [int, float]) or not isType(b, [int, float]) or not isType(f, Callable) or not isType(epsilon, float) or not isType(iteration, int) or epsilon < 0 or iteration < 0 or b < a or np.sign(f(a)) * np.sign(f(b)) >= 0:
        raise ValueError("Wrong input types")

    if np.sign(f(a)) * np.sign(f(b)) >= 0:
        raise ValueError(
            "Function f has to be of different signs at the edges of the range [a, b]")

    f_a = f(a)
    f_b = f(b)
    x0 = 0
    i = 0
    while iteration > -1 and (abs(a - b) > epsilon):
        i += 1

        if abs(f_a - f_b) < epsilon:
            break

        x0 = a - f(a) * (a - b) / (f(a) - f(b))
        f0 = f(x0)

        if abs(f0) < epsilon:
            break

        if f(a) * f0 < 0:
            b = x0

        elif f(b) * f0:
            a = x0

        elif f0 == 0:
            return x0, iteration

        iteration -= 1

    return x0, i - 1


def approxZeroNewton(f: Callable[[float], float], df: Callable[[float], float], ddf: Callable[[float], float], a: Union[int, float], b: Union[int, float], epsilon: float, iteration: int) -> Tuple[float, int]:
    """
    Approximates the solution to f(x) = 0 in range [a, b] using Newton's method

    :param f: callable function of variable x
    :param df: callable derivate of f (f')
    :param ddf: callable second-order derivate of f (f'')
    :param a: left range bound
    :param b: right range bound
    :param epsilon: the desired precision (stop condition)
    :param iteration: maximum iterations count limit (to prevent an infinite loop)

    :raises ValueError: when input types are wrong or either df or dff has same different on both range bounds

    :return: a tuple containing (in order): an approximate solution x, iterations made count 
    """

    if not isType(a, [int, float]) or not isType(b, [int, float]) or not isType(f, Callable) or not isType(df, typing.Callable) \
            or not isType(ddf, Callable) or not isType(epsilon, float) or not isType(iteration, int) or epsilon < 0 or iteration < 0 or b < a or np.sign(f(a)) * np.sign(f(b)) >= 0:
        raise ValueError("Wrong input types")

    if np.sign(df(a)) * np.sign(df(b)) <= 0:
        raise ValueError(
            "Df has to be of different signs at the edges of range [a, b]")

    if np.sign(ddf(a)) * np.sign(ddf(b)) <= 0:
        raise ValueError(
            "Ddf has to be of different signs at the edges of range [a, b]")

    x0 = a
    for x0 in [a, b]:
        if(f(x0) * ddf(x0) > 0):
            # initial condition found
            break

    i = 0
    while np.abs(f(x0)) > epsilon and i < iteration:
        x0 = x0 - f(x0) / df(x0)

        i += 1

    return x0, i
