from typing import Tuple
import numpy as np

from ..misc.utils import isType


def firstOrderSpline(x: np.ndarray, y: np.ndarray):
    """
    Computes the coefficients of a first order spline, according to the formulas:

    .. math::
        a_k=\\frac{y_{k+1}-y_k}{x_{k+1}-x_k}
    .. math::
        b_k=y_k-a_k*x_k

    :param x: function arguments (x axis)
    :param y: function values (y axis)

    :return: the coefficients of the linear function in the tuple of shape (a0, a1)"""

    if not isType(x, np.ndarray) or not isType(y, np.ndarray) or len(x) != len(y):
        return None

    a = []
    b = []

    for k in range(len(x) - 1):
        a.append((y[k + 1] - y[k]) / (x[k + 1] - x[k]))
        b.append(y[k] - a[k] * x[k])

    return (a, b)


def cubicSpline(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes the coefficients of a third order (cubic) spline

    :param x: function arguments (x axis)
    :param y: function values (y axis)

    :return: the coefficients in the tuple of shape (a0, a1, a2, a3)"""

    if not isType(x, np.ndarray) or not isType(y, np.ndarray) or len(x) != len(y):
        return np.nan, np.nan, np.nan, np.nan

    h = np.zeros(len(y))
    d = np.zeros(len(y))
    lmb = np.zeros(len(y))
    rho = np.zeros(len(y))
    m = np.zeros(len(y))
    b3 = np.zeros(len(y) - 1)
    b2 = np.zeros(len(y) - 1)
    b1 = np.zeros(len(y) - 1)
    b0 = np.zeros(len(y) - 1)

    tgtLen = len(x) - 1

    for k in range(tgtLen):
        h[k] = x[k + 1] - x[k]
        d[k] = (y[k + 1] - y[k]) / h[k]

    for k in range(tgtLen):
        lmb[k + 1] = h[k + 1] / (h[k] + h[k + 1])
        rho[k + 1] = h[k] / (h[k] + h[k + 1])

    for k in range(tgtLen - 1):
        m[k + 1] = 3 * (d[k + 1] - d[k]) / (h[k] + h[k + 1]) - \
            m[k] * rho[k + 1] / 2 - m[k + 2] * lmb[k + 1] / 2

    for k in range(tgtLen):
        b0[k] = y[k]
        b1[k] = d[k] - h[k] * (2 * m[k] + m[k + 1]) / 6
        b2[k] = m[k] / 2
        b3[k] = (m[k + 1] - m[k]) / 6 * h[k]

    xCut = x[:-1]
    a3 = b3
    a2 = b2 - 3 * b3 * xCut
    a1 = b1 - 2 * b2 * xCut + 3 * b3 * xCut**2
    a0 = b0 - b1 * xCut + b2 * xCut**2 - b3 * xCut**3

    return (a0, a1, a2, a3)
