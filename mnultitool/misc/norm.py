from typing import Any, List, Union, cast
import numpy as np

from ..misc.utils import isType


def maxNorm(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray]) -> float:
    """
    Computes the max norm (sometimes called the 'infinity norm') of scalars or vectors; both arguments' dimensions must be compatible

    :param xr: precise value, either a scalar or a vector of shape (n, 1)
    :param x: approximate value, either a scalar or a vector of shape (n, 1)

    :return: the value of the norm, max(abs(xr - x))
    """

    if not isType(xr, [int, float, List, np.ndarray]) or not isType(x, [int, float, List, np.ndarray]):
        return np.nan

    if (isType(xr, [int, float]) and not isType(x, [int, float])) or (isType(x, [int, float]) and not isType(xr, [int, float])):
        return np.nan

    xNp: np.ndarray
    if isType(x, [List]):
        xNp = np.array(x)
    else:
        xNp = cast(Any, x)

    xrNp: np.ndarray
    if isType(xr, [List]):
        xrNp = np.array(xr)
    else:
        xrNp = cast(Any, xr)

    if isType(xrNp, [int, float]) or isType(xNp, [int, float]):
        return np.max(np.abs(xrNp - xNp))
    else:
        if xrNp.shape == xNp.shape:
            return np.max(np.abs(xrNp - xNp))
        else:
            print(xrNp.shape, xNp.shape)
            return np.nan


def residualNorm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the matrix norm of the residues of an equation in a form of Ax = b


    :param A: matrix A (m, m) containing the equation's coefficients
    :param x: vector x (m, 1) containing the equation's solutions
    :param b: vector b (m, 1) containing the equation's right hand side's coefficients

    :return: matrix norm value of the equation's residues"""

    if not isType(A, np.ndarray) or not isType(x, np.ndarray) or not isType(b, np.ndarray):
        return np.nan

    m, n = A.shape
    if m != n or m != b.shape[0] or m != x.shape[0]:
        return np.nan

    return np.linalg.norm(b - A @ x)
