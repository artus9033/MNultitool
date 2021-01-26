from typing import Tuple
import numpy as np

from .misc.utils import isType


def svdAndReconstruction(A: np.ndarray, singularValues: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an SVD decomposition of matrix A & reconstructs
    the matrix using these components and supplied singular values of matrix A

    :param A: matrix A of shape (m,m)
    :param singularValues(np.ndarray): wektor wartości singularnych (m,)


    :returns: (U, S, V, M), where U, S & V are SVD decomposition components & M is the reconstruction of A"""

    if not isType(A, np.ndarray) or not isType(singularValues, np.ndarray):
        raise ValueError("Wrong input types")

    m, shouldBeM = A.shape

    if m != shouldBeM or m != singularValues.shape[0]:
        raise ValueError("Wrong input shapes")

    U, S, V = np.linalg.svd(A)

    return U, S, V, (U * singularValues) @ V
