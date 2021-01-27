import numpy as np

from ..misc.utils import isType


def isMatrixDiagDominant(A: np.ndarray) -> bool:
    """
    Checks if matrix is strictly diagonally dominated

    :param A: the matrix to be checked

    :raises ValueError: If the supplied np.ndarray does not have exactly 2 dimensions or it is not square

    :return: whether the matrix is strictly diagonally dominated
    """

    if not isType(A, np.ndarray) or len(A.shape) != 2:
        raise ValueError("A matrix must have exactly 2 dimensions")

    m, shouldBeM = A.shape
    if m != shouldBeM:
        raise ValueError(
            f"The matrix has to be square, while a shape of ({m}, {shouldBeM}) was supplied")

    diagVals = np.diag(np.abs(A))
    rowSumsExclDiag = np.sum(np.abs(A), axis=1) - diagVals

    if np.all(diagVals > rowSumsExclDiag):
        return True
    else:
        return False


def isMatrixSymmetric(A: np.ndarray) -> bool:
    """
    Checks if a square matrix is symmetric

    :param A: the matrix to be checked

    :raises ValueError: If the supplied np.ndarray does not have exactly 2 dimensions or it is not square

    :return: whether the matrix is symmetric
    """

    if not isType(A, np.ndarray) or len(A.shape) != 2:
        raise ValueError("A matrix must have exactly 2 dimensions")

    m, shouldBeM = A.shape
    if m != shouldBeM:
        raise ValueError(
            f"The matrix has to be square, while a shape of ({m}, {shouldBeM}) was supplied")

    for i in range(0, m):
        for j in range(0, round(len(A[i]) / 2)):
            if i != j and A[i][j] != A[j][i]:
                return False

    return True
