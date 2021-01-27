from typing import List, Tuple, cast
import numpy as np
import scipy.linalg as spLinalg

from ..misc.utils import isType
from .classification import isMatrixDiagDominant


def svdAndReconstruction(A: np.ndarray, singularValues: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an SVD decomposition of matrix A & reconstructs
    the matrix using these components and supplied singular values of matrix A

    :param A: matrix A of shape (m, m)
    :param singularValues: singular values vector of shape (m, 1)

    :raises ValueError: When inputs' types are incorrect or either shape is invalid

    :returns: (U, S, V, M), where U, S & V are SVD decomposition components & M is the reconstruction of A
    """

    if not isType(A, np.ndarray) or not isType(singularValues, np.ndarray):
        raise ValueError("Wrong input types")

    m, shouldBeM = A.shape
    if m != shouldBeM or m != singularValues.shape[0]:
        raise ValueError("Wrong input shapes")

    U, S, V = np.linalg.svd(A)

    return U, S, V, (U * singularValues) @ V


def eqSysRectToSquare(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms an equation system with a rectangular matrix of coefficients, to an equation system with a square matrix
    Note: both the matrix & the vector returned will be different from the inputs

    :param A: the rectangular coefficients matrix of shape (m, n)
    :param b: vector b of shape (m, 1), containg the right hand side coefficients

    :raises ValueError: When inputs' types are incorrect or either shape is invalid

    :return: tuple containing a square matrix of shape size (n, n) & a modified vector b of shape (n, 1)
    """

    if not isType(A, np.ndarray) or not isType(b, np.ndarray):
        raise ValueError("Wrong input types")

    m, n = A.shape
    if m != b.shape[0]:
        raise ValueError("Wrong input shapes")

    At = np.transpose(A)

    semiNewA = At @ A
    newB = At @ b

    return semiNewA, newB


def eqSysLeastSquaresAndSolve(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms an equation system using the least squares method according to the formula:

    .. math::
        A^T Ax = A^T b

    and solves the equation system using sp.linalg.solve

    :param A: a square coefficients matrix A of shape (m, m)
    :param b: vector of right hand side coefficients b of shape (m, 1)

    :return: a tuple containing (in order): vector of eq sys solutions x, transformed matrix A^T A, transformed vector A^T b
    """

    AtA, Atb = eqSysRectToSquare(A, b)

    x = np.linalg.solve(AtA, Atb)

    return x, AtA, Atb


def eqSysQRAndSolve(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms an equation system using QR decomposition and solves the equation system using sp.linalg.solve_triangular

    :param A: a square coefficients matrix A of shape (m, m)
    :param b: vector of right hand side coefficients b of shape (m, 1)

    :return: a tuple containing (in order): vector of eq sys solutions x, Q, R where Q & R are QR decomposition components
    """

    Q, R = cast(Tuple[np.ndarray, np.ndarray], np.linalg.qr(A, mode='reduced'))

    Qb = Q.T @ b
    x = spLinalg.solve_triangular(R, Qb)

    return x, Q, R


def eqSysSVDAndSolve(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms an equation system using SVD decomposition and solves the equation system using sp.linalg.solve

    :param A: a square coefficients matrix A of shape (m, m)
    :param b: vector of right hand side coefficients b of shape (m, 1)

    :return: a tuple containing (in order): vector of eq sys solutions x, U, S, V where U, S & V are SVD decomposition components
    """

    U, S, V = np.linalg.svd(A)

    c = U.T @ b
    w = np.linalg.solve(np.diag(S), c)
    x = V.T @ w

    return x, U, S, V


def eqSysLUAndSolveJacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                          epsilon: float = 1e-8, maxiter: int = 100, checkNecessaryCondition: bool = False) -> Tuple[np.ndarray, int]:
    """
    Transforms an equation system using LU decomposition and solves the equation system using Jacobi's method

    :param A: a square coefficients matrix A of shape (m, m)
    :param b: vector of right hand side coefficients b of shape (m, 1)
    :param x_init: initial solution of shape (m, 1)
    :param epsilon: the desired solution precision
    :param maxiter: a maximum iterations limit (to prevent an infinite loop), a positive integer

    :raises ValueError: If either input is of invalid type, matrix A does not have exactly 2 dimensions, vector b is of invalid shape,\
        b's first dimension does not match A's, matrix A is not square, x_init has an invalid shape, epsilon or maxiter are not positive integers,\
            or matrix A is not strictly diagonally dominant (in such a case Jacobi's method would not converge)

    :return: a tuple containing (in order): eq sys solution x of shape (m, 1), L, U, iter, where L & U are LU decomposition components \
        and iter is the number of iterations passed
    """

    if not isType(A, np.ndarray) or len(A.shape) != 2 or not isType(b, np.ndarray) or len(b.shape) > 2 \
            or not isType(x_init, np.ndarray) or not isType(epsilon, float) or not isType(maxiter, int) or maxiter < 0 or epsilon < 0:
        raise ValueError("Invalid input values or types")

    m, shouldBeM = A.shape
    if m != shouldBeM:
        raise ValueError("Matrix A is not square")

    if len(b.shape) == 1:
        if b.shape[0] != m:
            raise ValueError("Vector b's first dimension is of invalid length")
    else:
        shouldAlsoBeM, shouldBe1 = b.shape
        if shouldBe1 != 1 or m != shouldAlsoBeM:
            raise ValueError(
                "Vector b is of invalid shape")

    xInitShouldBeM, xInitShouldBe1 = x_init.shape
    if xInitShouldBe1 != 1 or xInitShouldBeM != m:
        raise ValueError("Vector xinit is of invalid shape")

    # necessary convergence condition of Jacobi's method
    if checkNecessaryCondition and not isMatrixDiagDominant(A):
        raise ValueError(
            "Matrix A is not strictly diagonally dominant, therefore Jacobi's method is not convergent for it")

    diagMat = np.diag(np.diag(A))
    LU = A - diagMat

    x = x_init
    invDiagMat = np.diag(1 / np.diag(diagMat))
    residuals = []
    for i in range(maxiter):
        x_new = invDiagMat @ (b - LU @ x)
        rnorm = np.linalg.norm(x_new - x)

        residuals.append(rnorm)

        if rnorm < epsilon:
            return x_new, i + 1

        x = x_new

    return x, maxiter


def frobeniusFromPolyCoeffs(coeffs: List[float]) -> np.ndarray:
    """
    Generates a Frobenius matrix for a given list of polynomial coefficients, ordered from the one next to the highest power of x, to the one next to the lowest (that is, just a scalar)

    For example: w(x) = 5x^3 + 4x - 3 => coeffs = [5, 0, 4, -3]

    .. math::
        \\setcounter{MaxMatrixCols}{20}
        \\begin{vmatrix}
            1     &            &           &                  \\\\
                  &  \\ddots   &           &                  \\\\
                  &            &     1     &                  \\\\
            -a_0  &    -a_1    &  \\dots   &    -a_{n-1}
        \\end{vmatrix}

    :raises ValueError: when the input is not a list, or the first coefficient is equal to 0 (and cannot be a divisor then)

    :returns: the Frobenius array corresponding to the polynomial with given coefficients
    """

    if not isType(coeffs, List):
        raise ValueError("Wrong input type")

    if coeffs[0] == 0:
        raise ValueError(
            "The highest power's coefficient is equal to 0 (therefore cannot be a divisor)")

    aBiggestPower, *coeffs = coeffs

    coeffs = np.array(coeffs) / aBiggestPower

    coeffsLen = len(coeffs)

    frob = np.zeros(shape=(coeffsLen, coeffsLen))

    # fill ones
    for i in range(coeffsLen - 1):
        frob[i][i + 1] = 1

    # fill last row
    lastRowInd = coeffsLen - 1
    for i in range(coeffsLen):
        frob[lastRowInd][len(frob[lastRowInd]) - i - 1] = -coeffs[i]

    return frob
