from typing import Any, Tuple, cast
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt


def plotVector(title: str, ax, vec: np.ndarray, showLegendColorbar: bool = True) -> None:
    """
    Plots a vector colorbar in a transposed form, with an additional legend of values, if needed

    :param title: the title of the figure (or axes, if on a subplot)
    :param ax: Axes object
    :param vec: the vector
    :param showLegendColorbar: whether to show the legend colorbar, optional
    """

    ax.set_title(title)
    cax = ax.matshow(np.transpose(vec.reshape(len(vec), 1)))

    if showLegendColorbar:
        ax.colorbar(cax, location="bottom")


def plotMatrixAndVector(A: np.ndarray, b: np.ndarray, bigTitle: str = "Visualization of matrix A & vector b", matrixTitle: str = "Matrix A", vectorTitle: str = "Vector b", figsize=None) -> Tuple[Any, Tuple[Axes, Axes]]:
    """
    Plots a colormap presenting matrix A and a colorbar presenting vector b, with a legend of the range of values

    :param A: the matrix to be presented
    :param b: the vector to be presented; _note: the vector will be copied & transposed for visualization_
    :param bigTitle: the suptitle of the figure, optional
    :param matrixTitle: the title of the subplot of the matrix, optional
    :param vectorTitle: the title of the subplot of the vector, optional
    :param figsize: the size of the plot, optional

    :return: a tuple containing the figure and a tuple of the two axes
    """

    if not figsize:
        figsize = (12, 10)

    fig, (ax1, ax2) = cast(Any, plt.subplots(
        2, 1, constrained_layout=True, figsize=(figsize)))

    fig.suptitle(bigTitle)

    ax1.set_title(matrixTitle)
    cax1 = ax1.matshow(A)
    fig.colorbar(cax1, location="top")

    fig.show()

    return fig, (ax1, ax2)
