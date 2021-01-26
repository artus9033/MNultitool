from typing import Optional
import numpy as np

from ..misc.utils import isType


def chebyshevNodes(n: int = 10) -> Optional[np.ndarray]:
    """Creates a vector of Chebyshev nodes in the shape of (n+1, )

    :param n: amount of Chebyshev nodes, a positive integer

    :returns: the vector of chebyshev nodes
    :rtype: np.ndarray
    """

    if not isType(n, int) or n < 0:
        return None

    # return np.cos(np.linspace(1, n, n) * np.pi / n)
    return np.array([np.cos(i * np.pi / n) for i in range(0, n + 1)])
