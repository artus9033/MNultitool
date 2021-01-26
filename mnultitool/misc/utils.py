from typing import Any, Union, List, cast

import numpy as np


def isType(val: Any, typeArr: Union[List[Any], Any]) -> bool:
    """Helper that checks if the supplied value `val` is of a specific type or of a type present in `typeArr`"""

    if isinstance(typeArr, List):
        anyMatched = False

        for type in typeArr:
            if isinstance(val, type):
                anyMatched = True

        return anyMatched
    else:
        if not isinstance(val, typeArr):
            return False
    return True


def absoluteError(v: Union[int, float, List, np.ndarray], v_approx: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculates an absolute error of two arguments, either scalars, or vectors

    :param v: precise value
    :param v_approx: approximate value

    :return: absolute error value or NaN if input data is ill-formed
    """

    try:
        if((not isType(v, [int, float, List, np.ndarray])) or (not isType(v_approx, [int, float, List, np.ndarray]))):
            return np.nan

        if isinstance(v, List):
            v = np.array(v)

        if isinstance(v_approx, List):
            v_approx = np.array(v_approx)

        if isinstance(v, np.ndarray):
            return np.abs(cast(np.ndarray, v) - v_approx)
        elif isinstance(v_approx, np.ndarray):
            return np.abs(cast(np.ndarray, v) - v_approx)
        else:
            return np.abs(cast(Union[int, float], v) - cast(Union[int, float], v_approx))
    except:
        return np.nan


def relativeError(v: Union[int, float, List, np.ndarray], v_approx: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculates a relative error of two arguments, either scalars, or vectors

    :param v: precise value
    :param v_approx: approximate value

    :return: relative error value or NaN if input data is ill-formed
    """

    try:
        if not isType(v, [int, float, List, np.ndarray]):
            return np.nan

        if isType(v, [int, float]):
            if v == 0:
                return np.nan
        else:
            hasZero = False

            for val in cast(Union[List, np.ndarray], v):
                if val == 0:
                    hasZero = True

            if hasZero:
                return np.nan

        res = absoluteError(v, v_approx)

        if res is np.nan:
            return np.nan
        else:
            return res / np.abs(v)
    except:
        return np.nan
