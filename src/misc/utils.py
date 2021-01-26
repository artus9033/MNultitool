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


def absoluteError(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Calculates an absolute error of two arguments, either scalars, or vectors

    Parameters:
    v (Union[int, float, List, np.ndarray]): precise value
    v_aprox (Union[int, float, List, np.ndarray]): approximate value

    Returns:
    err Union[int, float, np.ndarray]: absolute error value or NaN if input data is ill-formed
    """

    try:
        if((not isType(v, [int, float, List, np.ndarray])) or (not isType(v_aprox, [int, float, List, np.ndarray]))):
            return np.nan

        if isinstance(v, List):
            v = np.array(v)

        if isinstance(v_aprox, List):
            v_aprox = np.array(v_aprox)

        if isinstance(v, np.ndarray):
            return np.abs(cast(np.ndarray, v) - v_aprox)
        elif isinstance(v_aprox, np.ndarray):
            return np.abs(cast(np.ndarray, v) - v_aprox)
        else:
            return np.abs(cast(Union[int, float], v) - cast(Union[int, float], v_aprox))
    except:
        return np.nan


def relativeError(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Calculates a relative error of two arguments, either scalars, or vectors

    Parameters:
    v (Union[int, float, List, np.ndarray]): precise value
    v_aprox (Union[int, float, List, np.ndarray]): approximate value

    Returns:
    err Union[int, float, np.ndarray]: relative error value or NaN if input data is ill-formed
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

        res = absoluteError(v, v_aprox)

        if res is np.nan:
            return np.nan
        else:
            return res / np.abs(v)
    except:
        return np.nan
