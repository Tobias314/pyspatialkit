import numpy as np


def next_bigger_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    k = dtype.kind
    if k in {'u', 'i', 'b'}:
        return np.dtype(k + str(dtype.itemsize + 1))
    elif k == 'f':
        if dtype.itemsize < 8:
            return np.dtype('f8')
        elif dtype.itemsize < 4:
            return np.dtype('f4')
        else:
            return dtype
    else:
        return dtype
