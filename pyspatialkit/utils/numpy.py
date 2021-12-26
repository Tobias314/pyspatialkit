import sys

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

def to_endianess(arr: np.ndarray, to_endianess:str='<', inplace=True)-> np.ndarray:
    arr_endianess = arr.dtype.byteorder
    if arr_endianess=='=':
        if sys.byteorder=='little':
            arr_endianess = '<'
        else:
            arr_endianess = '>'
    if arr_endianess == '|' or (arr_endianess==to_endianess):
        if not inplace:
            return arr.copy()
    else:
        arr = arr.byteswap(inplace=inplace)
    return arr