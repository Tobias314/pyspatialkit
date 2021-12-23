from typing import Dict

import numpy as np

def datascheme_to_str_dict(datascheme: Dict[str, np.dtype]) -> Dict[str, str]:
    res = {}
    for key, value in datascheme.values():
        res[key] = np.dtype(value).str
    return res

def datascheme_from_str_dict(str_dict: Dict[str, str]) -> Dict[str, np.dtype]:
    res = {}
    for key, value in str_dict.values():
        res[key] = np.dtype(value)
    return res