
from logging import getLogger, Logger
import warnings
import inspect
import sys
#from typing import Type

def logger() -> Logger:
    logger = getLogger("pygeodata")
    return logger

def dbg(*args, **kwargs):
    print(*args, **kwargs)

def raise_warning(warning: str, category = UserWarning):
    warnings.warn(warning, category, stacklevel=2)
