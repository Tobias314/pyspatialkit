
from logging import getLogger, Logger

def logger() -> Logger:
    logger = getLogger("pygeodata")
    return logger

def dbg(*args, **kwargs):
    print(*args, **kwargs)
