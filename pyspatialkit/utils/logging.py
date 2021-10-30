
from logging import getLogger, Logger

def logger() -> Logger:
    logger = getLogger("pygeodata")
    return logger