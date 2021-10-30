from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ....crs.geocrs import NoneCRS

class GeoRasterBackend(ABC):

    @property
    def crs():
        return NoneCRS

    @property
    @abstractmethod
    def has_pyramide():
        raise NotImplementedError

    @property
    @abstractmethod
    def num_bands():
        raise NotImplementedError

    @abstractmethod
    def get_raster_data(min_coords: Tuple, max_coords: Tuple, resolution: Tuple):
        raise NotImplementedError

    def supports_writes() -> bool:
        return False

    def write_raster_data(min_coords: Tuple, max_coords: Tuple, band: int, data: np.ndarray):
        raise NotImplemented