from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon
import sentinelhub

from ..crs.geocrs import GeoCrs, NoneCRS
from ..globals import get_default_crs
from .georect import GeoRect

class GeoBox2d:
    """Describing a box area in 2D space. Described by a min and max point and a crs.
    """

    def __init__(self, min: Tuple[float, float], max: Tuple[float, float], crs: Optional[GeoCrs] = None):
        if crs is None:
            crs = get_default_crs()
        self.crs = crs
        assert len(min)==2 and len(max)==2
        self.bounds = np.array([*min, *max])

    @classmethod
    def from_bounds(cls, bounds: Tuple[float, float, float, float], crs: Optional[GeoCrs] = None):
        return cls(min=bounds[:2], max=bounds[2:], crs=crs)

    @classmethod
    def from_georect(cls, georect: GeoRect, use_bounds: bool = False):
        if not use_bounds and not georect.is_axis_aligned:
            raise ValueError("GeoRect is not axis alound. Set use_bounds=True if you want to crete the GeoBox from the bounds of the GeoRect!")            
        return cls.from_bounds(georect.bounds, crs=georect.crs)

    @property
    def min(self):
        return self.bounds[:2]

    @property
    def max(self):
        return self.bounds[2:]

    @property
    def center(self):
        return (self.max + self.min) / 2

    def to_georect(self):
        return GeoRect(self.min, self.max, crs=self.crs)

    def to_geoshape(self):
        return self.to_georect().to_geoshape() #TODO: maybe don't go via georect

    def to_shapely(self):
        return self.to_georect().to_shapely() #TODO: maybe don't transform to 

    def get_edge_lengths(self) -> np.ndarray:
        return self.max - self.min

    def substract_borders(self, border_size: Tuple[float, float]) -> 'GeoBox3d':
        border_size = np.array(border_size)
        return GeoBox3d(self.bounds[:2] + border_size, self.bounds[2:] - border_size, self.crs)

    def __str__(self)-> str:
        return "GeoBox2d(min: {}, max: {})".format(list(self.min), list(self.max))

    def __repr__(self)-> str:
        return str(self) 