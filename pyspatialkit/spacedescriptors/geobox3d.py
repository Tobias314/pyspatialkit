from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon
import sentinelhub

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer
from .tiles3dboundingvolume import Tiles3dBoundingVolume
from ..globals import DEFAULT_CRS


class GeoBox3d(Tiles3dBoundingVolume):
    """Describing a Volume in 3D space. Described by a min and max point and a crs.
    """

    def __init__(self, min: Tuple[float, float, float], max: Tuple[float, float ,float], crs: GeoCrs = DEFAULT_CRS):
        self.crs = crs
        assert len(min)==3 and len(max)==3
        self.bounds = np.array([*min, *max])
        self.min = np.array(min)
        self.max = np.array(max)

    @classmethod
    def from_bounds(bounds: Tuple[float, float, float, float, float, float], crs: GeoCrs = DEFAULT_CRS):
        return GeoBox3d(min=bounds[:3], max=bounds[3:], crs=crs)

    @property
    def min(self):
        return self.bounds[:3]

    @property
    def max(self):
        return self.bounds[3:]

    def to_tiles3d_dict(self) -> Dict[str: List[float]]:
        return {'region', list(self.bounds)}