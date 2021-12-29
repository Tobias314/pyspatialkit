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

CRS_4979 = GeoCrs.from_epsg(4979)
CRS_4978 = GeoCrs.from_epsg(4978)

class GeoBox3d(Tiles3dBoundingVolume):
    """Describing a Volume in 3D space. Described by a min and max point and a crs.
    """

    def __init__(self, min: Tuple[float, float, float], max: Tuple[float, float ,float], crs: GeoCrs = DEFAULT_CRS,
                  to_epsg4978_transformer: Optional[GeoCrsTransformer] = None, to_epsg4979_transformer: Optional[GeoCrsTransformer] = None):
        self.crs = crs
        if to_epsg4978_transformer is not None:
            if not (to_epsg4978_transformer.from_crs == crs and to_epsg4978_transformer.to_crs == CRS_4978):
                raise AttributeError("to_epsg4978 transformer attribute needs to be None or a crs transformer from the geobox3d crs to epsg 4978!")
        self._to_epsg4978_transformer = to_epsg4978_transformer
        if to_epsg4979_transformer is not None:
            if not (to_epsg4979_transformer.from_crs == crs and to_epsg4979_transformer.to_crs == CRS_4979):
                raise AttributeError("to_epsg4997 transformer attribute needs to be None or a crs transformer from the geobox3d crs to epsg 4979!")
        self._to_epsg4979_transformer = to_epsg4979_transformer
        assert len(min)==3 and len(max)==3
        self.bounds = np.array([*min, *max])

    @classmethod
    def from_bounds(cls, bounds: Tuple[float, float, float, float, float, float], crs: GeoCrs = DEFAULT_CRS,
                     to_epsg4978_transformer: Optional[GeoCrsTransformer] = None,
                     to_epsg4979_transformer: Optional[GeoCrsTransformer] = None):
        return GeoBox3d(min=bounds[:3], max=bounds[3:], crs=crs, to_epsg4978_transformer=to_epsg4978_transformer,
                        to_epsg4979_transformer=to_epsg4979_transformer)

    @property
    def min(self):
        return self.bounds[:3]

    @property
    def max(self):
        return self.bounds[3:]

    @property
    def center(self):
        return (self.max + self.min) / 2

    @property
    def to_epsg4978_transformer(self):
        if self._to_epsg4978_transformer is None:
            self._to_epsg4978_transformer = GeoCrsTransformer(self.crs, CRS_4978)
        return self._to_epsg4978_transformer

    @property
    def to_epsg4979_transformer(self):
        if self._to_epsg4979_transformer is None:
            self._to_epsg4979_transformer = GeoCrsTransformer(self.crs, CRS_4979)
        return self._to_epsg4979_transformer

    def get_edge_lengths(self) -> np.ndarray:
        return self.max - self.min

    def to_tiles3d_bounding_volume_dict(self) -> Dict[str, List[float]]:
        if not self.crs.is_geocentric:
            min_pt = self.min
            min_pt = np.array(self.to_epsg4979_transformer.transform(min_pt[0], min_pt[1], min_pt[2]))
            max_pt = self.max
            max_pt = np.array(self.to_epsg4979_transformer.transform(max_pt[0], max_pt[1], max_pt[2]))
            return {'region': [*np.minimum(min_pt, max_pt), *np.maximum(min_pt, max_pt)]}
        else:
            min_pt = self.min
            min_pt = np.array(self.to_epsg4978_transformer.transform(min_pt[0], min_pt[1], min_pt[2]))
            max_pt = self.max
            max_pt = np.array(self.to_epsg4978_transformer.transform(max_pt[0], max_pt[1], max_pt[2]))
            min_pt, max_pt = np.minimum(min_pt, max_pt), np.maximum(min_pt, max_pt)
            edge_half_length = (max_pt - min_pt) / 2
            x_vec = np.array([1,0,0]) * edge_half_length[0]
            y_vec = np.array([0,1,0]) * edge_half_length[1]
            z_vec = np.array([0,0,1]) * edge_half_length[2]
            res_vec = [*((min_pt + max_pt) / 2), *x_vec, *y_vec, *z_vec]
            return {'box': res_vec}