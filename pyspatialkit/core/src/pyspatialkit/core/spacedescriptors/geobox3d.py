from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer
from .tiles3dboundingvolume import Tiles3dBoundingVolume
from ..globals import get_default_crs
from .geobox2d import GeoBox2d

CRS_4979 = GeoCrs.from_epsg(4979)
CRS_4978 = GeoCrs.from_epsg(4978)

class GeoBox3d(Tiles3dBoundingVolume):
    """Describing a Volume in 3D space. Described by a min and max point and a crs.
    """

    def __init__(self, min: Tuple[float, float, float], max: Tuple[float, float ,float], crs: Optional[GeoCrs] = None,
                  to_epsg4978_transformer: Optional[GeoCrsTransformer] = None, to_epsg4979_transformer: Optional[GeoCrsTransformer] = None):
        if crs is None:
            crs = get_default_crs()
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
    def from_bounds(cls, bounds: Tuple[float, float, float, float, float, float], crs: Optional[GeoCrs] = None,
                     to_epsg4978_transformer: Optional[GeoCrsTransformer] = None,
                     to_epsg4979_transformer: Optional[GeoCrsTransformer] = None):
        return GeoBox3d(min=bounds[:3], max=bounds[3:], crs=crs, to_epsg4978_transformer=to_epsg4978_transformer,
                        to_epsg4979_transformer=to_epsg4979_transformer)

    @classmethod 
    def from_geobox2d(cls, geobox: GeoBox2d, min_height:float, max_height: float,
                      to_epsg4978_transformer: Optional[GeoCrsTransformer] = None,
                      to_epsg4979_transformer: Optional[GeoCrsTransformer] = None):
        bounds = [*geobox.min, min_height, *geobox.max, max_height]
        return cls.from_bounds(bounds, crs=geobox.crs, to_epsg4978_transformer=to_epsg4978_transformer, to_epsg4979_transformer=to_epsg4979_transformer)

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

    def substract_borders(self, border_size: Tuple[float, float, float]) -> 'GeoBox3d':
        border_size = np.array(border_size)
        return GeoBox3d(self.bounds[:3] + border_size, self.bounds[3:] - border_size, self.crs)

    def to_tiles3d_bounding_volume_dict(self) -> Dict[str, List[float]]:
        if not self.crs.is_geocentric:
            min_pt = self.min
            max_pt = self.max
            x = np.array([min_pt[0], max_pt[0]])
            y = np.array([min_pt[1], max_pt[1]])
            z = np.array([min_pt[2], max_pt[2]])
            x,y,z = np.meshgrid(x,y,z, indexing='ij')
            x,y,z = x.flatten(), y.flatten(), z.flatten()
            x,y,z = np.array(self.to_epsg4979_transformer.transform(x,y,z))
            pts = np.stack([x,y,z], axis=1)
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            min_pt[:2] = np.deg2rad(min_pt[:2])
            max_pt[:2] = np.deg2rad(max_pt[:2])
            return {'region': [min_pt[0], min_pt[1], max_pt[0], max_pt[1], min_pt[2], max_pt[2]]}
        else:
            center = self.max - self.min
            edge_half_length = (max_pt - min_pt) / 2
            right = center + np.array([1,0,0]) * edge_half_length[0]
            forward = center + np.array([0,1,0]) * edge_half_length[1]
            up = center + np.array([0,0,1]) * edge_half_length[2]
            pts = np.array([center, right, forward, up])
            pts = np.stack(self.to_epsg4978_transformer.transform(pts[:,0], pts[:,1], pts[:,2]), axis=1)
            res_vec = [*pts[0], *(pts[1]- pts[0]), *(pts[2]- pts[0]), *(pts[3]- pts[0])]
            return {'box': res_vec}

    def __str__(self)-> str:
        return "GeoBox3d(min: {}, max: {})".format(list(self.min), list(self.max))

    def __repr__(self)-> str:
        return str(self) 