from typing import Optional, Tuple, Union, List
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon

from ..utils.linalg import projective_transform_from_pts

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer


class GeoRect:

    def __init__(self, bottom_left: Tuple[float, float],  top_right: Tuple[float, float],
                 bottom_right: Optional[Tuple[float, float]] = None, top_left: Optional[Tuple[float, float]] = None,
                 crs: GeoCrs = NoneCRS):
        self.crs = crs
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        if top_left is None or bottom_right is None:
            self.top_left = np.array((bottom_left[0], top_right[1]))
            self.bottom_right = np.array((top_right[0], bottom_left[1]))
        else:
            self.top_left = np.array(top_left)
            self.bottom_right = np.array(bottom_right)
        self._create_cache()

    @classmethod
    def from_points(points: Union[List[Tuple[float,float]], np.ndarray], crs: GeoCrs) -> 'GeoRect':
        return GeoRect(points[0], points[2], points[1], points[3], crs=crs)

    def from_bounds(bounds: Tuple[float,float,float,float], crs: GeoCrs) -> 'GeoRect':
        return GeoRect(bounds[:2], bounds[2:], crs=crs)

    def _create_cache(self, points: Optional[Union[List[Tuple[float,float]], np.ndarray]] = None):
        if points is not None:
            self.bottom_left = np.array(points[0])
            self.bottom_right = np.array(points[1])
            self.top_right = np.array(points[2])
            self.top_left = np.array(points[3])
        if ((self.bottom_left - self.bottom_right)[1] == 0 and (self.top_left - self.top_right)[1]==0 
         and (self.bottom_left - self.top_left)[0] == 0 and (self.bottom_right - self.top_right)[0]==0):
            self.is_axis_aligned = True
        else:
            self.is_axis_aligned = False
        self._transform = projective_transform_from_pts(source_pts=np.array([[0,0],[1,0],[1,1],[0,1]]), destination_pts=np.array(self.points))

    def to_shapely(self) -> Polygon:
        return Polygon([self.top_left, self.top_right, self.bottom_right, self.bottom_left])

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace=True) -> 'GeoRect':
        if crs_transformer is None:
            crs_transformer = GeoCrsTransformer(self.crs, new_crs)
        xx, yy = crs_transformer.transform(self.xx, self.yy)
        points = np.stack([xx,yy], axis=1)
        if inplace:
            self._create_cache(points=points)
            self.crs = new_crs
            return self
        else:
            return GeoRect.from_points(points, crs=new_crs)

    def get_bounds(self):
        return self.to_shapely().bounds

    @property
    def transform(self):
        return self._transform

    @property
    def points(self):
        return np.array([self.bottom_left, self.bottom_right, self.top_right, self.top_left])

    @property
    def xx(self):
        return self.points[:, 0]

    @property
    def yy(self):
        return self.points[:, 1]