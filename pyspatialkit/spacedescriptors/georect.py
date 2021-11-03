from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon

from ..utils.linalg import projective_transform_from_pts

from ..crs.geocrs import GeoCRS, NoneCRS


class GeoRect:

    def __init__(self, top_left: Tuple[float, float],  bottom_right: Tuple[float, float],
                 top_right: Optional[Tuple[float, float]] = None, bottom_left: Optional[Tuple[float, float]] = None,
                 crs: GeoCRS = NoneCRS):
        self.crs = crs
        self.top_left = top_left
        self.bottom_right = bottom_right
        if top_right is None or bottom_left is None:
            self.top_right = (bottom_right[0], top_left[1])
            self.bottom_left = (top_left[0], bottom_right[1])
        else:
            self.top_right = top_right
            self.bottom_left = bottom_left
        self._transform = projective_transform_from_pts(source_pts=np.array([[0,1],[1,1],[1,0],[0,0]]), destenation_pts=np.array(self.points))

    def to_shapely(self) -> Polygon:
        return Polygon([self.top_left, self.top_right, self.bottom_right, self.bottom_left])

    def get_bounds(self):
        return self.to_shapely().bounds

    @property
    def transform(self):
        return self._transform

    @property
    def points(self):
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]