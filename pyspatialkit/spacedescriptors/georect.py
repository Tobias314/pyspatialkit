from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon

from ..utils.linalg import projective_transform_from_pts

from ..crs.geocrs import GeoCRS, NoneCRS


class GeoRect:

    def __init__(self, bottom_left: Tuple[float, float],  top_right: Tuple[float, float],
                 top_left: Optional[Tuple[float, float]] = None, bottom_right: Optional[Tuple[float, float]] = None,
                 crs: GeoCRS = NoneCRS):
        self.crs = crs
        self.bottom_left = bottom_left
        self.top_right = top_right
        if top_left is None or bottom_right is None:
            self.top_left = (bottom_left[0], top_right[1])
            self.bottom_right = (top_right[0], bottom_left[1])
        else:
            self.top_left = top_left
            self.bottom_right = bottom_right
        self._transform = projective_transform_from_pts(source_pts=np.array([[0,1],[1,1],[1,0],[0,0]]), destination_pts=np.array(self.points))

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