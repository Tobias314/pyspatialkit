from typing import Optional, Tuple, Union, List

import numpy as np
from shapely.geometry import Polygon
import sentinelhub

from ..utils.linalg import projective_transform_from_pts

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer
from ..dataobjects.geoshape import GeoShape


class GeoRect:

    def __init__(self, bottom_left: Tuple[float, float],  top_right: Tuple[float, float],
                 bottom_right: Optional[Tuple[float, float]] = None, top_left: Optional[Tuple[float, float]] = None,
                 crs: GeoCrs = NoneCRS):
        if crs.is_geocentric:
            raise AttributeError('CRS for 2d GeoRect cannot be geocentric!')
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
    def from_points(cls, points: Union[List[Tuple[float,float]], np.ndarray], crs: GeoCrs) -> 'GeoRect':
        return GeoRect(points[0], points[2], points[1], points[3], crs=crs)

    @classmethod
    def from_bounds(cls, bounds: Union[Tuple[float,float,float,float],  Tuple[float,float,float,float,float, float]], crs: GeoCrs) -> 'GeoRect':
        if len(bounds) == 4:
            return GeoRect(bounds[:2], bounds[2:], crs=crs)
        else:
            return GeoRect(bounds[:2], bounds[3:5], crs=crs)

    @classmethod
    def from_min_max(cls, min_pt: Tuple[float, float], max_pt: Tuple[float, float], crs: GeoCrs)-> 'GeoRect':
        return GeoRect(min_pt, max_pt, crs=crs)

    @classmethod
    def from_sentinelhub_bbox(cls, sentinelhub_bbox: sentinelhub.BBox):
        return GeoRect(sentinelhub_bbox.lower_left, sentinelhub_bbox.upper_right, crs=GeoCrs(sentinelhub_bbox.crs))

    def copy(self):
        return GeoRect(bottom_left=self.bottom_left, top_right=self.top_right, bottom_right=self.bottom_right,  top_left=self.top_left, crs=self.crs)

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
        self._transform = None
        self._bounds = None

    def to_shapely(self) -> Polygon:
        return Polygon([self.top_left, self.top_right, self.bottom_right, self.bottom_left])

    def to_geoshape(self) -> GeoShape:
        return GeoShape(self.to_shapely(), self.crs)

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
    def bounds(self):
        if self._bounds is None:
            self._bounds = self.get_bounds()
        return self._bounds

    @property
    def transform(self):
        if self._transform is None:
            self._transform = projective_transform_from_pts(source_pts=np.array([[0,0],[1,0],[1,1],[0,1]]), destination_pts=np.array(self.points))
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