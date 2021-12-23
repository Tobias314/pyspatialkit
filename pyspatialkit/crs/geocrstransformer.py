from typing import Any, Tuple, Optional, Union

import pyproj
from pyproj import Transformer
from pyproj.enums import TransformDirection as TransformDirection
from shapely.ops import transform as shapely_transform

from .geocrs import GeoCrs, NoneCRS
from ..utils.shapely import SHAPELY_GEOMETRY_TYPE

class GeoCrsTransformer:

    def __init__(self, from_crs: GeoCrs, to_crs: GeoCrs, always_xy=True) -> None:
        if isinstance(from_crs, NoneCRS) or isinstance(to_crs, NoneCRS):
            raise AttributeError("A Transformation cannot be created from a NoneCrs!")
        self.from_crs = from_crs
        self.to_crs = to_crs
        self.always_xy = always_xy
        self.proj_transformer = Transformer.from_crs(crs_from=self.from_crs.proj_crs, crs_to=self.to_crs.proj_crs, always_xy=always_xy)

    def transform(self, xx: Any, yy: Any, zz: Optional[Any] = None, 
                   direction: Union[TransformDirection, str] = 'TransformDirection.FORWARD') -> Tuple[Any, Any]:
        return self.proj_transformer.transform(xx, yy, zz)

    def transform_tuple(self, tup: Union[Tuple[float, float], Tuple[float, float, float]], 
                         direction: Union[TransformDirection, str] = 'TransformDirection.FORWARD') -> Union[Tuple[float, float], Tuple[float, float, float]]:
        if len(tup) == 2:
            return self.transform(tup[0], tup[1])
        else:
            return self.transform(tup[0], tup[1], tup[2])

    def transform_shapely_shape(self, shape: SHAPELY_GEOMETRY_TYPE, 
                                 direction: Union[pyproj.enums.TransformDirection, str] = 'TransformDirection.FORWARD') -> SHAPELY_GEOMETRY_TYPE:
        return shapely_transform(self.proj_transformer.transform, shape)
