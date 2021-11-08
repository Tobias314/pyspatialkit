from pyproj import Transformer, crs

from typing import Any, Tuple

from .geocrs import GeoCrs, NoneCRS

class GeoCrsTransformer:

    def __init__(self, from_crs: GeoCrs, to_crs: GeoCrs, always_xy=True) -> None:
        if isinstance(from_crs, NoneCRS) or isinstance(to_crs, NoneCRS):
            raise AttributeError("A Transformation cannot be created from a NoneCrs!")
        self.from_crs = from_crs
        self.to_crs = to_crs
        self.always_xy = always_xy
        self.proj_transformer = Transformer.from_crs(crs_from=self.from_crs.proj_crs, crs_to=self.to_crs.proj_crs, always_xy=always_xy)

    def transform(self, xx: Any, yy: Any) -> Tuple[Any, Any]:
        return self.proj_transformer.transform(xx, yy)
