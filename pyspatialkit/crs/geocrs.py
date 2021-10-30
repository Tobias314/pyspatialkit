from typing import Union, Optional

from pyproj import CRS

class GeoCRS:
    def __init__(self, crs: Optional[Union[CRS, 'GeoCRS']] = None) -> None:
        if isinstance(crs, CRS):
            self.proj_crs = crs
        elif isinstance(crs, GeoCRS):
            self.proj_crs = crs.proj_crs
        else:
            self.proj_crs = None

class NoneCRS(GeoCRS):
    def __init__(self) -> None:
        super().__init__(crs=None)
