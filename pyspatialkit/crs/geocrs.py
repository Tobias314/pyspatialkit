from typing import Union, Optional

from pyproj import CRS

class GeoCrs:
    def __init__(self, crs: Optional[Union[CRS, 'GeoCrs']] = None) -> None:
        if isinstance(crs, CRS):
            self.proj_crs = crs
        elif isinstance(crs, GeoCrs):
            self.proj_crs = crs.proj_crs
        else:
            self.proj_crs = None

    @classmethod
    def from_dict(cls, dict):
        proj_crs = None
        try:
            proj_crs = CRS(dict['proj_crs'])
        except:
            pass
        return GeoCrs(proj_crs)

    def to_dict(self):
        res = {}
        res['proj_crs'] = str(self.proj_crs)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, GeoCrs):
            return self.proj_crs == other.proj_crs
        return False

class NoneCRS(GeoCrs):
    def __init__(self) -> None:
        super().__init__(crs=None)

    @property
    def proj_crs(self):
        raise TypeError("NoneCRS is a placeholder for a none existing crs.")
