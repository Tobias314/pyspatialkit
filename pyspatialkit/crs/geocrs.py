from typing import Union, Optional

from pyproj import CRS

class GeoCrs:
    def __init__(self, crs: Optional[Union[CRS, 'GeoCrs']] = None) -> None:
        if isinstance(crs, CRS):
            self._proj_crs = crs
        elif isinstance(crs, GeoCrs):
            self._proj_crs = crs.proj_crs
        else:
            self._proj_crs = None

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
        res['proj_crs'] = str(self._proj_crs)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, GeoCrs):
            return self._proj_crs == other._proj_crs
        return False

    @property
    def proj_crs(self) -> CRS:
        return self._proj_crs

    @proj_crs.setter
    def set_proj_crs(self, proj_crs: CRS):
        self._proj_crs = proj_crs

class NoneCRS(GeoCrs):
    def __init__(self) -> None:
        super().__init__(crs=None)

    @property
    def proj_crs(self) -> CRS:
        raise TypeError("NoneCRS is a placeholder for a none existing crs.")

    @proj_crs.setter
    def set_proj_crs(self, proj_crs: CRS):
         raise TypeError("NoneCRS is a placeholder for a none existing crs.")
