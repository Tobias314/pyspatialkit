from typing import Union, Optional, Dict

from numpy.lib.arraysetops import isin
from pyproj import CRS
import rasterio as rio
import sentinelhub

class GeoCrs:
    def __init__(self, crs: Optional[Union[CRS, 'GeoCrs', str, sentinelhub.CRS]] = None) -> None:
        try:
            if isinstance(crs, CRS):
                self._proj_crs = crs
            elif isinstance(crs, GeoCrs):
                self._proj_crs = crs.proj_crs
            elif isinstance(crs, str) or isinstance(crs, rio.crs.CRS):
                self._proj_crs = CRS(crs)
            elif isinstance(crs, sentinelhub.CRS):
                self._proj_crs = crs.pyproj_crs()
            else:
                self._proj_crs = None
        except:
            self._proj_crs = None

    @property
    def is_geocentric(self) -> bool:
        return self._proj_crs.is_geocentric

    @classmethod
    def from_str(cls, string: str):
        proj_crs = None
        try:
            proj_crs = CRS(string)
        except:
            pass
        return GeoCrs(proj_crs)

    @classmethod
    def from_dict(cls, dict):
        return cls.from_str(dict['proj_crs'])

    @classmethod
    def from_epsg(cls, epsg_code: int):
        return GeoCrs(CRS.from_epsg(epsg_code))

    def to_dict(self) -> Dict:
        res = {}
        res['proj_crs'] = self.to_str()
        return res

    def to_str(self) -> str:
        return str(self._proj_crs)

    def to_pyproj_crs(self):
        return self.proj_crs

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

    def __str__(self)-> str:
        return self.to_str()

    def __repr__(self)-> str:
        return str(self) 

class NoneCRS(GeoCrs):
    def __init__(self) -> None:
        super().__init__(crs=None)

    @property
    def is_geocentric(self) -> bool:
        return False

    @property
    def proj_crs(self) -> CRS:
        raise TypeError("NoneCRS is a placeholder for a none existing crs.")

    @proj_crs.setter
    def set_proj_crs(self, proj_crs: CRS):
         raise TypeError("NoneCRS is a placeholder for a none existing crs.")
