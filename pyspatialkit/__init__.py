from pyproj import CRS

from .crs.geocrs import GeoCrs

DEFAULT_CRS = GeoCrs(crs= CRS.from_epsg(4326))