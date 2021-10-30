from pyproj import CRS

from .crs.geocrs import GeoCRS

DEFAULT_CRS = GeoCRS(crs= CRS.from_epsg(4326))