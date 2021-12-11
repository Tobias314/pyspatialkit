from pyproj import CRS

from .crs.geocrs import GeoCrs
import geoviews as gv

DEFAULT_CRS = GeoCrs(crs= CRS.from_epsg(4326))

GEOVIEWS_BACK_MAP = gv.tile_sources.OSM