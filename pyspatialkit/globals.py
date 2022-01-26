from pyproj import CRS

from .crs.geocrs import GeoCrs
import geoviews as gv
gv.extension('bokeh')

DEFAULT_CRS = GeoCrs(crs= CRS.from_epsg(4326))

GEOVIEWS_BACK_MAP = gv.tile_sources.OSM

def get_default_crs() -> GeoCrs:
    global DEFAULT_CRS
    return DEFAULT_CRS