from typing import types, TYPE_CHECKING

from pyproj import CRS

from .crs.geocrs import GeoCrs

DEFAULT_CRS = GeoCrs(crs=CRS.from_epsg(4326))
TILE3D_CRS = GeoCrs.from_epsg(4978)

def get_default_crs() -> GeoCrs:
    global DEFAULT_CRS
    return DEFAULT_CRS

def get_geoviews() -> types.ModuleType('geoviews'):
    import geoviews as gv
    gv.extension('bokeh')
    return gv

back_map_type = object
if TYPE_CHECKING:
    back_map_type = get_geoviews().element.geo.WMTS
def get_geoviews_back_map() -> back_map_type:
     return get_geoviews().tile_sources.OSM