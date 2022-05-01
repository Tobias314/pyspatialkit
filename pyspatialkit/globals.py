from typing import types, TYPE_CHECKING

from pyproj import CRS
import tiledb as tdb
import loky

from .crs.geocrs import GeoCrs

TILE3D_CRS = GeoCrs.from_epsg(4978)

DEFAULT_CRS = GeoCrs(crs= CRS.from_epsg(4326))

LOKY_START_METHOD_SET = False
if not LOKY_START_METHOD_SET:
    loky.backend.context.set_start_method('spawn')
    LOKY_START_METHOD_SET = True

def configure_tiledb():
    config = tdb.Config()
    config["sm.consolidation.step_min_frags"] = 2
    config["sm.consolidation.step_max_frags"] = 20
    config["sm.consolidation.steps"] = 100
    #config["sm.consolidation.step_size_ratio"] = 0.25
    tdb.default_ctx(config=config)

configure_tiledb()


def get_geoviews() -> types.ModuleType('geoviews'):
    import geoviews as gv
    gv.extension('bokeh')
    return gv

back_map_type = object
if TYPE_CHECKING:
    back_map_type = get_geoviews().element.geo.WMTS
def get_geoviews_back_map() -> back_map_type:
     return get_geoviews().tile_sources.OSM

def get_default_crs() -> GeoCrs:
    global DEFAULT_CRS
    return DEFAULT_CRS

def get_process_pool_idle_timeout()->float:
    return 600