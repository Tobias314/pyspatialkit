from typing import List

from abc import ABC, abstractmethod

import geoviews as gv
import geopandas as gpd
from pyproj import CRS

from ..crs.geocrs import GeoCrs
from ..spacedescriptors.georect import GeoRect
from ..globals import GEOVIEWS_BACK_MAP



class GeoTiler2d(ABC):

    def __init__(self, reference_crs: GeoCrs):
        self.reference_crs = reference_crs

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def get_all_tiles(self, number) -> List[GeoRect]:
        print(number)
        tiles = []
        for tile in self:
            tiles.append(tile)
        return None

    def to_geopandas(self, as_boundary_lines: bool=False) -> gpd.GeoDataFrame:
        geoms: List[GeoRect] = []
        for tile in self.get_all_tiles():
            if as_boundary_lines:
                geoms.append(tile.to_shapely().boundary)
            else:
                geoms.append(tile.to_shapely())
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=self.reference_crs.to_pyproj_crs())
        return gdf

    def to_geoviews(self, filled:bool=False):
        gdf = self.to_geopandas().to_crs(CRS.from_epsg(4326))
        if filled:
            return gv.Polygons(gdf)
        else:
            return gv.Path(gdf)

    def plot(self, filled:bool=False):
        return GEOVIEWS_BACK_MAP * self.to_geoviews(filled=filled)
