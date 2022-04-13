from ctypes import ArgumentError
from typing import Tuple, List, Optional, Union

import numpy as np

from .geotiler2d import GeoTiler2d
from ..dataobjects.geoshape import GeoShape
from ..spacedescriptors.georect import GeoRect
from ..crs.geocrs import GeoCrs
from ..utils.tiling import raster_bounds2d, bounds_from_polygon
from ..spacedescriptors.geobox2d import GeoBox2d
from ..spacedescriptors.collections.geoboxcollection import GeoBoxCollection


class GeoBoxTiler2d(GeoTiler2d):
    def __init__(self, aoi: Union[GeoShape, GeoBox2d, GeoRect], raster_size:Tuple[float, float], reference_crs: GeoCrs, border_size:Tuple[float, float]=(0,0)):
        super().__init__(reference_crs)
        self.aoi = aoi
        if isinstance(self.aoi, (GeoBox2d, GeoRect)):
            self.aoi = self.aoi.to_geoshape()
        if(self.aoi.crs != reference_crs):
            self.aoi.to_crs(reference_crs)
        self.raster_size = np.array(raster_size)
        self.border_size = np.array(border_size)
        if (self.raster_size < 2 * self.border_size).any():
            raise ArgumentError("Border size cannot be larger than half of raster size!")
        self._tiles: Optional[GeoBoxCollection] = None

    def partition(self, num_partitions):
        return self.tiles.partition(num_partitions)

    def __iter__(self) -> 'GeoBoxTile2dIterator':
        return GeoBoxTile2dIterator(self)

    def __len__(self) -> int:
        return len(self.tiles)

    def _generate_tiles(self) -> GeoBoxCollection:
        bounds = bounds_from_polygon(self.aoi)
        tiles = []

        origins = raster_bounds2d(bounds, self.raster_size, border_size=self.border_size)
        for origin in origins:
            #tile = GeoRect(origin - self.border_size, origin + self.raster_size + self.border_size, crs=self.reference_crs)
            tile = GeoBox2d(origin, origin + self.raster_size + 2 * self.border_size, crs=self.reference_crs)
            if self.aoi.shape.intersects(tile.to_georect().to_shapely()):
                tiles.append(tile)
        tiles = GeoBoxCollection.from_geobox_list(tiles)
        return tiles

    def get_all_tiles(self) -> GeoBoxCollection:
        return self.tiles

    @property
    def tiles(self):
        if self._tiles is None:
            self._tiles = self._generate_tiles()
        return self._tiles

class GeoBoxTile2dIterator:

    def __init__(self, geoboxtiler2d: GeoBoxTiler2d):
        self.tiler = geoboxtiler2d
        self.current = 0

    def __iter__(self) -> 'GeoBoxTile2dIterator':
        return self

    def __next__(self) -> GeoBox2d:
        if self.current >= len(self.tiler.tiles):
            raise StopIteration
        tmp = self.tiler.tiles[self.current]
        self.current+=1
        return tmp