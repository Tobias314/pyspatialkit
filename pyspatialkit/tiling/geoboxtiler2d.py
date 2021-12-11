from ctypes import ArgumentError
from typing import Tuple, List

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, MultiPoint

from .geotiler2d import GeoTiler2d
from ..dataobjects.geoshape import GeoShape
from ..spacedescriptors.georect import GeoRect
from ..crs.geocrs import GeoCrs


class GeoBoxTiler2d(GeoTiler2d):
    def __init__(self, aoi: GeoShape, raster_size:Tuple[float, float], reference_crs: GeoCrs, border_size:Tuple[float, float]=(0,0)):
        super().__init__(reference_crs)
        self.aoi = aoi
        if(self.aoi.crs != reference_crs):
            self.aoi.to_crs(reference_crs)
        self.raster_size = np.array(raster_size)
        self.border_size = np.array(border_size)
        if (self.raster_size < 2 * self.border_size).any():
            raise ArgumentError("Border size cannot be larger than half of raster size!")
        self._tiles = None

    def __iter__(self) -> 'GeoBoxTile2dIterator':
        return GeoBoxTile2dIterator(self)

    def _generate_tiles(self) -> List[GeoRect]:
        bounds = []
        if isinstance(self.aoi, (MultiLineString, MultiPolygon, MultiPoint)):
            for geom in self.aoi:
                bounds.append(geom.bounds)
        else:
            bounds = [self.aoi.bounds]
        tiles = []
        inner_size = self.raster_size - 2 * self.border_size
        for min_x, min_y, max_x, max_y in bounds:
            xs = np.arange(min_x, max_x, inner_size[0])
            ys = np.arange(min_y, max_y, inner_size[1])
            origins = np.stack(np.meshgrid(xs, ys, indexing='ij'),axis=2).reshape((-1,2))
            for origin in origins:
                tile = GeoRect(origin - self.border_size, origin + self.raster_size + self.border_size, crs=self.reference_crs)
                if self.aoi.shape.intersects(tile.to_shapely()):
                    tiles.append(tile)
        return tiles

    def get_all_tiles(self):
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

    def __next__(self) -> GeoRect:
        if self.current >= len(self.tiler.tiles):
            raise StopIteration
        tmp = self.tiler.tiles[self.current]
        self.current+=1
        return tmp