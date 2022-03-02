from typing import Tuple, List, Union, Optional
from abc import ABC, abstractmethod

import numpy as np

from ..dataobjects.geoshape import GeoShape
from ..spacedescriptors.geobox3d import GeoBox3d
from ..spacedescriptors.collections.geoboxcollection import GeoBoxCollection
from ..dataobjects.geoshape import GeoShape
from ..crs.geocrs import GeoCrs
from ..utils.tiling import bounds_from_polygon, raster_bounds2d, raster_bounds3d

class GeoBoxTile3dIterator:

    def __init__(self, geobox_tiler_3d: 'GeoBoxTiler3d'):
        self.tiler = geobox_tiler_3d
        self.current = 0

    @abstractmethod
    def __iter__(self) -> 'GeoBoxTile3dIterator':
        self

    @abstractmethod
    def __next__(self) -> Tuple[GeoBox3d, GeoBox3d]:
        if self.current >= len(self.tiler.tiles):
            raise StopIteration
        tmp = self.tiler.tiles[self.current]
        self.current+=1
        return tmp, tmp.substract_borders(self.tiler.border_size)


class GeoBoxTiler3d:

    def __init__(self, aoi: GeoShape, min_height: float, max_height: float,
                 raster_size: Union[Tuple[float, float], Tuple[float, float, float]],
                 reference_crs: Optional[GeoCrs] = None, border_size:Union[Tuple[float, float], Tuple[float, float, float]]=(0,0,0)):
        self._tiles: Optional[GeoBoxCollection] = None
        self.aoi = aoi
        self.raster_size = np.array(raster_size)
        self.height_span = np.array((min_height, max_height))
        if reference_crs is None:
            self.crs = self.aoi.crs
        else:
            self.crs = reference_crs
        if self.raster_size.shape == 3:
            if len(border_size) == 2:
                self.border_size = np.array([border_size[0], border_size[1], 0])
            else:
                self.border_size = np.array(border_size)
        else:
            self.border_size = np.array([border_size[0], border_size[1], 0])

    def _generate_tiles(self) -> GeoBoxCollection:
        bounds = bounds_from_polygon(self.aoi)
        if self.raster_size.shape[0] == 3:
            for i in range(len(bounds)):
                b = bounds[i]
                bounds[i] = (b[0], b[1], self.height_span[0], b[2], b[3], self.height_span[1])
        tiles = []
        inner_size = self.raster_size - 2 * self.border_size[:self.raster_size.shape[0]]
        if self.raster_size.shape[0] == 2:
            origins = raster_bounds2d(bounds, self.raster_size, border_size=self.border_size[:2])
            mins = np.concatenate((origins, np.full((origins.shape[0], 1), self.height_span[0])), axis=1)
            maxs = mins + np.array([self.raster_size[0] + 2*border_size[0], self.raster_size[1] + 2*border_size[1], self.height_span[1] - self.height_span[0]])
        else:
            mins = raster_bounds3d(bounds, self.raster_size, border_size=self.border_size)
            maxs = mins + np.array([self.raster_size[0] + 2*border_size[0], self.raster_size[1] + 2*border_size[1], self.raster_size[2] + 2*border_size[2]])
        return GeoBoxCollection.from_mins_maxs(mins, maxs, self.crs)

    def partition(self, num_tiles:int):
        return self.tiles.partition(num_tiles)

    def __iter__(self) -> GeoBoxTile3dIterator:
        return GeoBoxTile3dIterator(self)

    @property
    def tiles(self):
        if self._tiles is None:
            self._tiles = self._generate_tiles()
        return self._tiles