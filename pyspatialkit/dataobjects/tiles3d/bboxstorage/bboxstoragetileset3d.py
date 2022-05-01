from typing import NamedTuple, Union, Dict, Optional, TYPE_CHECKING, Tuple, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import json
import os

import numpy as np

from ....spacedescriptors.geobox3d import GeoBox3d
from ..tileset3d import Tileset3d
from .bboxstoragetile3d import BBoxStorageTile3d, BBoxStorageTileIdentifier
from ....crs.geocrs import GeoCrs
from ....crs.geocrstransformer import GeoCrsTransformer
from ..tiles3dcontentobject import TILES3D_CONTENT_TYPE_TO_FILE_ENDING

from ....storage.bboxstorage.bboxstorage import BBoxStorage
from ....utils.logging import dbg
from ..tiles3dcontentobject import Tiles3dContentObject

class BBoxStorageTileSet3d(Tileset3d):

    def __init__(self, bboxstorage: BBoxStorage, crs: GeoCrs, geometric_error_multiplier = 8, num_tiles_per_level_edge = 2):
        super().__init__()
        if not issubclass(bboxstorage.object_type, Tiles3dContentObject):
            raise TypeError("Can only create a tileset for BBoxStorage instances storing of type Tiles3dContentObject (transformable to 3dTiles!")
        self.bboxstorage = bboxstorage
        self.content_type = self.bboxstorage.object_type.get_content_type_tile3d()
        self.crs = crs
        self.geometric_error_multiplier = geometric_error_multiplier
        self.num_tiles_per_level_edge = num_tiles_per_level_edge
        if isinstance(self.num_tiles_per_level_edge, int) or isinstance(self.num_tiles_per_level_edge, float):
            self.num_tiles_per_level_edge = [self.num_tiles_per_level_edge] * 3
        self.num_tiles_per_level_edge = np.array(self.num_tiles_per_level_edge)
        self.to_epsg4978_transformer = GeoCrsTransformer(self.crs, GeoCrs.from_epsg(4978))
        if not self.crs.is_geocentric:
            self.to_epsg4979_transformer = GeoCrsTransformer(self.crs, GeoCrs.from_epsg(4979))
        self.min = np.array(self.bboxstorage.bounds[:3])
        self.max = np.array(self.bboxstorage.bounds[3:])
        self.extent = self.max - self.min
        self.max_level = int(np.max(np.ceil(np.log(self.extent / self.tile_size) / np.log(self.num_tiles_per_level_edge))))

    def get_tile_size_for_level(self, level: int)-> np.ndarray:
        tile_size = self.tile_size * self.num_tiles_per_level_edge**(level)
        tile_size = np.minimum(tile_size, self.extent)
        return tile_size

    @property
    def tile_size(self) -> np.ndarray:
        return self.bboxstorage.tile_size

    @property
    def tileset_version(self) -> Union[float, str]:
        return 1.0

    @property
    def properties(self) -> Dict[str, Dict[str, float]]:
        return {}

    @property
    def geometric_error(self) -> float:
        return self.geometric_error_multiplier #* (self.max_level + 1)

    @property
    def has_pyramid(self) -> bool:
        return self.bboxstorage.has_pyramid

    def get_root(self) -> BBoxStorageTile3d:
        return self.get_tile_by_identifier(BBoxStorageTileIdentifier(self.max_level, (0,0,0), object_id = -1))

    def get_bbox_from_identifier(self,identifier: BBoxStorageTileIdentifier) -> GeoBox3d:
        if identifier.object_id == -1:
            tile_size = self.get_tile_size_for_level(identifier.level)
            min_pt = self.min + tile_size * np.array(identifier.tile_indices)
            bbox = GeoBox3d(min_pt, min_pt+tile_size, crs=self.crs, to_epsg4978_transformer=self.to_epsg4978_transformer,
                            to_epsg4979_transformer=self.to_epsg4979_transformer)
            return bbox
        else:
            if self.bboxstorage.has_pyramid == False and identifier.level != 0:
                raise ValueError("Storage has no pyramids so only layer 0 has object bounding boxes!")
            bounds = self.bboxstorage.get_bounds_for_identifiert(tile_indices=identifier.tile_indices, object_identifier=identifier.object_id)
            return GeoBox3d.from_bounds(bounds)

    def get_tile_by_identifier(self,identifier: BBoxStorageTileIdentifier) -> BBoxStorageTile3d:
        ge = self.geometric_error_multiplier #* identifier.level
        res = BBoxStorageTile3d(self, identifier=identifier, geometric_error=ge)
        dbg('got tile')
        return res
