from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod

import numpy as np

from ....dataobjects.tiles3d.tileset3d import Tileset3d
from ..tile3d import RefinementType, Tile3d
from ....spacedescriptors.geobox3d import GeoBox3d
from ..tiles3dcontentobject import Tiles3dContentType
from ..tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType, TILES3D_CONTENT_TYPE_TO_FILE_ENDING

if TYPE_CHECKING:
    from . import bboxstoragetileset3d

class BBoxStorageTileIdentifier(NamedTuple):
    level: int
    tile_indices: Tuple[int, int ,int]
    object_id: int = -1

    def __str__(self):
        return str("{}_{}_{}_{}_{}.json".format(self.level, self.tile_indices[0], self.tile_indices[1], self.tile_indices[2], self.object_id))

class BBoxStorageTile3d(Tile3d):
    def __init__(self, tileset: 'bboxstoragetileset3d.BBoxStorageTileSet3d',
                  identifier: BBoxStorageTileIdentifier, geometric_error=0):
        self._reset_cached()
        self.tileset = tileset
        self.bboxstorage = self.tileset.bboxstorage
        self.tile_identifier = identifier
        self.bbox = self.tileset.get_bbox_from_identifier(self.identifier)
        self._geometric_error = geometric_error

    def get_bounding_volume(self) -> GeoBox3d:
        return self.bbox

    def get_geometric_error(self) -> float:
        return self._geometric_error

    def get_refine(self) -> RefinementType:
        return RefinementType.REPLACE        

    def get_content(self) -> Tiles3dContentObject:
        #TODO: method and whole class not valid at the moment. Continue work here
        if self.tile_identifier.object_id != -1:
            content = self.bboxstorage.get_object_for_identifier(tile_indices=self.tile_identifier.tile_indices,
                                                                 object_identifier=self.tile_identifier.object_id)
            return content
        else:
            return None
    
    def content_to_bytes(self, content: Tiles3dContentObject) -> bytes:
        return content.to_bytes_tiles3d(transformer=self.tileset.to_epsg4978_transformer)

    def get_content_type(self) -> Tiles3dContentType:
        return self.tileset.content_type

    def get_children(self) -> List['Tile3d']:
        children = []
        if self.tile_identifier.object_id != -1:
            return children
        if self.tile_identifier.level > 0:
            min_idx = np.array(self.tile_identifier.tile_indices) * 2
            tile_size = self.tileset.get_tile_size_for_level(self.level - 1)
            for i in range(min_idx[0], min_idx[0] + self.tileset.num_tiles_per_level_edge[0]):
                if self.tileset.min[0] + i * tile_size[0] >= self.tileset.max[0]:
                    continue
                for j in range(min_idx[1], min_idx[1] + self.tileset.num_tiles_per_level_edge[1]):
                    if self.tileset.min[1] + j * tile_size[1] >= self.tileset.max[1]:
                        continue
                    for k in range(min_idx[2], min_idx[2] + self.tileset.num_tiles_per_level_edge[2]):
                        if self.tileset.min[2] + k * tile_size[2] >= self.tileset.max[2]:
                            continue
                        children.append(self.tileset.get_tile_by_identifier(BBoxStorageTileIdentifier(self.level-1, (i,j,k), -1)))
        if self.tile_identifier.object_id == -1 and (self.tile_identifier.level == 0 or self.tileset.has_pyramid):
            object_ids = self.bboxstorage.get_object_ids_for_tile(tile_indices=self.tile_identifier.tile_indices)
            for object_id in object_ids:
                children.append(self.tileset.get_tile_by_identifier(BBoxStorageTileIdentifier(self.level, self.tile_identifier.tile_indices,
                                                                                              object_id)))
        return children

    def get_identifier(self) -> BBoxStorageTileIdentifier:
        return self.tile_identifier

    def get_cost(self) -> float:
        return 1

    @property
    def level(self)-> int:
        return self.tile_identifier.level