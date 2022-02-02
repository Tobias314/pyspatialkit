from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod

import numpy as np

from ....dataobjects.tiles3d.tileset3d import Tileset3d
from ..tile3d import RefinementType, Tile3d
from ....spacedescriptors.geobox3d import GeoBox3d
from ..tiles3dcontentobject import Tiles3dContentType
from ..tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType, TILES3D_CONTENT_TYPE_TO_FILE_ENDING

if TYPE_CHECKING:
    from . import geopointcloudtileset3d

class GeoPointCloudTileIdentifier(NamedTuple):
    level: int
    tile_indices: Tuple[int, int ,int]

class GeoPointCloudTile3d(Tile3d):
    def __init__(self, tileset: 'geopointcloudtileset3d.GeoPointCloudTileset3d',
                  identifier: GeoPointCloudTileIdentifier):
        self._reset_cached()
        self.tileset = tileset
        self.tile_identifier = identifier
        self.bbox = self.tileset.get_bbox_from_identifier(self.identifier)

    def get_bounding_volume(self) -> GeoBox3d:
        return self.bbox

    def get_geometric_error(self) -> float:
        return 0

    def get_refine(self) -> RefinementType:
        return RefinementType.REPLACE        

    def get_content(self) -> Tiles3dContentObject:
        if self.level == 0:
            pcl = self.tileset.point_cloud.get_data(self.bbox)
            pcl.to_crs(crs_transformer=self.tileset.to_epsg4978_transformer, inplace=True)
            return pcl
        else:
            return None

    def get_content_type(self) -> Tiles3dContentType:
        return Tiles3dContentType.POINT_CLOUD

    def get_children(self) -> List['Tile3d']:
        children = []
        if self.level == 0:
            return children
        min_idx = np.array(self.tile_identifier.tile_indices) * 2
        tile_size = self.tileset.get_tile_size_for_level(self.level - 1)
        for i in range(min_idx[0], min_idx[0] + self.tileset.num_tiles_per_level_edge[0]):
            if self.tileset.min[0] + i * tile_size[0] > self.tileset.max[0]:
                continue
            for j in range(min_idx[1], min_idx[1] + self.tileset.num_tiles_per_level_edge[1]):
                if self.tileset.min[1] + j * tile_size[1] > self.tileset.max[1]:
                    continue
                for k in range(min_idx[2], min_idx[2] + self.tileset.num_tiles_per_level_edge[2]):
                    if self.tileset.min[2] + k * tile_size[2] > self.tileset.max[2]:
                        continue
                    children.append(self.tileset.get_tile_by_identifier(GeoPointCloudTileIdentifier(self.level-1, (i,j,k))))
        return children

    def get_identifier(self) -> GeoPointCloudTileIdentifier:
        return self.tile_identifier

    def get_cost(self) -> float:
        return 1

    @property
    def level(self)-> int:
        return self.tile_identifier.level