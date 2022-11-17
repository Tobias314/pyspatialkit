from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from pyspatialkit.core.dataobjects.tiles3d.tileset3d import Tileset3d
from pyspatialkit.core.dataobjects.tiles3d.tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType
from pyspatialkit.core.dataobjects.tiles3d.tile3d import RefinementType, Tile3d
from pyspatialkit.core.spacedescriptors.geobox3d import GeoBox3d
#from ..tiles3dcontentobject import Tiles3dContentType
#from ..tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType, TILES3D_CONTENT_TYPE_TO_FILE_ENDING

from .persistent_r_tree import PersistentRTreeNode

if TYPE_CHECKING:
    from . import geopointcloudtileset3d

class PersistentRTreeTileIdentifier(BaseModel):
    node_id: int
    object_index: Optional[int] = None

    def __str__(self):
        return f"{self.node_id}{'' if self.object_index is None else f'_{self.object_index}'}.json"

class PersistentRTreeContentTile3d(Tile3d):
    def __init__(self, tileset: '.persistent_r_tree_tileset3d.PersistentRTreeTileset3d', tree_node: PersistentRTreeNode, object_index: int):
        self.tree_node = tree_node
        self.object_index = object_index
        self.tile_identifier = PersistentRTreeTileIdentifier(node_id=tree_node.node_id, object_index=object_index)
        super().__init__(tileset=tileset)

    def get_children(self)->List['PersistentRTreeTile3d']:
        return []

    def get_bounding_volume(self) -> GeoBox3d:
        return GeoBox3d.from_bounds(self.tileset.project_bbox_to_3d(self.tree_node.object_bboxes[self.object_index]),
                                     crs=self.tileset.crs)

    def get_geometric_error(self) -> float:
        return self.tileset.bbox_to_geometric_error(self.tree_node.object_bboxes[self.object_index])

    def get_refine(self) -> RefinementType:
        return RefinementType.REPLACE        

    def get_content(self) -> Tiles3dContentObject:
        return self.tree_node.get_object_at(self.object_index)
    
    def content_to_bytes(self, content: Tiles3dContentObject) -> bytes:
        return content.to_bytes_tiles3d(transformer=self.tileset.to_epsg4978_transformer)

    def get_content_type(self) -> Tiles3dContentType:
        return self.tileset.content_type

    def get_identifier(self) -> PersistentRTreeTileIdentifier:
        return self.tile_identifier

    def get_cost(self) -> float:
        return 1


class PersistentRTreeTile3d(Tile3d):
    def __init__(self, tileset: '.persistent_r_tree_tileset3d.PersistentRTreeTileset3d', tree_node: PersistentRTreeNode):
        self.tree_node = tree_node
        self.tile_identifier = PersistentRTreeTileIdentifier(node_id=self.tree_node.node_id)
        super().__init__(tileset=tileset)

    def get_children(self)->List['PersistentRTreeTile3d']:
        res = []
        for child_node in self.tree_node.get_child_nodes():
            res.append(PersistentRTreeTile3d(tileset=self.tileset, tree_node=child_node))
        for object_index in range(len(self.tree_node.object_ids)):
            res.append(PersistentRTreeContentTile3d(tileset=self.tileset, tree_node=self.tree_node,
                                                     object_index=object_index))
        return res

    def get_bounding_volume(self) -> GeoBox3d:
        return GeoBox3d.from_bounds(self.tileset.project_bbox_to_3d(self.tree_node.bbox), crs=self.tileset.crs)

    def get_geometric_error(self) -> float:
        return self.tileset.bbox_to_geometric_error(self.tree_node.bbox)

    def get_refine(self) -> RefinementType:
        return RefinementType.REPLACE        

    def get_content(self) -> Tiles3dContentObject:
        return None
    
    def content_to_bytes(self, content: Tiles3dContentObject) -> bytes:
        return content.to_bytes_tiles3d(transformer=self.tileset.to_epsg4978_transformer)

    def get_content_type(self) -> Tiles3dContentType:
        return self.tileset.content_type

    def get_identifier(self) -> PersistentRTreeTileIdentifier:
        return self.tile_identifier

    def get_cost(self) -> float:
        return 1