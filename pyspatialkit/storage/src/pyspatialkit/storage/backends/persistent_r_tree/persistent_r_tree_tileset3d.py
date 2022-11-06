from typing import NamedTuple, Union, Dict, Optional, TYPE_CHECKING, Tuple, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import json
import os

import numpy as np

from pyspatialkit.core import GeoCrs
from pyspatialkit.core import GeoBox3d
from pyspatialkit.core.crs.geocrstransformer import GeoCrsTransformer
from pyspatialkit.core.dataobjects.tiles3d.tileset3d import Tileset3d
from pyspatialkit.core.dataobjects.tiles3d.tiles3dcontentobject import Tiles3dContentType, TILES3D_CONTENT_TYPE_TO_FILE_ENDING

from .persistent_r_tree import PersistentRTree
from .persistent_r_tree_tile3d import PersistentRTreeTile3d, PersistentRTreeTileIdentifier, PersistentRTreeContentTile3d


def default_project_bbox_to_3d(bbox: np.ndarray)->np.ndarray:
    dims = len(bbox) // 2
    if dims > 3:
        bbox = bbox[[0,1,2,dims,dims+1,dims+2]]
    if dims < 3:
        min_pt = np.concatenate([bbox[:dims], np.zeros(3-dims)], axis=0)
        max_pt = np.concatenate([bbox[dims:], np.full(3-dims, np.linalg.norm(bbox[dims:]-bbox[:dims]))], axis=0)
        bbox = np.concatenate([min_pt,max_pt])
    return bbox

def default_bbox_to_geometric_error(bbox: np.ndarray)->float:
    return np.linalg.norm(bbox).item()


class PersistentRTreeTileset3d(Tileset3d):

    def __init__(self, tree: PersistentRTree, crs: GeoCrs, content_type:Tiles3dContentType = Tiles3dContentType.MESH,
                 root_geometric_error: float = 100, project_bbox_to_3d: Callable[[np.ndarray,GeoCrs], GeoBox3d] = default_project_bbox_to_3d,
                 bbox_to_geometric_error: Callable[[np.ndarray], float] = default_bbox_to_geometric_error):
        super().__init__()
        self.tree = tree
        self.crs = crs
        self.root_geometric_error = root_geometric_error
        self.project_bbox_to_3d = project_bbox_to_3d
        self.bbox_to_geometric_error = bbox_to_geometric_error
        self.content_type = content_type

    @property
    def tileset_version(self) -> Union[float, str]:
        return 1.0

    @property
    def properties(self) -> Dict[str, Dict[str, float]]:
        return {}

    @property #TODO
    def geometric_error(self) -> float:
        return self.root_geometric_error

    def get_root(self) -> PersistentRTreeTile3d:
        return PersistentRTreeTile3d(tileset=self, tree_node=self.tree.get_root_node())

    def get_bbox_from_identifier(self, identifier: PersistentRTreeTileIdentifier)->GeoBox3d: #TODO: think about only loding bbox
        return self.get_tile_by_identifier(identifier).bounding_volume

    def get_tile_by_identifier(self, identifier: PersistentRTreeTileIdentifier)-> PersistentRTreeTile3d: #TODO improve caching
        tree_node = self.tree.get_node(node_id=identifier.node_id)
        if identifier.object_index is None:
            return PersistentRTreeTile3d(tileset=self, tree_node=tree_node)
        else:
            return PersistentRTreeContentTile3d(tilest=self, tree_node=tree_node, object_index=identifier.object_index)