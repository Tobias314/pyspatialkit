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

class BBoxStorageTileSet3d(Tileset3d):

    def __init__(self, bboxstorage: BBoxStorage, crs: GeoCrs, geometric_error_multiplier = 8, num_tiles_per_level_edge = 2):
        super().__init__()
        self.bboxstorage = bboxstorage
        self.crs = crs
        self.geometric_error_multiplier = geometric_error_multiplier
        self.num_tiles_per_level_edge = num_tiles_per_level_edge
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

    def get_root(self) -> BBoxStorageTile3d:
        return self.get_tile_by_identifier(BBoxStorageTileIdentifier(self.max_level, (0,0,0)))

    def get_bbox_from_identifier(self,identifier: BBoxStorageTileIdentifier) -> GeoBox3d:
        tile_size = self.get_tile_size_for_level(identifier.level)
        min_pt = self.min + tile_size * np.array(identifier.tile_indices)
        bbox = GeoBox3d(min_pt, min_pt+tile_size, crs=self.crs, to_epsg4978_transformer=self.to_epsg4978_transformer,
                         to_epsg4979_transformer=self.to_epsg4979_transformer)
        return bbox

    def get_tile_by_identifier(self,identifier: BBoxStorageTileIdentifier) -> BBoxStorageTile3d:
        ge = self.geometric_error_multiplier #* identifier.level
        res = BBoxStorageTile3d(self, identifier=identifier, geometric_error=ge)
        dbg('got tile')
        return res
    
    def to_static_directory(self, directory_path: Union[str, Path], max_per_file_depth:Optional[int]=None,
                             max_per_file_cost:Optional[int]=None):
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            directory_path.mkdir(parents=True)
        def uri_generator(tile: BBoxStorageTile3d) -> str:
            level = tile.identifier.level
            indices = tile.identifier.tile_indices
            return str("{}_{}_{}_{}.json".format(level, indices[0], indices[1], indices[2]))
        def content_uri_generator(tile: BBoxStorageTile3d) -> str:
            tile_uri = uri_generator(tile)
            return tile_uri + '_content' + TILES3D_CONTENT_TYPE_TO_FILE_ENDING[tile.content_type.name]
        def content_to_file(tile: BBoxStorageTile3d):
            if tile.content is not None:
                serialized_bytes = tile.content.to_bytes_tiles3d()
                file_path = content_uri_generator(tile)
                with open(directory_path / file_path, "wb") as f:
                    f.write(serialized_bytes)
        backlog = []
        backlog.append(self.root)
        while backlog:
            root_tile = backlog.pop()
            json_dict, new_roots = self.materialize(tile_uri_generator=uri_generator, tile_content_uri_generator=content_uri_generator,
                                                     root_tile=root_tile, max_depth=max_per_file_depth,
                                                     max_cost=max_per_file_cost, callback=content_to_file)
            with open(directory_path / uri_generator(root_tile), 'w') as f:
                json.dump(json_dict, f)
            backlog += new_roots
