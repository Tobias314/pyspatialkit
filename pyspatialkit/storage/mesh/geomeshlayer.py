from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Callable, Union, List
import json
from collections.abc import Iterable

import time
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
import cv2 as cv

from ...globals import get_default_crs
from ...tiling.geotiler2d import GeoTiler2d
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.geocrstransformer import GeoCrsTransformer
from ...crs.utils import crs_bounds
from ..bboxstorage.bboxstorage import BBoxStorage
from ...dataobjects.geomesh import GeoMesh
from ...spacedescriptors.geobox3d import GeoBox3d
from ...spacedescriptors.geobox2d import GeoBox2d
from ...dataobjects.tiles3d.bboxstorage.bboxstoragetileset3d import BBoxStorageTileSet3d

BACKEND_DIRECTORY_NAME = "backend"
DEFAULT_MIN_ELEVATION = -20000
DEFAULT_MAX_ELEVATION = 100000

class GeoMeshLayer(GeoLayer):
    
    def initialize(self, crs: Optional[GeoCrs] = None,  bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                   tile_size: Union[int, Tuple[float, float, float]] = 1000, 
                   tile_cache_size: int = 100, object_cache_size: int = 1000):
        if crs is None:
            crs = get_default_crs()
        self._crs = crs
        if bounds is None:
            bounds = crs_bounds(crs)
        if len(bounds) == 4:
            bounds = (bounds[0], bounds[1], DEFAULT_MIN_ELEVATION,
                      bounds[2], bounds[3], DEFAULT_MAX_ELEVATION)
        self._bounds = bounds
        if isinstance(tile_size, int):
            self.tile_size = tuple([tile_size for i in range(len(self.bounds) // 2)])
        else:
            self.tile_size = tile_size
        self.tile_cache_size = tile_cache_size
        self.object_cache_size = object_cache_size
        self.backend = BBoxStorage(self.directory_path / BACKEND_DIRECTORY_NAME, bounds=self.bounds, tile_size=self.tile_size,
                                   object_type=GeoMesh, tile_cache_size=self.tile_cache_size, object_cache_size=self.object_cache_size)
        self._visualizer_tileset = None

    def persist_data(self, dir_path: Path):
        config = {}
        config['crs'] = self.crs.to_dict()
        config['bounds'] = tuple(self.bounds)
        config['tile_size'] = tuple(self.tile_size)
        config['tile_cache_size'] = self.tile_cache_size
        config['ojbect_cache_size'] = self.object_cache_size
        with open(dir_path / 'config.json', 'w') as json_file:
            json.dump(config, json_file)
        self.backend.persist_to_file()

    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as config:
            config = json.load(config)
            self._crs = GeoCrs.from_dict(config['crs'])
            self._bounds = config['bounds']
            self.tile_size = config['tile_size']
            self.tile_cache_size = config['tile_cache_size']
            self.object_cache_size = config['ojbect_cache_size']
            self.backend = BBoxStorage(dir_path / BACKEND_DIRECTORY_NAME, bounds=self.bounds, tile_size=self.tile_size,
                                       object_type=GeoMesh, tile_cache_size=self.tile_cache_size, object_cache_size=self.object_cache_size)
            self._visualizer_tileset = None

    def get_data(self, geobox: Union[GeoBox3d, GeoBox2d]) -> List[GeoMesh]:
        bbox_min, bbox_max = geobox.min, geobox.max
        if isinstance(geobox, GeoBox2d):
            geobox = GeoBox3d.from_geobox2d(geobox, min_height=self.bounds[2], max_height=self.bounds[5])
        if geobox.crs == self.crs:
            meshes = self.backend.get_objects_for_bbox(bbox_min, bbox_max)
        else:
            min_coords = geobox.min
            max_coords = geobox.max
            transformer = GeoCrsTransformer(geobox.crs, self.crs)
            min_coords = transformer.transform_tuple(min_coords)
            max_coords = transformer.transform_tuple(max_coords)
            meshes = self.backend.get_objects_for_bbox(bbox_min, bbox_max)
            meshes = [mesh.to_crs(geobox.crs, inplace=True) for mesh in meshes]
        meshes = [mesh for mesh in meshes if mesh.is_intersecting_bbox(bbox_min, bbox_max)]
        return meshes

    def write_data(self, geomesh: GeoMesh):
        if geomesh.crs != self.crs:
            geomesh = geomesh.to_crs(self.crs, inplace=False)
        self.backend.write_objecs([geomesh])

    def write_data_batch(self, geomeshes: List[GeoMesh]):
        for mesh in geomeshes:
            if mesh.crs != geomeshes[0].crs:
                raise AttributeError("All meshes in write batch need to have the same crs!")
        if geomeshes[0].crs != self.crs:
            transformer = GeoCrsTransformer(geomeshes[0].crs, self.crs)
            write_meshes = []
            for mesh in geomeshes:
                write_meshes.append(mesh.to_crs(transformer=transformer, inplace=False))
            geomeshes = write_meshes
        self.backend.write_objecs(geomeshes)

    def _delete_permanently(self):
        self.backend.delete_permanently()

    def invalidate_cache(self):
        self.backend.invalidate_cache()

    @property
    def visualizer_tileset(self) -> BBoxStorageTileSet3d:
        if self._visualizer_tileset is None:
            self._visualizer_tileset = BBoxStorageTileSet3d(self.backend, crs=self.crs)
        return self._visualizer_tileset

    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return self._crs

    @property
    def bounds(self):
        return self._bounds