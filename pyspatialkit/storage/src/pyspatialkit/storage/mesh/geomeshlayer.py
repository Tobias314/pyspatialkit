from pathlib import Path
from typing import Optional, Tuple, Callable, Union, List
import json
from collections.abc import Iterable

import numpy as np

from pyspatialkit.core.dataobjects.geomesh import GeoMesh
from pyspatialkit.core.spacedescriptors.geobox3d import GeoBox3d
from pyspatialkit.core.spacedescriptors.geobox2d import GeoBox2d
from pyspatialkit.core.globals import get_default_crs
from pyspatialkit.core.crs.geocrs import GeoCrs, NoneCRS
from pyspatialkit.core.crs.geocrstransformer import GeoCrsTransformer
from pyspatialkit.core.crs.utils import crs_bounds

from ..geolayer import GeoLayer
from ..backends.persistent_r_tree.persistent_r_tree import PersistentRTree
from ..backends.persistent_r_tree.persistent_r_tree_tileset3d import PersistentRTreeTileset3d


class GeoMeshLayer(GeoLayer):
    
    def initialize(self, crs: Optional[GeoCrs] = None):
        if crs is None:
            crs = get_default_crs()
        self._crs = crs
        self.dimensions = 3
        self.index_dir = self.directory_path / 'index'
        self.index_dir.mkdir(parents=True)
        self.object_dir = self.directory_path / 'objects'
        self.object_dir.mkdir(parents=True)
        self.backend = PersistentRTree(tree_path=self.index_dir, object_type=GeoMesh, dimensions=self.dimensions,
                                       data_path=self.object_dir)
        self._visualizer_tileset = None

    def persist_data(self, dir_path: Path):
        config = {}
        config['crs'] = self.crs.to_dict()
        config['dimensions'] = self.dimensions
        with open(dir_path / 'config.json', 'w') as json_file:
            json.dump(config, json_file)

    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as config:
            config = json.load(config)
            self._crs = GeoCrs.from_dict(config['crs'])
            self.dimensions = config['dimensions']
            self.tile_size = config['tile_size']
            self.index_dir = dir_path / 'index'
            self.object_dir = dir_path / 'objects'
            self.backend = PersistentRTree(tree_path=self.index_dir, object_type=GeoMesh, dimensions=self.dimension,
                                           data_path=self.object_dir)
            self._visualizer_tileset = None

    def get_data(self, geobox: Union[GeoBox3d, GeoBox2d]) -> List[GeoMesh]:
        bbox_min, bbox_max = geobox.min, geobox.max
        if isinstance(geobox, GeoBox2d):
            geobox = GeoBox3d.from_geobox2d(geobox, min_height=self.bounds[2], max_height=self.bounds[5])
        if geobox.crs == self.crs:
            meshes = self.backend.query_bbox(np.concatenate([bbox_min, bbox_max], axis=0))
        else:
            min_coords = geobox.min
            max_coords = geobox.max
            transformer = GeoCrsTransformer(geobox.crs, self.crs)
            min_coords = transformer.transform_tuple(min_coords)
            max_coords = transformer.transform_tuple(max_coords)
            meshes = self.backend.query_bbox(np.concatenate([bbox_min, bbox_max], axis=0))
            meshes = [mesh.to_crs(geobox.crs, inplace=True) for mesh in meshes]
        meshes = [mesh for mesh in meshes if mesh.is_intersecting_bbox(bbox_min, bbox_max)]
        return meshes

    def write_data(self, geomesh: GeoMesh):
        if geomesh.crs != self.crs:
            geomesh = geomesh.to_crs(self.crs, inplace=False)
        self.backend.insert([geomesh])

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
        self.backend.insert(geomeshes)

    def _delete_permanently(self):
        self.backend.delete_permanently()

    def invalidate_cache(self):
        self.backend.invalidate_cache()

    @property
    def visualizer_tileset(self):
        if self._visualizer_tileset is None:
            self._visualizer_tileset = PersistentRTreeTileset3d(tree=self.backend, crs=self.crs)
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