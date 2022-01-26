from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List, Union
import json
from collections.abc import Iterable
import shutil

import time
import numpy as np
from pyproj.enums import TransformDirection
from skimage.transform import resize
from matplotlib import pyplot as plt


from ...globals import DEFAULT_CRS
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.geocrstransformer import TransformDirection
from ...crs.geocrstransformer import GeoCrsTransformer
from ...crs.utils import crs_bounds
from .tiledbsparsebackend import TileDbSparseBackend
from ...spacedescriptors.georect import GeoRect
from ...dataobjects.georaster import GeoRaster
from ...utils.datascheme import datascheme_to_str_dict, datascheme_from_str_dict
from ...spacedescriptors.geobox3d import GeoBox3d
from ...spacedescriptors.geobox2d import GeoBox2d
from ...dataobjects.geopointcloud import GeoPointCloud, GeoPointCloudReadable, GeoPointCloudWritable
from ...tiling.geoboxtiler3d import GeoBoxTiler3d
from ...utils.logging import dbg


BACKEND_DIRECTORY_NAME = "backend"

DEFAULT_MIN_ELEVATION = -20000
DEFAULT_MAX_ELEVATION = 100000
DEFAULT_POINT_PER_METER_1D = 300


class GeoPointCloudLayer(GeoLayer, GeoPointCloudReadable, GeoPointCloudWritable):

    def initialize(self, data_scheme: Dict[str, np.dtype], crs: GeoCrs = DEFAULT_CRS, bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                   point_density=0.01, build_pyramid: bool = True, rgb_max:float = 1):
        self._data_scheme = data_scheme
        self._crs = crs
        if bounds is None:
            bounds = crs_bounds(crs)
        if len(bounds) == 4:
            bounds = (bounds[0], bounds[1], DEFAULT_MIN_ELEVATION,
                      bounds[2], bounds[3], DEFAULT_MAX_ELEVATION)
        self._bounds = bounds
        self.build_pyramid = build_pyramid
        self._eager_pyramid_update = True
        self.point_density = point_density
        tmp = DEFAULT_POINT_PER_METER_1D / self.point_density
        self.backend_space_tile_size = (tmp, tmp, tmp)
        self.rgb_max = rgb_max
        self.backend = TileDbSparseBackend(bounds=self._bounds, directory_path=self.directory_path / BACKEND_DIRECTORY_NAME,
                                           data_scheme=self._data_scheme,
                                           space_tile_size=self.backend_space_tile_size,
                                           build_pyramid=self.build_pyramid)

    def persist_data(self, dir_path: Path):
        config = {}
        config['crs'] = self.crs.to_dict()
        config['bounds'] = list(self._bounds)
        config['build_pyramid'] = self.build_pyramid
        config['data_schema'] = datascheme_to_str_dict(self._data_scheme)
        config['point_density'] = self.point_density
        config['backend_space_tile_size'] = self.backend_space_tile_size
        config['rgb_max'] = self.rgb_max
        with open(dir_path / 'config.json', 'w') as json_file:
            json.dump(config, json_file)

    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            self._crs = GeoCrs.from_dict(config['crs'])
            self._bounds = np.array(config['bounds'])
            self.build_pyramid = config['build_pyramid']
            self._data_scheme = datascheme_from_str_dict(config['data_schema'])
            self.point_density = config['point_density']
            self.backend_space_tile_size = config['backend_space_tile_size']
            self.rgb_max = config['rgb_max']
            self.backend = TileDbSparseBackend(bounds=self._bounds, directory_path=self.directory_path / BACKEND_DIRECTORY_NAME,
                                                data_scheme=self._data_scheme,
                                                space_tile_size=self.backend_space_tile_size,
                                                build_pyramid=self.build_pyramid)

    def get_data(self, geobox: Union[GeoBox3d, GeoBox2d], attributes: Optional[List[str]] = None) -> GeoPointCloud:
        if isinstance(geobox, GeoBox2d):
            geobox = GeoBox3d.from_geobox2d(geobox, min_height=self.bounds[2], max_height=self.bounds[5])
        if geobox.crs == self.crs:
                data = self.backend.get_data([*geobox.min, *geobox.max], attributes=attributes)
        else:
            min_coords = geobox.min
            max_coords = geobox.max
            transformer = GeoCrsTransformer(geobox.crs, self.crs)
            min_coords = transformer.transform_tuple(min_coords)
            max_coords = transformer.transform_tuple(max_coords)
            data = self.backend.get_data([*geobox.min, *geobox.max], attributes=attributes)
            data['x'], data['y'], data['z'] = transformer.transform(data['x'], data['y'], data['z'], direction=TransformDirection.INVERSE)
        return GeoPointCloud.from_pandas(data, crs=geobox.crs, rgb_max=self.rgb_max)

    def write_data(self, geopointcloud: GeoPointCloud):
        if geopointcloud._crs != self.crs:
            geopointcloud.to_crs(self.crs)
        self.backend.write_data(geopointcloud.data)

    def begin_pyramid_update_transaction(self):
        self._eager_pyramid_update = False

    def commit_pyramid_update_transaction(self):
        if not self._eager_pyramid_update:
            if self.build_pyramid:
                self.backend.update_pyramid()
            self._eager_pyramid_update = True

    def apply(self, tiler: GeoBoxTiler3d,
              transformer: Callable[[GeoPointCloud], Optional[GeoPointCloud]],
              attributes: Optional[List[str]] = None, output_layer: Optional['GeoPointCloudLayer'] = None):
        if output_layer is None:
            output_layer = self
        output_layer.begin_pyramid_update_transaction()
        for box3d, box_without_border in tiler:
            pc = self.get_data(box3d,  attributes=attributes)
            pc = transformer(pc, box_without_border)
            if pc is not None:
                output_layer.write_data(pc)
        output_layer.commit_pyramid_update_transaction()

    def _delete_permanently(self):
        self.backend.delete_permanently()

    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return self._crs

    @property
    def bounds(self):
        return self._bounds

    @property
    def data_scheme(self):
        return self._data_scheme
