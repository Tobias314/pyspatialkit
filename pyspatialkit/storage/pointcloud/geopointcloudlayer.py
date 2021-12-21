from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Callable
import json
from collections.abc import Iterable

import time
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
import cv2 as cv

from ...globals import DEFAULT_CRS
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.utils import crs_bounds
from .tiledbsparsebackend import TileDbSparseBackend
from ...spacedescriptors.georect import GeoRect
from ...dataobjects.georaster import GeoRaster

BACKEND_DIRECTORY_NAME = "backend"

class GeoPointCloudLayer(GeoLayer):
    
    def initialize(self, dtype: np.dtype, crs: GeoCrs = DEFAULT_CRS, bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                    point_density=0.01, build_pyramid:bool=True):
        self.dtype = dtype
        self._crs = crs
        if bounds is None:
            bounds = crs_bounds(crs)
            bounds = (bounds[0], bounds[1], -20000, bounds[2], bounds[3], 100000) #TODO: think about constants, maybe pull them out
        self.bounds = bounds
        self.build_pyramid = build_pyramid
        self._eager_pyramid_update = True
        self.backend = TileDbSparseBackend(bounds=self.bounds, num_bands=self.num_bands, dtype=self.dtype,
                                     directory_path=self.directory_path / BACKEND_DIRECTORY_NAME, fill_value=self.fill_value, 
                                     pixel_size_xy=self.pixel_size_xy, build_pyramid=self.build_pyramid)

    def persist_data(self, dir_path: Path):
        config = {}
        config['dtype'] = np.dtype(self.dtype).str
        config['crs'] = self.crs.to_dict()
        config['bounds'] = self.bounds
        config['build_pyramid'] = self.build_pyramid
        with open(dir_path / 'config.json', 'w') as json_file:
            json.dump(config, json_file)

    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            self.dtype = np.dtype(config['dtype'])
            self._crs = GeoCrs.from_dict(config['crs'])
            self.bounds = config['bounds']
            self.build_pyramid = config['build_pyramid']
            self.backend = TileDbSparseBackend(bounds=self.bounds, num_bands=self.num_bands, dtype=self.dtype,
                                        directory_path=self.directory_path / BACKEND_DIRECTORY_NAME, fill_value=self.fill_value, 
                                        pixel_size_xy=self.pixel_size_xy, build_pyramid=self.build_pyramid)


    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return self._crs