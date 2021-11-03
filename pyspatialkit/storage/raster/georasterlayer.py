from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np

from pyspatialkit import DEFAULT_CRS
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCRS, NoneCRS
from ...crs.utils import crs_bounds
from .tiledbbackend import TileDbBackend
from ...spacedescriptors.georect import GeoRect

BACKEND_DIRECTORY_NAME = "backend"

class GeoRasterLayer(GeoLayer):
    
    @abstractmethod
    def initialize(self, num_bands:int, dtype: np.dtype,
                    crs: GeoCRS = DEFAULT_CRS, bounds: Optional[Tuple[float, float, float, float]] = None, fill_value=0,
                    pixel_size: Tuple[float, float] = (1,1,), build_pyramid:bool=True):
        self.num_bands = num_bands
        self.dtype = dtype
        self._crs = crs
        if bounds is None:
            bounds = crs_bounds(crs)
        self.bounds = bounds
        self.pixel_size = pixel_size
        self.fill_value = fill_value
        self.build_pyramid = build_pyramid
        self.backend = TileDbBackend(bounds=self.bounds, num_bands=self.num_bands, dtype=self.dtype,
                                     directory_path=self.directory_path / BACKEND_DIRECTORY_NAME, fill_value=self.fill_value, 
                                     pixel_size=self.pixel_size, build_pyramid=self.build_pyramid)

    @abstractmethod
    def persist_data(self, dir_path: Path):
        config = {}
        config['num_bands'] = self.num_bands
        config['dtype'] = np.dtype(self.dtype).str
        config['crs'] = self.crs.to_dict()
        config['bounds'] = self.bounds
        config['pixel_size'] = self.pixel_size
        config['fill_value'] = self.fill_value
        config['build_pyramid'] = self.build_pyramid
        with open(dir_path / 'config.json', 'w') as json_file:
            json.dump(config, json_file)


    @abstractmethod
    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            self.num_bands = config['num_bands']
            self.dtype = np.dtype(config['dtype'])
            self._crs = GeoCRS.from_dict(config['crs'])
            self.bounds = config['bounds']
            self.pixel_size = config['pixel_size']
            self.fill_value = config['fill_value']
            self.build_pyramid = config['build_pyramid']
            self.backend = TileDbBackend(bounds=self.bounds, num_bands=self.num_bands, dtype=self.dtype,
                                        directory_path=self.directory_path / BACKEND_DIRECTORY_NAME, fill_value=self.fill_value, 
                                        pixel_size=self.pixel_size, build_pyramid=self.build_pyramid)

    @abstractmethod
    def get_raster_for_rect(self, georect: GeoRect, x_resolution: int, y_resolution: int, band=None, no_data_value=0) -> GeoRaster:
        raise NotImplementedError

    @abstractmethod
    def writer_raster_data(self, georaster: GeoRaster):
        raise NotImplementedError

    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return self._crs