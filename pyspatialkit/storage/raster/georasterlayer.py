from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import json
from collections.abc import Iterable

import numpy as np

from pyspatialkit import DEFAULT_CRS
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.utils import crs_bounds
from .tiledbbackend import TileDbBackend
from ...spacedescriptors.georect import GeoRect
from ...datastructures.georaster import GeoRaster

BACKEND_DIRECTORY_NAME = "backend"

class GeoRasterLayer(GeoLayer):
    
    def initialize(self, num_bands:int, dtype: np.dtype,
                    crs: GeoCrs = DEFAULT_CRS, bounds: Optional[Tuple[float, float, float, float]] = None, fill_value=0,
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

    def load_data(self, dir_path: Path):
        with open(dir_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            self.num_bands = config['num_bands']
            self.dtype = np.dtype(config['dtype'])
            self._crs = GeoCrs.from_dict(config['crs'])
            self.bounds = config['bounds']
            self.pixel_size = config['pixel_size']
            self.fill_value = config['fill_value']
            self.build_pyramid = config['build_pyramid']
            self.backend = TileDbBackend(bounds=self.bounds, num_bands=self.num_bands, dtype=self.dtype,
                                        directory_path=self.directory_path / BACKEND_DIRECTORY_NAME, fill_value=self.fill_value, 
                                        pixel_size=self.pixel_size, build_pyramid=self.build_pyramid)

    def get_raster_for_rect(self, georect: GeoRect, resolution: Optional[Tuple[int,int]]=None, band=None, no_data_value=0) -> GeoRaster:
        if georect.crs != self.crs:
            georect.to_crs(self.crs)
        bounds = np.array(georect.get_bounds()).astype(int)
        if resolution is None:
            resolution = (bounds[2:] - bounds[:2]) / self.backend.pixel_size
        raster_data = self.backend.get_data(bounds, resolution)
        if len(raster_data.shape)==2:
            raster_data = raster_data[:,:,np.newaxis]
        if band is not None:
            raster_data = raster_data[:,:,band]
        if georect.is_axis_aligned:
            return GeoRaster(georect, raster_data)
        else:
            bounds_rect = GeoRect(bottom_left=bounds[:2], top_right=bounds[2:], crs=self.crs)
            bounds_raster = GeoRaster(bounds_rect, raster_data)
            res_shape = [resolution[0], resolution[1], 1]
            if isinstance(band, Iterable):
                res_shape[2] = len(band)
            elif band is None:
                res_shape[2] = raster_data.shape[2]
            result_raster = GeoRaster(georect=georect, data=np.full(res_shape, no_data_value, dtype=raster_data.dtype))
            result_raster.merge_projected_other(bounds_raster)
            return result_raster

    def writer_raster_data(self, georaster: GeoRaster):
        if georaster.crs != self.crs:
            georaster.to_crs(self.crs)
        bounds = np.array(georaster.georect.get_bounds()).astype(int)
        self.backend.write_data(bounds, georaster.data)

    @property
    def name(self):
        return self.folder_path.name

    @property
    def crs(self):
        return self._crs