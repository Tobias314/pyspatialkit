from typing import Tuple, Union
import threading
from pathlib import Path

import numpy as np
from numpy.core.fromnumeric import shape

from crs.geocrs import GeoCRS
from geopandas import GeoSeries, GeoDataFrame
from rasterio.features import rasterize
from matplotlib import pyplot as plt

from ..spacedescriptors.georect import GeoRect
from ..utils.geopandas import projective_transform
from ..utils.rasterio import save_np_array_as_geotiff


class GeoRaster:

    def __init__(self, georect: GeoRect, data: np.ndarray):
        self.georect = georect
        self.data = data
        self._tranform = georect.transform @ np.array([[1/self.data.shape[0], 0, 0],  [0, 1/self.data.shape[1], 0], [0, 0, 1]])

    @property
    def crs(self) -> GeoCRS:
        return self.georect.crs

    # TODO: might not be pixelaccurate
    def project_into_other(self, other: 'GeoRaster', no_data_value: float = None):
        assert self.data.shape[2] == other.data.shape[2]
        assert self.data.dtype == other.data.dtype
        if self.crs != other.crs:
            raise AttributeError("GeoRasters have different crs.")
        raster_a = self
        raster_b = other
        if self.shape[0] * self.shape[1] > other.shape[0] * other.shape[1]:
            raster_a, raster_b = raster_b, raster_a
        

    def rasterize_shapes(self, shapes: Union[GeoSeries, GeoDataFrame], pixel_value=1, buffer=0):
        if shapes.size == 0:
            return
        if shapes.crs != self.crs.proj_crs:
            shapes = shapes.to_crs(self.crs.proj_crs)
        if isinstance(shapes, GeoDataFrame):
            shapes = shapes['geometry']
        if buffer != 0:
            shapes = shapes.buffer(buffer)
        shapes = projective_transform(shapes, np.linalg.inv(self.transform))
        self.data = rasterize(shapes, self.data)

    def plot_plt(self):
        plt.imshow(self.data)
    
    def plot(self):
        self.plot_plt()

    def to_file(self, file_path: Union[str, Path]):
        save_np_array_as_geotiff(self.data, self.transform, self.crs, file_path)

    @property
    def transform(self):
        return self._transform

    @property
    def shape(self):
        return self.data.shape