from typing import Tuple, Union, Optional
import threading
from pathlib import Path

import numpy as np
from geopandas import GeoSeries, GeoDataFrame
from pyproj import crs
from rasterio.features import rasterize
from matplotlib import pyplot as plt


from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer
from ..spacedescriptors.georect import GeoRect
from ..utils.geopandas import projective_transform
from ..utils.rasterio import save_np_array_as_geotiff


class GeoRaster:

    def __init__(self, georect: GeoRect, data: np.ndarray):
        self.georect = georect
        if len(data.shape) == 3:
            self.data = data
        elif len(data.shape) == 2:
            self.data = data[:,:, np.newaxis]
        else:
            raise AttributeError('Raster data must be 2D! Data attribute has shape {}'.format(data.shape) +
                                  ' but needs to have either shape (width, height) or (width, heith,num_channels))!')
        self._transform = georect.transform @ np.array([[1/self.data.shape[0], 0, 0],  [0, 1/self.data.shape[1], 0], [0, 0, 1]])

    @property
    def crs(self) -> GeoCrs:
        return self.georect.crs

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace=True) -> 'GeoRaster':
        if inplace:
            self.georect.to_crs(new_crs=new_crs, crs_transformer=crs_transformer, inplace=True)
            self._transform = self.georect.transform @ np.array([[1/self.data.shape[0], 0, 0],  [0, 1/self.data.shape[1], 0], [0, 0, 1]])
            return self
        else:
            new_georect = self.georect.to_crs(new_crs=new_crs, crs_transformer=crs_transformer, inplace=False)
            return GeoRaster(new_georect, self.data.copy())
        

    # TODO: might not be pixelaccurate
    # TODO: do profiling to optimize performence because this function might be critical for the overall performance
    def merge_projected_other(self, other: 'GeoRaster', self_no_data_value: Optional[float] = None, self_mask: Optional[np.ndarray] = None,
                               other_no_data_value: Optional[float] = None, other_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.data.shape[2] == other.data.shape[2]
        assert self.data.dtype == other.data.dtype
        if self.crs != other.crs:
            raise AttributeError("GeoRasters have different crs.")
        a_raster = other
        b_raster = self
        a_mask, a_no_data_value = other_mask, other_no_data_value
        b_mask, b_no_data_value = self_mask, self_no_data_value
        flipped = False
        if self.shape[0] * self.shape[1] < other.shape[0] * other.shape[1]:
            a_raster, b_raster = b_raster, a_raster
            a_no_data_value, b_no_data_value = b_no_data_value, a_no_data_value
            b_mask, a_mask = a_mask, b_mask
            flipped = True
        coords = np.stack(np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing='ij'))
        if a_mask is not None and b_no_data_value is not None:
            coords = coords[:, a_mask & a_raster.data == b_no_data_value]
        elif a_mask is not None:
            coords = coords[:, a_mask]
        elif b_no_data_value is not None:
            coords = coords[:, a_raster.data == b_no_data_value]
        else:
            coords = coords.reshape((2, -1))
        eps = np.finfo(np.float32).eps # we will add a small epsilon to prevent off by one errors caused by rounding
        projected_coords = np.linalg.inv(b_raster.transform) @ a_raster.transform @ np.concatenate([coords.astype(float) + eps, np.ones([1, coords.shape[1]])])
        projected_coords = (projected_coords[:2] / projected_coords[2]).astype(int)
        projected_coords_mask = (projected_coords[0] >= 0) & (projected_coords[0] < b_raster.shape[0]) & \
                                 (projected_coords[1] >= 0) & (projected_coords[1] < b_raster.shape[0])
        if b_mask is not None and b_no_data_value is not None:
            projected_coords_mask = b_mask[projected_coords[0], projected_coords[1]] \
                & (b_raster.data == b_no_data_value)[projected_coords[0], projected_coords[1]]
        elif b_mask is not None:
            projected_coords_mask = b_mask[projected_coords[0], projected_coords[1]]
        elif b_no_data_value is not None:
            projected_coords_mask = (b_raster.data == b_no_data_value)[projected_coords[0], projected_coords[1]]
        coords = coords[:, projected_coords_mask]
        projected_coords = projected_coords[:, projected_coords_mask]
        if flipped:
            projected_coords, coords = coords, projected_coords
        else:
            self.data[projected_coords[0], projected_coords[1]] = other.data[coords[0], coords[1]]
        return projected_coords, coords    

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
        plt.imshow(np.swapaxes(self.data, 0, 1))
    
    def plot(self):
        self.plot_plt()

    def to_file(self, file_path: Union[str, Path]):
        save_np_array_as_geotiff(self.data, self.transform, self.crs, file_path)

    @property
    def transform(self):
        return self._transform

    @property
    def shape(self)-> Union[Tuple[int, int], Tuple[int, int, int]]:
        return self.data.shape