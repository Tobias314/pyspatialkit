from typing import Tuple, Union, Optional, List
import threading
from pathlib import Path

import numpy as np
from geopandas import GeoSeries, GeoDataFrame
from pyproj import crs
from rasterio.features import rasterize
from matplotlib import pyplot as plt
import rasterio as rio


from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer
from ..spacedescriptors.georect import GeoRect
from ..utils.geopandas import projective_transform
from ..utils.rasterio import save_np_array_as_geotiff
from ..utils.projection import project_skimage
from .geoshape import GeoShape
from ..utils.logging import raise_warning

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
        #images indices are [row,col] while coordinate transforms work with [x,y], so we need to swap x,y and flip y
        self._row_col_to_x_y = np.array([[0, 1/self.data.shape[1], 0],  [-1/self.data.shape[0], 0, 1], [0, 0, 1]])
        self._swap_x_y = np.array([[0,1,0],[1,0,0],[0,0,1]])
        self._transform = self.georect.transform @ self._row_col_to_x_y

    @classmethod
    def from_file(cls, file_path: Union[str, Path], band: Optional[Union[int, List[int]]]=None) -> 'GeoRaster':
        raster_dataset = rio.open(str(file_path))
        raster_data = raster_dataset.read(band)
        raster_data = np.moveaxis(raster_data, 0, -1)
        #raster_data = np.moveaxis(raster_data, 1, 0)
        #raster_data = np.flip(raster_data, 1)
        bl = raster_dataset.xy(raster_data.shape[1], 0)
        br = raster_dataset.xy(raster_data.shape[1], raster_data.shape[0])
        tr = raster_dataset.xy(0, raster_data.shape[0])
        tl = raster_dataset.xy(0, 0)
        crs = GeoCrs(raster_dataset.crs)
        georect = GeoRect(bl, tr, br, tl, crs=crs)
        return GeoRaster(georect=georect, data=raster_data)

    def copy(self):
        return GeoRaster(self.georect.copy(), self.data.copy())

    @property
    def crs(self) -> GeoCrs:
        return self.georect.crs

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace=True) -> 'GeoRaster':
        if inplace:
            self.georect.to_crs(new_crs=new_crs, crs_transformer=crs_transformer, inplace=True)
            #images indices are [row,col] while coordinate transforms work with [x,y], so wie need to swap x,y
            self._transform = self.georect.transform @ self._row_col_to_x_y

            return self
        else:
            new_georect = self.georect.to_crs(new_crs=new_crs, crs_transformer=crs_transformer, inplace=False)
            return GeoRaster(new_georect, self.data.copy())
        

    # TODO: might not be pixelaccurate
    def merge_projected_other(self, other: 'GeoRaster', self_no_data_value: Optional[float] = None, self_mask: Optional[np.ndarray] = None,
                               other_no_data_value: Optional[float] = None, other_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.data.shape[2] == other.data.shape[2]
        assert self.data.dtype == other.data.dtype
        if self.crs != other.crs:
            raise AttributeError("GeoRasters have different crs.")
        inv_mat = (self._swap_x_y @ np.linalg.inv(other.transform) @ self.transform @ self._swap_x_y).round(9) #TODO: We round to prevent problems from float inaccuracies. Think about a better method for this!!!
        project_skimage(other.data, self.data, inv_mat, other_no_data_value, other_mask,
                        self_no_data_value, self_mask)

    def rasterize_shapes(self, shapes: Union[GeoShape, GeoSeries, GeoDataFrame], pixel_value=1, buffer=0):
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

    def rasterize_shape(self, shape: GeoShape, pixel_value=1, buffer=0):
        shapes = pd.GeodataFrame(geometry=shape.shape, crs=shape.crs.proj_crs)
        self.rasterize_shapes(shapes, pixel_value, buffer)

    def plot_plt(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots()
        data = self.data
        if data.shape[2] == 4:
            data = self.data[:,:,:3]
        if data.shape[2]==3 and data.dtype != np.uint8:
            data = data.astype(np.uint8)
            raise_warning("3 channel image with dtype not np.uint8. Transformed dtype to np.uint8!!!", UserWarning)
        ax.imshow(data)
    
    def plot(self):
        self.plot_plt()

    def to_file(self, file_path: Union[str, Path], ignore_not_affine: bool = False):
        save_np_array_as_geotiff(self.data, self.transform @ self._swap_x_y, self.crs, file_path, ignore_not_affine)

    @property
    def transform(self):
        return self._transform

    @property
    def inv_transform(self):
        return np.linalg.inv(self._transform)

    @property
    def shape(self)-> Union[Tuple[int, int], Tuple[int, int, int]]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype