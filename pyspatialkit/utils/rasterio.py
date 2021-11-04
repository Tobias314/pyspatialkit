from typing import Union
from pathlib import Path

import numpy as np
import rasterio as rio
from affine import Affine

from crs.geocrs import GeoCRS

from ..spacedescriptors.georect import GeoRect

def rasterio_dataset_writer_from_rasterio_dataset(rasterio_dataset: rio.io.DatasetReader, path: Union[str, Path],
                                                  driver: str = 'GTiff'):
    return rio.open(path,
                       'w',
                       driver=driver,
                       height=rasterio_dataset.height,
                       width=rasterio_dataset.width,
                       count=rasterio_dataset.count,
                       dtype=rasterio_dataset.dtypes[0],
                       crs=rasterio_dataset.crs,
                       transform=rasterio_dataset.transform)

def save_raster(raster: rio.io.DatasetReader, path: Union[str, Path], driver: str = 'GTiff'):
    #only supports one dtype per raster for now
    assert has_only_one_dtype(raster)
    new_dataset = rasterio_dataset_writer_from_rasterio_dataset(raster, path, driver)
    bands = list(range(1, raster.count + 1))
    for band in bands:
        new_dataset.write(raster.read(band), band)

def save_np_array_as_geotiff(img: np.ndarray, transform: np.ndarray, crs: GeoCRS, path: Union[str, Path]):
    bands = img.shape[2] if len(img.shape)>2 else 1
    img_dim = [img.shape[0],img.shape[1]]
    if not (transform[2,:2] == np.array([0,0])).all():
        raise AttributeError("The image needs to have an affine transformation for saving")
    transform = transform / transform[2,2]
    transform = Affine(*transform[0], *transform[1])
    new_dataset = rio.open(path,
                           'w',
                           driver = 'GTiff',
                           height = img_dim[0],
                           width = img_dim[1],
                           count = bands,
                           dtype = img.dtype,
                           crs = rio.crs.CRS.from_string(crs.proj_crs.to_string()),
                           transform = transform
                           )
    if bands == 1:
        new_dataset.write(img[:,:,0], 1)
    else:
        for band in range(1, bands+1):
            new_dataset.write(img[:,:,band-1], band)

def has_only_one_dtype(raster: rio.io.DatasetReader):
    result = True
    for dtype in raster.dtypes:
        result = result and dtype == raster.dtypes[0]
    return result
