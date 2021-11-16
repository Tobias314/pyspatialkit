import sys
sys.path.append('../../')
sys.path.append('../')
from utils import get_testdata_path, get_tmp_path

import time
import numpy as np
from skimage.transform import warp
from matplotlib import pyplot as plt
from pyproj import CRS
import rasterio as rio
from pyspatialkit.storage.geostorage import GeoStorage
from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS, GeoCrs

storage = GeoStorage(directory_path=get_tmp_path() / 'geostorage')

raster1 = GeoRaster.from_file(get_testdata_path() / "dop100rgbi_32_734_5748_2_st_2020.tif", band=[1,2,3])
raster2 = GeoRaster.from_file(get_testdata_path() / "dop100rgbi_32_736_5748_2_st_2020.tif", band=[1,2,3])

web_crs = GeoCrs(CRS.from_epsg(3857))
raster1.to_crs(web_crs)
raster2.to_crs(web_crs)

raster_layer = storage.add_raster_layer('raster_layer', 3, raster1.dtype, crs=raster1.crs, bounds=[-20026377, -20026377, 20048967, 20048967])

raster_layer.writer_raster_data(raster1)
raster_layer.writer_raster_data(raster2)
