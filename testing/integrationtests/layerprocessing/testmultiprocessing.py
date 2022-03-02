import multiprocessing
import sys
import unittest

sys.path.append('../../../')
from ...utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
import logging

from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS
from pyspatialkit.spacedescriptors.geobox2d import GeoBox2d
from pyspatialkit.tiling.geoboxtiler2d import GeoBoxTiler2d
from pyspatialkit.layerprocessing.decorators import layerprocessor

@layerprocessor
def identity(tile: GeoBox2d, height: GeoRasterLayer):
    img=height.get_data(tile.to_georect())
    img.data += 10
    height.write_data(img)

class TestDecorators(unittest.TestCase):

    def setUp(self):
        self.dir_path = get_tmp_path() / 'rasterlayer'
        self.crs = NoneCRS()
        self.raster_layer = GeoRasterLayer(directory_path=self.dir_path, num_bands=1, dtype=float, crs=self.crs, bounds=[0,0,100,100], build_pyramid=False)

    def tearDown(self):
        close_all_files_delete_dir(self.dir_path)

    def test_layerprocessing(self):
        self.raster_layer = GeoRasterLayer(directory_path=self.dir_path, num_bands=1, dtype=float, crs=self.crs, bounds=[0,0,100,100], build_pyramid=False)
        aoi = GeoBox2d.from_bounds([0,0,100, 100], crs=self.crs)
        self.raster_layer.write_data(GeoRaster(GeoRect.from_bounds([0,0,100,100], crs=self.crs), np.ones((100,100,1))))
        tiler = GeoBoxTiler2d(aoi=aoi, raster_size=(50,50), reference_crs=self.crs)
        identity(tiler=tiler, num_workers=2)(self.raster_layer)
        rect = GeoRect((0,0), (100,100), crs=self.crs)
        self.assertEqual(self.raster_layer.get_data(rect).data.mean(), 11.0)