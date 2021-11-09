import unittest
import sys
sys.path.append('../../../../')
from ....utils import get_testdata_path, close_all_files_delete_dir

import numpy as np
import logging

from pyspatialkit.datastructures.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS

class TestGeoRasterLayer(unittest.TestCase):

    def setUp(self):
        self.dir_path = get_testdata_path() / 'rasterlayer'
        self.raster_layer = GeoRasterLayer(directory_path=self.dir_path, num_bands=1, dtype=float, crs=NoneCRS(), bounds=[0,0,10000,10000])

    def tearDown(self):
        close_all_files_delete_dir(self.dir_path)

    def test_read_write(self):
        raster = GeoRaster(GeoRect((750,100), (1750,1100), crs=NoneCRS()), np.ones((1000,1000)))
        self.raster_layer.writer_raster_data(raster)
        res = self.raster_layer.get_raster_for_rect(GeoRect((0,0), (1000,1000), crs=NoneCRS()))
        self.assertEqual(res.data.sum(), 900*250)