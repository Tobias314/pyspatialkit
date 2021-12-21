import unittest
import sys
sys.path.append('../../../../')
from ....utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
import pandas as pd
import logging

from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS
from pyspatialkit.storage.pointcloud.tiledbsparsebackend import TileDbSparseBackend

class TestTileDbSparseBackend(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(100, dtype=np.float64)
        self.y = self.x.copy()
        self.z = self.x.copy()
        self.a = np.arange(100, dtype=np.float64) + 10000
        self.b = np.arange(100, dtype=np.float64) + 20000
        self.c = np.arange(100, dtype=np.float64) + 30000
        self.df = pd.DataFrame({'x':self.x,'y':self.y,'z':self.z,'a':self.a,'b':self.b,'c':self.c})
        self.backend_bounds = [-10,-10,-5,200,150, 101]
        self.data_schema = {'x':np.float64, 'y':np.float64, 'z':np.float64, 'a':np.float64, 'b':np.float64, 'c':np.float64}
        self.dir_path = get_tmp_path() / 'pointcloudbackend'
        self.sparse_backend = TileDbSparseBackend(directory_path=self.dir_path, bounds=self.backend_bounds,
                                                   data_scheme=self.data_schema, space_tile_size=(10,10,10), data_tile_capacity=5,
                                                   build_pyramid=True, base_point_density=1, num_pyramid_layers=3)

    def tearDown(self):
        self.sparse_backend.delete_permanently()

    def test_read_write(self):
        self.sparse_backend.write_data(data=self.df)
        self.sparse_backend.update_pyramid()
        read_df = self.sparse_backend.get_data(bounds=[2,2,2,20,20,20], min_num_points=10)
        print(read_df)
