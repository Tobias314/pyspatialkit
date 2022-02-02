import unittest
import sys
sys.path.append('../../../../')
from ....utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
import pandas as pd
import logging
from shapely.geometry import Polygon

from pyspatialkit.spacedescriptors.geobox3d import GeoBox3d
from pyspatialkit.dataobjects.geoshape import GeoShape
from pyspatialkit.crs.geocrs import NoneCRS
from pyspatialkit.storage.pointcloud.tiledbsparsebackend import TileDbSparseBackend
from pyspatialkit.storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer
from pyspatialkit.dataobjects.geopointcloud import GeoPointCloud
from pyspatialkit.tiling.geoboxtiler3d import GeoBoxTiler3d

class TestGeoPointCloudLayer(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(100, dtype=np.float64)
        self.y = self.x.copy()
        self.z = self.x.copy()
        self.a = np.arange(100, dtype=np.float64) + 10000
        self.b = np.arange(100, dtype=np.float64) + 20000
        self.c = np.arange(100, dtype=np.float64) + 30000
        self.df = pd.DataFrame({'x':self.x,'y':self.y,'z':self.z,'a':self.a,'b':self.b,'c':self.c})
        self.pc = GeoPointCloud.from_pandas(self.df, crs=NoneCRS())
        self.backend_bounds = [-10,-10,-5,200,150, 101]
        self.data_schema = {'x':np.float64, 'y':np.float64, 'z':np.float64, 'a':np.float64, 'b':np.float64, 'c':np.float64}
        self.dir_path = get_tmp_path() / 'geopointlcoudlayer'
        self.pclayer = GeoPointCloudLayer(directory_path=self.dir_path, crs=NoneCRS(), bounds=self.backend_bounds,
                                           data_scheme=self.data_schema, point_density=1, build_pyramid=False)
        self.aoi = GeoShape(Polygon([(0,0), (100,0), (100,100), (0, 100)]), crs=NoneCRS())
        self.tiler = GeoBoxTiler3d(self.aoi, -10, 200, (20,20))

    def tearDown(self):
        self.pclayer.delete_permanently()

    def test_read_write(self):
        self.pclayer.write_data(self.pc)
        read_pc = self.pclayer.get_data(GeoBox3d(min=(50,50,50), max=(75,75,75), crs=NoneCRS()))
        self.assertEqual(read_pc.shape, (26,6))
        self.assertEqual(read_pc.x.to_numpy(), 62.5)

    def test_apply(self):
        self.pclayer.write_data(self.pc)
        means = []
        def write_mean(pc: GeoPointCloud):
            means.append(pc.xyz.mean(axis=0))
        self.pclayer.apply(self.tiler, write_mean)
        print(means)
