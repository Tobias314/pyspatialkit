import unittest
import sys
sys.path.append('../../../../../')
from .....utils import get_tmp_path, close_all_files_delete_dir
from shutil import rmtree

import numpy as np
import pandas as pd
import logging

from pyspatialkit.spacedescriptors.geobox3d import GeoBox3d
from pyspatialkit.crs.geocrs import GeoCrs, NoneCRS
from pyspatialkit.storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer
from pyspatialkit.dataobjects.geopointcloud import GeoPointCloud
from pyspatialkit.dataobjects.tiles3d.pointcloud.geopointcloudtileset3d import GeoPointCloudTileset3d

class TestGeoPointCloudTileset3d(unittest.TestCase):

    def setUp(self):
        pcl_length = 100
        self.x = np.arange(pcl_length, dtype=np.float64)
        self.y = np.arange(pcl_length, dtype=np.float64)
        self.z = np.arange(pcl_length, dtype=np.float64)
        self.r = np.full((pcl_length,), 200, dtype=np.uint8)
        self.g = np.full((pcl_length,), 200, dtype=np.uint8)
        self.b = np.full((pcl_length,), 200, dtype=np.uint8)
        self.df = pd.DataFrame({'x':self.x,'y':self.y,'z':self.z,'r':self.r,'g':self.g,'b':self.b})
        self.pc = GeoPointCloud.from_pandas(self.df, crs=GeoCrs.from_epsg(3857), rgb_max=255)
        self.backend_bounds = [-10,-10,-10,200,200, 200]
        self.data_schema = self.pc.data_scheme
        self.dir_path = get_tmp_path() / 'geopointcloud3dtileset/'

    def tearDown(self):
        pass
        #rmtree(str(self.dir_path))

    def test_write_tileset_to_static_dictionary(self):
       tileset = GeoPointCloudTileset3d(self.pc, tile_size=(30,30,30))
       tileset.to_static_directory(self.dir_path)
