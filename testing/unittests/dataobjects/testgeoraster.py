#%%
import unittest
import sys
sys.path.append('../../../')
from testing.utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.crs.geocrs import NoneCRS

class TestGeoRaster(unittest.TestCase):

    def setUp(self):
        self.raster1 = GeoRaster(GeoRect((0,0), (1000,1000), crs=NoneCRS()), np.zeros((1000,1000)))
        self.raster2 = GeoRaster(GeoRect((750,300), (1750,1100), crs=NoneCRS()), np.ones((1000,1000)))
        print(matplotlib.rcParams['backend'])

    def test_merge_projected_other(self):
        self.raster1.merge_projected_other(self.raster2)
        self.assertEqual(self.raster1.data.sum(), 175000)
        #plt.imshow(self.raster1.data)

#t = TestGeoRaster()
#t.setUp()
#t.test_merge_projected_other()
