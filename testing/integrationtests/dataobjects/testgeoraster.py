import unittest
import sys
sys.path.append('../../../../')
from testing.utils import get_testdata_path

import numpy as np
import logging
from pyproj import CRS

from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS, GeoCrs

class TestGeoRaster(unittest.TestCase):
    
    def test_load_file_compute_transformation(self):
        raster = GeoRaster.from_file(get_testdata_path() / "dop100rgbi_32_734_5748_2_st_2020.tif", band=[1,2,3])
        web_crs = GeoCrs(CRS.from_epsg(3857))
        raster.to_crs(web_crs)
        inputs = [np.array([raster.shape[0],0,1]), np.array([raster.shape[0],raster.shape[1],1]), np.array([0,raster.shape[1],1]), np.array([0,0,1])]
        outputs = [[1379969.31, 6770114.44, 1.], [1383194.67, 6769962.91, 1.],
                    [1383346.53, 6773197.13, 1.], [1380119.89, 6773348.82, 1.]]
        for i, input in enumerate(inputs):
            t = raster.transform @ input
            t /= t[2]
            sum = np.abs(t - np.array(outputs[i])).sum()
            self.assertLess(sum, 1)