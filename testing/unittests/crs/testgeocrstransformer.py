#%%
import unittest
import sys
sys.path.append('../../../')
from testing.utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

from pyspatialkit.crs.geocrs import GeoCrs
from pyspatialkit.crs.geocrstransformer import GeoCrsTransformer

class TestGeoCrsTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = GeoCrsTransformer(GeoCrs.from_epsg(4326), GeoCrs.from_epsg(3857))

    def test_transform_shapely_shape(self):
        shape = Polygon([(0,0), (0,1), (1,1), (0,1)])
        shape = self.transformer.transform_shapely_shape(shape)
        self.assertAlmostEqual(shape.boundary.coords[2][0], 111319.49079327357)
        self.assertAlmostEqual(shape.boundary.coords[2][1], 111325.1428663851)
        self.assertAlmostEqual(shape.boundary.coords[1][0], 0.0)
        self.assertAlmostEqual(shape.boundary.coords[1][1], 111325.1428663851)
