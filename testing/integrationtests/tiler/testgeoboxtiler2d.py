import unittest
import sys
sys.path.append('../../../')
from ...utils import get_testdata_path

import numpy

from pyspatialkit.tiling.geoboxtiler2d import GeoBoxTiler2d
from pyspatialkit.crs.geocrs import GeoCrs
from pyspatialkit.dataobjects.geoshape import GeoShape

class TestGeoBoxTiler2d(unittest.TestCase):

    def setUp(self):
        print('lol1')
        self.gshp = GeoShape.from_shapefile(get_testdata_path() / "test_area_two_polygons.shp")
        self.tiler = GeoBoxTiler2d(self.gshp, (1000,1000), border_size=(100,100), reference_crs=GeoCrs.from_epsg(3857))

    def test_get_all_tiles(self):
        gdf = self.tiler.to_geopandas()
        gdf.to_file('tiles.shp')
        tiles = self.tiler.get_all_tiles()
        self.assertEqual(len(tiles), 15)
