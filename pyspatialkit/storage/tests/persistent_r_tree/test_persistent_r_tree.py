#%%
import unittest

from pyspatialkit.storage import PersistenRTree

class TestPersistentRTree(unittest.TestCase):

    # def setUp(self):
    #     self.transformer = GeoCrsTransformer(GeoCrs.from_epsg(4326), GeoCrs.from_epsg(3857))

    def test_initialization(self):
        rtree = PersistenRTree(path=':memory:', )
        
