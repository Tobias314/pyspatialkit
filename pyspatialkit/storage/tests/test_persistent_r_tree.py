#%%
import unittest
from typing import Optional

import numpy as np

from pyspatialkit.core.interfaces import BBoxSerializable
from pyspatialkit.storage import PersistentRTree

class SimpleBBoxData2d(BBoxSerializable):

    def __init__(self, bbox: np.ndarray, data: int):
        self.bbox = bbox
        self.data = data

    def to_bytes(self) -> bytes:
        return np.array([self.data]).astype(int).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, bbox: Optional[np.ndarray]) -> Optional['BBoxSerializable']:
        data = int(np.frombuffer(data, dtype=int, count=1)[0])
        return cls(bbox=bbox, data=data)

    def is_intersecting_bbox(self, bbox: np.ndarray) -> bool:
        return (self.bbox[:2] <= bbox[2:] & self.bbox[2:] >= bbox[:2]).all()

    def get_bounds(self)-> np.ndarray:
        return self.bbox

class TestPersistentRTree(unittest.TestCase):

    # def setUp(self):
    #     self.transformer = GeoCrsTransformer(GeoCrs.from_epsg(4326), GeoCrs.from_epsg(3857))

    # def test_initialization(self):
    #     rtree = PersistentRTree(root_path=':memory:', dimensions=2)

    def test_range_query(self):
        rtree = PersistentRTree(root_path=':memory:', object_type=SimpleBBoxData2d, dimensions=2)
        obj1 = SimpleBBoxData2d(np.array([0,0,2,2]), 42)
        obj2 = SimpleBBoxData2d(np.array([1,1,5,5]), 43)
        obj3 = SimpleBBoxData2d(np.array([10,10,12,12]), 44)
        rtree.insert([obj1, obj2, obj3])
        res = rtree.query_bbox(bbox=np.array([1,1,4,4]))
        res = {obj.data for obj in res}
        self.assertEqual(res, {42, 43})



        
