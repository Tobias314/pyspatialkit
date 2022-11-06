#%%
import unittest
from typing import Optional
from pathlib import Path
from typing import List

import numpy as np

from pyspatialkit.core.interfaces import BBoxSerializable
from pyspatialkit.storage import PersistentRTree, RTreeNode

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

    def _iterate_nodes(self, root_node: RTreeNode)->List[int]:
        res = [obj.data for obj in root_node.objects]
        for child in root_node.get_child_nodes():
            res += self._iterate_nodes(child)
        return res

    def test_tree_iteration(self):
        rtree = PersistentRTree(tree_path=':memory:', object_type=SimpleBBoxData2d, dimensions=2)
        objects = []
        for i in range(100):
            objects.append(SimpleBBoxData2d(np.array([i,i,i+2,i+2]), i))
        rtree.insert(objects)
        root = rtree.get_node(1)
        res = set(self._iterate_nodes(root_node=root))
        self.assertEqual(res, set(range(100)))

    def test_range_query(self):
        rtree = PersistentRTree(tree_path=':memory:', object_type=SimpleBBoxData2d, dimensions=2)
        obj1 = SimpleBBoxData2d(np.array([0,0,2,2]), 42)
        obj2 = SimpleBBoxData2d(np.array([1,1,5,5]), 43)
        obj3 = SimpleBBoxData2d(np.array([10,10,12,12]), 44)
        rtree.insert([obj1, obj2, obj3])
        #res = rtree.query_bbox(bbox=np.array([1,1,4,4]))
        res = rtree[1:4, 1:4]
        res = {obj.data for obj in res}
        self.assertEqual(res, {42, 43})



        
