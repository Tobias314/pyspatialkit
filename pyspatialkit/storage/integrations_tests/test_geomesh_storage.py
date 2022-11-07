#%%
import unittest
from typing import Optional
from pathlib import Path
from typing import List
import tempfile
import shutil

import numpy as np

from pyspatialkit.core import GeoCrs
from pyspatialkit.core import GeoMesh

from pyspatialkit.storage import GeoMeshLayer

class TestPersistentRTree(unittest.TestCase):

    def setUp(self):
        self.crs = GeoCrs.from_epsg(3857)
        self.test_mesh1 = GeoMesh.get_box_mesh(crs=self.crs)
        self.test_mesh1.vertices = self.test_mesh1.vertices * 1000

    def test_storage_and_tileset3d(self):
        tmp_dir = tempfile.TemporaryDirectory()
        mesh_layer = GeoMeshLayer(directory_path=Path(tmp_dir.name) / 'GeoMeshLayer', crs=self.crs)
        mesh_layer.write_data(self.test_mesh1)
        tileset = mesh_layer.visualizer_tileset
        tileset_dir = Path('./Tileset')
        if tileset_dir.is_dir():
            shutil.rmtree(str(tileset_dir))
        tileset.to_static_directory(directory_path=tileset_dir)



        
