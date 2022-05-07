import unittest
import sys
sys.path.append('../../../../')
from ....utils import get_tmp_path, close_all_files_delete_dir
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import logging

from pyspatialkit.spacedescriptors.geobox3d import GeoBox3d
from pyspatialkit.crs.geocrs import NoneCRS, GeoCrs
from pyspatialkit.storage.mesh.geomeshlayer import GeoMeshLayer
from pyspatialkit.dataobjects.geomesh import GeoMesh

class TestGeoMeshLayer(unittest.TestCase):

    def setUp(self):
        self.bounds = [0,0,0, 20,20,20]
        self.tile_size = 10
        self.crs = GeoCrs.from_epsg(3857)
        self.mesh_layer = GeoMeshLayer(directory_path=get_tmp_path() / "geomeshlayer", crs=self.crs, bounds=self.bounds, tile_size=self.tile_size)
        vertices = np.array([[1,1,18], [15,1,18], [8,5,18]]).astype(float)
        faces = np.array([[0,1,2]])
        self.test_mesh = GeoMesh.from_vertices_faces(vertices, faces, crs=self.crs)
        self.mesh_layer.write_data(self.test_mesh)
        self.tileset3d_path = get_tmp_path() / "geomeshlayer_tileset"

    def tearDown(self):
        self.mesh_layer.delete_permanently()
        #shutil.rmtree(self.tileset3d_path, ignore_errors=True)

    def test_read(self):
        aoi = GeoBox3d([10,0,0], [20,10,10], crs=self.crs)
        res = self.mesh_layer.get_data(aoi)
        self.assertEqual(len(res), 1)
        aoi = GeoBox3d([0,0,0], [20,10,10], crs=self.crs)
        res = self.mesh_layer.get_data(aoi)
        self.assertEqual(len(res), 1)
        aoi = GeoBox3d([0,0,0], [9,9,9], crs=self.crs)
        res = self.mesh_layer.get_data(aoi)
        self.assertEqual(len(res), 1)
        aoi = GeoBox3d([11,11,11], [19,19,19], crs=self.crs)
        res = self.mesh_layer.get_data(aoi)
        self.assertEqual(len(res), 0)

    def test_visualization_tileset(self):
        tileset = self.mesh_layer.visualizer_tileset
        directory_path = self.tileset3d_path
        #directory_path = Path('/home/tobias/data/github/3dtiles_testbed/cesium_testbed/') / 'geomeshlayer_tileset'
        tileset.to_static_directory(directory_path = directory_path, max_per_file_depth=2)
