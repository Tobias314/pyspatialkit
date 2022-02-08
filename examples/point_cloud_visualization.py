import sys
sys.path.append('../')
from pathlib import Path
import shutil
import math

import numpy as np
import pandas as pd
import logging
from shapely.geometry import Polygon

from pyspatialkit.spacedescriptors.geobox3d import GeoBox3d
from pyspatialkit.dataobjects.geoshape import GeoShape
from pyspatialkit.crs.geocrs import GeoCrs
from pyspatialkit.storage.pointcloud.tiledbsparsebackend import TileDbSparseBackend
from pyspatialkit.storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer
from pyspatialkit.dataobjects.geopointcloud import GeoPointCloud
from pyspatialkit.tiling.geoboxtiler3d import GeoBoxTiler3d
from pyspatialkit.storage.geostorage import GeoStorage
from pyspatialkit.visualization.cesium.backend.server import start_server
from pyspatialkit.spacedescriptors.georect import GeoRect

from pyspatialkit.dataobjects.tiles3d.pointcloud.geopointcloudtileset3d import GeoPointCloudTileset3d

from testing.utils import get_tmp_path, get_testdata_path

def get_simple_storage():
    crs = GeoCrs.from_epsg(4979)
    num_points = 100
    dir_path = get_tmp_path() / 'geostorage_simple_tmp'
    if dir_path.is_dir():
        shutil.rmtree(str(dir_path), ignore_errors=True)
    storage = GeoStorage(dir_path)
    layer_name = 'pointcloud_layer'
    pc_data_scheme = {'x': np.dtype('float64'), 'y': np.dtype('float64'), 'z': np.dtype('float64')}
    min_pt = [0, 0, 500]
    x = np.rad2deg(np.linspace(0,1,10, dtype=np.float64)) + min_pt[0]
    y = np.rad2deg(np.linspace(0,1,10, dtype=np.float64)) + min_pt[1]
    z = np.linspace(0,0.99,10, dtype=np.float64) + min_pt[2]
    df = pd.DataFrame({'x':x,'y':y,'z':z})
    pc = GeoPointCloud.from_pandas(df, crs=crs)
    max_pt = [x.max(), y.max(), 501]
    bounds = [min_pt[0],min_pt[1],min_pt[2], max_pt[0],max_pt[1],max_pt[2]]
    if storage.has_layer(layer_name):
        pc_layer = storage.get_layer(layer_name)
    else:
        ts = math.degrees(0.5)
        pc_layer = storage.add_point_cloud_layer(layer_name, crs=crs, bounds=bounds, data_scheme=pc_data_scheme,
                                                 point_density=1, build_pyramid=False, backend_space_tile_size=(ts,ts,0.5))
        pc_layer.write_data(pc)
    return storage

def get_simple_storage2():
    crs = GeoCrs.from_epsg(25832)
    num_points = 100
    dir_path = get_tmp_path() / 'geostorage_simple_tmp2'
    if dir_path.is_dir():
        shutil.rmtree(str(dir_path), ignore_errors=True)
    storage = GeoStorage(dir_path)
    layer_name = 'pointcloud_layer'
    pc_data_scheme = {'x': np.dtype('float64'), 'y': np.dtype('float64'), 'z': np.dtype('float64')}
    min_pt = [0, 0, 500]
    num_points = 100
    x = (np.linspace(0,100,num_points, dtype=np.float64)) + min_pt[0]
    y = (np.linspace(0,100,num_points, dtype=np.float64)) + min_pt[1]
    z = np.linspace(0,200,num_points, dtype=np.float64) + min_pt[2]
    df = pd.DataFrame({'x':x,'y':y,'z':z})
    pc = GeoPointCloud.from_pandas(df, crs=crs)
    max_pt = [x.max(), y.max(), z.max()]
    bounds = [min_pt[0],min_pt[1],min_pt[2], max_pt[0],max_pt[1],max_pt[2]]
    if storage.has_layer(layer_name):
        pc_layer = storage.get_layer(layer_name)
    else:
        ts = 200
        pc_layer = storage.add_point_cloud_layer(layer_name, crs=crs, bounds=bounds, data_scheme=pc_data_scheme,
                                                 point_density=1, build_pyramid=False, backend_space_tile_size=(ts,ts,ts))
        pc_layer.write_data(pc)
    return storage

def get_dom_storage():
    crs = GeoCrs.from_epsg(25832)
    bounds = [733856, 5747905,-200, 741405, 5752795, 200]
    aoi_rect = GeoRect.from_bounds(bounds, crs=crs)
    dir_path = get_tmp_path() / 'geostorage_tmp'
    storage = GeoStorage(dir_path)
    layer_name = 'pointcloud_layer'
    pc_data_scheme = {'x': np.dtype('float64'), 'y': np.dtype('float64'), 'z': np.dtype('float64')}
    if storage.has_layer(layer_name):
        pc_layer = storage.get_layer(layer_name)
    else:
        pc_layer = storage.add_point_cloud_layer(layer_name, crs=crs, bounds=bounds, data_scheme=pc_data_scheme,
                                                 point_density=1, build_pyramid=False)
        dom_folder = get_testdata_path() / 'dom'
        pathlist = dom_folder.glob('*.xyz')
        for path in pathlist:
            pc = GeoPointCloud.from_xyz_file(path, crs=crs)
            pc_layer.write_data(pc)
    return storage

def main():
    #storage = get_simple_storage2()
    storage = get_dom_storage()
    start_server(storage)        


if __name__=='__main__':
    main()