import sys
sys.path.append('../')
from pathlib import Path
import shutil

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

def main():
    crs = GeoCrs.from_epsg(25832)
    # num_points = 50
    # dir_path = get_tmp_path() / 'geostorage_simple_tmp'
    # if dir_path.is_dir():
    #     shutil.rmtree(str(dir_path))
    # min_pt = [733800, 5747800]
    # x = np.arange(num_points, dtype=np.float64) * 100 / num_points + min_pt[0]
    # y = np.arange(num_points, dtype=np.float64) * 100 / num_points + min_pt[1] 
    # z = np.arange(num_points, dtype=np.float64) * 100 / num_points + 500
    # df = pd.DataFrame({'x':x,'y':y,'z':z})
    # pc = GeoPointCloud.from_pandas(df, crs=crs)
    # bounds = [min_pt[0]-100,min_pt[1]-100,500,min_pt[0]+100,min_pt[1]+100,700]
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
        #pc_layer.write_data(pc)
    start_server(storage)        


if __name__=='__main__':
    main()