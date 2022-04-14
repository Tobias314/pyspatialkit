import sys
sys.path.append('../tools/')

import time
from pathlib import Path
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import matplotlib
import tiledb
from tqdm.auto import tqdm
import geopandas as gpd

from pyspatialkit.storage.geostorage import GeoStorage
from pyspatialkit.dataobjects.geoshape import GeoShape
from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.dataobjects.geopointcloud import GeoPointCloud
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.crs.geocrs import GeoCrs
from pyspatialkit.spacedescriptors.geobox2d import GeoBox2d
from pyspatialkit.visualization.cesium.backend.server import start_server
from pyspatialkit.layerprocessing.decorators import layerprocessor
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer
from pyspatialkit.tiling.geoboxtiler2d import GeoBoxTiler2d

from geodownloader.geodownloader_saxony_anhalt import download_geodata



CRS = GeoCrs.from_epsg(25832)
MIN_TREE_HEIGHT = 3
NDVI_THRESH = 0.3

@layerprocessor
def tree_detection(box: GeoBox2d, rgbi_layer: GeoRasterLayer, dom_layer: GeoPointCloudLayer, dgm_layer: GeoPointCloudLayer, res_layer: GeoRasterLayer):
    georect = box.to_georect()
    rgbi_raster = rgbi_layer.get_data(georect)
    dom_pc = dom_layer.get_data(GeoBox2d.from_georect(georect))
    dgm_pc = dgm_layer.get_data(GeoBox2d.from_georect(georect))
    dgm_raster = GeoRaster(rgbi_raster.georect, np.zeros((*rgbi_raster.shape[:2], 1)))
    dgm_raster = dgm_pc.project_to_georaster(georaster=dgm_raster, value_field='height', interpolate_holes=True)
    dom_raster = GeoRaster(rgbi_raster.georect, np.zeros((*rgbi_raster.shape[:2], 1)))
    dom_raster = dom_pc.project_to_georaster(georaster=dom_raster, value_field='height', interpolate_holes=True)
    height_above_ground = dom_raster.data - dgm_raster.data
    ndvi = (rgbi_raster.data[:,:,3] - rgbi_raster.data[:,:,0]) / (rgbi_raster.data[:,:,3] + rgbi_raster.data[:,:,0])
    mask = (ndvi >= NDVI_THRESH) & (height_above_ground[:,:,0] >= MIN_TREE_HEIGHT)
    result = GeoRaster(georect, (mask * 255).astype(np.uint8))
    #print(result.data.sum())
    res_layer.write_data(result)

def main():
    aoi = GeoShape.from_shapefile('example_aoi.shp').to_crs(CRS)
    directory_path = Path('tree_detection_data/')
    geostorage_path = directory_path / 'geostore/'
    #We use a small script to donwload some free data, no use of PySpatialKit here!
    rgbi_input_dir = directory_path /'rgbi'
    if not rgbi_input_dir.is_dir():
        download_geodata(data_type='rgbi100', num_threads=2, output_dir=rgbi_input_dir,aoi=aoi.to_geopandas())
    dom_input_dir = directory_path / 'dom'
    if not dom_input_dir.is_dir():
        download_geodata(data_type='dom2', num_threads=2, output_dir=dom_input_dir, aoi=aoi.to_geopandas())
    dgm_input_dir = directory_path /'dgm'
    if not dgm_input_dir.is_dir():
        download_geodata(data_type='dgm2', num_threads=2, output_dir=dgm_input_dir, aoi=aoi.to_geopandas())

    #Create a geostorage if it does not exist
    if not geostorage_path.is_dir():
        print("CREATING DATA STORE...")
        #We use PySpatialKit to integrate all data into a single GeoStorage with three layers
        #First we create the GeoStorage
        geostorage_path.mkdir(exist_ok=True, parents=True)
        storage = GeoStorage(geostorage_path)
        #We add a raster layer for the rgbi data
        rgbi_layer = storage.add_raster_layer('rgbi', num_bands = 4, dtype=np.uint8, crs=CRS, bounds=aoi.bounds,
                                              build_pyramid=True)
        rgbi_layer.begin_pyramid_update_transaction()
        pathlist = list((directory_path /'rgbi').glob('*.tif'))
        for path in tqdm(pathlist):
            raster = GeoRaster.from_file(path)
            raster.data = raster.data.astype(np.uint8)
            rgbi_layer.write_data(raster)
        rgbi_layer.commit_pyramid_update_transaction()
        #Create digital terrain model layer
        pc_data_scheme = {'x': np.dtype('float64'), 'y': np.dtype('float64'), 'z': np.dtype('float64')}
        dgm_layer = storage.add_point_cloud_layer('dgm', crs=CRS, bounds=aoi.bounds, data_scheme=pc_data_scheme,
                                                  build_pyramid=False, point_density=1)
        pathlist = list((directory_path /'dgm').glob('*.xyz'))
        for path in tqdm(pathlist):
            pc = GeoPointCloud.from_xyz_file(path, crs=CRS)
            dgm_layer.write_data(pc)
        #Create digital surface model layer
        dom_layer = storage.add_point_cloud_layer('dom', crs=CRS, bounds=aoi.bounds, data_scheme=pc_data_scheme,
                                                  build_pyramid=False, point_density=1)
        pathlist = list((directory_path /'dom').glob('*.xyz'))
        for path in tqdm(pathlist):
            pc = GeoPointCloud.from_xyz_file(path, crs=CRS)
            dom_layer.write_data(pc)
    else:
        print("FOUND EXISTING DATA STORE, NO DATA STORE IS CREATED")
    
    #The actual processing
    #We reload all layers from disk (in case they already exisited and we did not create them above)
    storage = GeoStorage(geostorage_path)
    rgbi_layer = storage.get_layer('rgbi')
    dom_layer = storage.get_layer('dom')
    dgm_layer = storage.get_layer('dgm')
    #We delete any old results and create a new result layer
    storage.delete_layer_permanently('result')
    res_layer = storage.add_raster_layer('result', num_bands = 1, dtype=np.uint8, crs=CRS, bounds=aoi.bounds,
                                         build_pyramid=False)
    size = 1000
    tiler = GeoBoxTiler2d(aoi=aoi, raster_size=(size,size), reference_crs=CRS)
    start_time = time.time()
    print("STARTING THE PROCESSING")
    tree_detection(tiler=tiler, num_workers=3)(rgbi_layer, dom_layer, dgm_layer, res_layer)
    print("DONE! processing took {} seconds.".format(time.time() - start_time))
    
    
if __name__ == "__main__":
    main()


