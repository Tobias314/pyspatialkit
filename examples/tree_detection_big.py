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

DATA_PATH = Path('/hpi/fs00/home/tobias.pietz/pyspatialkit/data/')
TMP_DATA_PATH = Path('/tmp/pyspatialkit/data/')
GEOSTORAGE_PATH = TMP_DATA_PATH / 'geostore/'
CRS = GeoCrs.from_epsg(25832)
AOI = GeoShape.from_shapefile(DATA_PATH / 'aoi.shp').to_crs(CRS)
AOI_BOUNDS = AOI.bounds

MIN_TREE_HEIGHT = 3
NDVI_THRESH = 0.3

@layerprocessor
def tree_detection(box: GeoBox2d, rgbi_layer: GeoRasterLayer, dom_layer: GeoPointCloudLayer, dgm_layer: GeoPointCloudLayer, res_layer: GeoRasterLayer):
    georect = box.to_georect()
    rgb_raster = rgbi_layer.get_data(georect)
    dom_pc = dom_layer.get_data(GeoBox2d.from_georect(georect))
    dgm_pc = dgm_layer.get_data(GeoBox2d.from_georect(georect))
    dgm_raster = GeoRaster(rgb_raster.georect, np.zeros((*rgb_raster.shape[:2], 1)))
    dgm_raster = dgm_pc.project_to_georaster(georaster=dgm_raster, value_field='height', interpolate_holes=True)
    dom_raster = GeoRaster(rgb_raster.georect, np.zeros((*rgb_raster.shape[:2], 1)))
    dom_raster = dom_pc.project_to_georaster(georaster=dom_raster, value_field='height', interpolate_holes=True)
    height_above_ground = dom_raster.data - dgm_raster.data
    ndvi = (rgb_raster.data[:,:,3] - rgb_raster.data[:,:,0]) / (rgb_raster.data[:,:,3] + rgb_raster.data[:,:,0])
    mask = (ndvi >= NDVI_THRESH) & (height_above_ground[:,:,0] >= MIN_TREE_HEIGHT)
    result = GeoRaster(georect, (mask * 255).astype(np.uint8))
    #print(result.data.sum())
    res_layer.write_data(result)

def main():
    storage = GeoStorage(GEOSTORAGE_PATH)
    rgbi_layer = storage.get_layer('rgbi')
    dom_layer = storage.get_layer('dom')
    dgm_layer = storage.get_layer('dgm')
    storage.delete_layer_permanently('result')
    res_layer = storage.add_raster_layer('result', num_bands = 1, dtype=np.uint8, crs=CRS, bounds=AOI_BOUNDS,
                                         build_pyramid=False)
    #sizes = [1000, 2000, 4000, 8000]
    size = 2000
    #for size in sizes:
    tiler = GeoBoxTiler2d(aoi=AOI, raster_size=(size,size), reference_crs=CRS)
    start_time = time.time()
    tree_detection(tiler=tiler, num_workers=20)(rgbi_layer, dom_layer, dgm_layer, res_layer)
    print("DONE! processing took {} seconds.".format(time.time() - start_time))
    
    
if __name__ == "__main__":
    main()