import numpy as np
import geopandas as gpd
from sentinelhub import BBoxSplitter
import rasterio as rio

from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.geostorage import GeoStorage
from pyspatialkit.dataobjects.georaster import GeoRaster


def main():
    storage = GeoStorage(directory_path='./geostorage')
    raster1 = GeoRaster.from_file("../testdata/dop100rgbi_32_734_5748_2_st_2020.tif", band=[1,2,3])
    raster_layer = storage.add_raster_layer('raster_layer', 3, raster1.dtype, crs=raster1.crs, bounds=[732215,5747667, 737543,5750536])
    raster_layer.writer_raster_data(raster1)
    gdf = gpd.read_file("../testdata/test_area.shp")
    gdf = gdf.to_crs(raster_layer.crs.proj_crs)
    test_shape = gdf.geometry[0]
    bbox_splitter = BBoxSplitter([test_shape], gdf.crs, 5)
    bbox_list = np.array(bbox_splitter.get_bbox_list())
    for i, bbx in enumerate(bbox_list):
        rect = GeoRect.from_sentinelhub_bbox(bbx)
        georaster = raster_layer.get_raster_for_rect(rect, [500,500])
        georaster.to_file("test_{}.tif".format(i), ignore_not_affine=True)

if __name__=="__main__":
    main()