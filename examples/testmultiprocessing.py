import multiprocessing
import sys
sys.path.append('../')
from testing.utils import get_tmp_path, close_all_files_delete_dir

import numpy as np
import logging

from pyspatialkit.dataobjects.georaster import GeoRaster
from pyspatialkit.spacedescriptors.georect import GeoRect
from pyspatialkit.storage.raster.georasterlayer import GeoRasterLayer
from pyspatialkit.crs.geocrs import NoneCRS
from pyspatialkit.spacedescriptors.geobox2d import GeoBox2d
from pyspatialkit.tiling.geoboxtiler2d import GeoBoxTiler2d
from pyspatialkit.layerprocessing.decorators import layerprocessor

@layerprocessor
def identity(tile: GeoBox2d, height: GeoRasterLayer):
    #data=height.get_data(tile.to_georect())
    print('i')

def main():
    dir_path = get_tmp_path() / 'rasterlayer'
    crs = NoneCRS()
    raster_layer = GeoRasterLayer(directory_path=dir_path, num_bands=1, dtype=float, crs=crs, bounds=[0,0,100,100], build_pyramid=True)
    aoi = GeoBox2d.from_bounds([0,0,100, 100], crs=crs)
    tiler = GeoBoxTiler2d(aoi=aoi, raster_size=(50,50), reference_crs=crs)
    identity(tiler=tiler, num_workers=2)(raster_layer)


if __name__ == "__main__":
    main()