from abc import abstractmethod
from pathlib import Path

import numpy as np

from pyspatialkit import DEFAULT_CRS
from ..geolayer import GeoLayer
from ...crs.geocrs import GeoCRS, NoneCRS


class GeoRasterLayer(GeoLayer):
    def initialize(self, crs: GeoCRS = DEFAULT_CRS()):
        self._crs = crs

    def persist_data(dir_path: Path):
        raise NotImplementedError

    @abstractmethod
    def load_data(dir_path: Path):
        raise NotImplementedError

    @property
    def crs(self):
        return self._crs

    """"
    def get_raster_for_rect_and_band(self, georect: GeoRect, x_resolution: int, y_resolution: int, band=1,
                                     no_data_value=0):
        raise NotImplementedError
    """
    #TODO: change return type to GeoRaster
    @abstractmethod
    def get_raster_for_rect(self, georect: GeoRect, x_resolution: int, y_resolution: int, band=None, no_data_value=0) -> GeoRaster:
        """
        Get rasterdata of the layer for the specified georect in the specified resolution
        Args:
            georect ():
            x_resolution ():
            y_resolution ():
            band (): Either integer for one band or tuple of integers for multiple bands or None for all available bands
            no_data_value ():
        Returns:
        """
        raise NotImplementedError
        """
        if isinstance(band, int):
            return self.get_raster_for_rect_and_band(georect, x_resolution, y_resolution, band, no_data_value)
        else:
            result = np.ones((len(band), y_resolution, x_resolution), dtype=self.dtype) * no_data_value
            for b in band:
                result[b - 1] = self.get_raster_for_rect_and_band(georect, x_resolution, y_resolution, b, no_data_value)
            return result
        """

    @abstractmethod
    def writer_raster_data(self, georaster: GeoRaster):
        raise NotImplementedError

    def save_rect_to_geotiff(self, georect: GeoRect, x_resolution: int, y_resolution: int, path, band=1,
                             no_data_value=0):
        # TODO: write test
        # Todo, make a memory efficient function for very large images which uses rasterio feature to write small parts to disk
        georect.to_crs(self.crs)
        if isinstance(band, int):
            band = [band]
        result = np.ones((len(band), y_resolution, x_resolution), dtype=self.dtype) * no_data_value
        for b in band:
            result[b - 1] = self.get_raster_for_rect_and_band(georect, x_resolution, y_resolution, b, no_data_value)
        save_np_array_as_geotiff(result, georect, path)

    def get_coverage_polygons(self):
        """
            Returns polygons of the area covered by raster images in this layer.
                    Returns:
                            Geopandas dataframe consisting of polygons
        """
        raise NotImplementedError