from types import ClassMethodDescriptorType
from typing import Union, Optional
from pathlib import Path
from geopandas.geodataframe import GeoDataFrame

import shapely
from shapely.geometry import Point, Polygon, LineString, LinearRing, MultiPolygon, MultiLineString, MultiPoint
import geoviews as gv
import geopandas as gpd

from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer
from ..globals import GEOVIEWS_BACK_MAP

class GeoShape:

    def __init__(self, shape: Union[Point, Polygon, LineString], crs: GeoCrs):
        self.shape = shape
        self.crs = crs

    @classmethod
    def from_shapefile(self, file_path: Union[str, Path]) -> 'GeoShape':
        gdf: GeoDataFrame = gpd.GeoDataFrame.from_file(file_path)
        if len(gdf)==0:
            raise AttributeError("Shapefile is empty")
        geom_probe = gdf.geometry[0]
        if len(gdf)==1:
            return GeoShape(geom_probe, GeoCrs(gdf.crs))
        multi_geom = None
        if isinstance(geom_probe, Polygon):
            multi_geom = MultiPolygon(list(gdf.geometry))
        elif isinstance(geom_probe, (LineString, LinearRing)):
            multi_geom = MultiLineString(list(gdf.geometry))
        elif isinstance(geom_probe, Point):
            multi_geom = MultiPoint(list(gdf.geometry))
        return GeoShape(multi_geom, GeoCrs(gdf.crs))        

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace=True) -> 'GeoShape':
        if crs_transformer is None:
            crs_transformer = GeoCrsTransformer(self.crs, new_crs)
        if inplace:
            self.crs = new_crs
            self.shape = crs_transformer.transform_shapely_shape(self.shape)
            return self
        else:
            return GeoShape(crs_transformer.transform_shapely_shape(self.shape), crs=new_crs)

    @property
    def bounds(self):
        return self.shape.bounds

    def to_shapely(self):
        return self.shape

    def to_geoviews(self):
        shp = self.to_crs(GeoCrs.from_epsg(4326),inplace=False)
        return gv.Shape(shp.to_shapely())

    def plot(self):
        return GEOVIEWS_BACK_MAP * self.to_geoviews()
