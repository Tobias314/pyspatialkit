from types import ClassMethodDescriptorType
from typing import Union, Optional, Tuple
from pathlib import Path
from geopandas.geodataframe import GeoDataFrame

import shapely
from shapely.geometry import Point, Polygon, LineString, LinearRing, MultiPolygon, MultiLineString, MultiPoint
import geopandas as gpd

from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer
from ..globals import get_geoviews, get_geoviews_back_map
from .dataobjectinterface import DataObjectInterface

class GeoShape(DataObjectInterface):

    def __init__(self, shape: Union[Point, Polygon, LineString], crs: GeoCrs):
        self.shape = shape
        self.crs = crs

    @classmethod
    def from_shapefile(cls, file_path: Union[str, Path]) -> 'GeoShape':
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

    @classmethod
    def from_bounds(cls, bounds: Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]], crs: GeoCrs) -> 'GeoShape':
        if len(bounds)==4:
            return cls.from_bounds_2d(bounds, crs)
        elif crs.is_geocentric:
            raise AttributeError("Cannot create 2d shape for bounds in geocentric CRS!")
        else:
            return cls.from_bounds_2d((*bounds[:2], *bounds[3:5]), crs)

    @classmethod
    def from_georect(cls, rect: 'georect.GeoRect'):
        return GeoShape(shape=rect.to_shapely(), crs=rect.crs)

    @classmethod
    def from_bounds_2d(cls, bounds: Tuple[float, float, float, float], crs: GeoCrs) -> 'GeoShape':
        rect = Polygon([bounds[:2], [bounds[2], bounds[1]], bounds[2:], (bounds[0], bounds[3])])
        return cls(rect, crs)

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
    
    def to_geopandas(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry = [self.shape], crs=self.crs.to_pyproj_crs())

    def to_geopandas(self):
        return gpd.GeoDataFrame(geometry=[self.to_shapely()], crs=self.crs.to_pyproj_crs())

    def to_geoviews(self):
        shp = self.to_crs(GeoCrs.from_epsg(4326),inplace=False)
        return get_geoviews().Shape(shp.to_shapely())

    def plot(self):
        return get_geoviews_back_map() * self.to_geoviews()
