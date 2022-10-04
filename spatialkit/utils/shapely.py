from typing import Union

from shapely.geometry import LinearRing, LineString, Point, Polygon, MultiLineString, MultiPoint, MultiPolygon

SHAPELY_SINGLE_GEOMETRY_TYPE = Union[LinearRing, LineString, Point, Polygon]
SHAPELY_MULTI_GEOMETRY_TYPE = Union[MultiLineString, MultiPoint, MultiPoint]
SHAPELY_GEOMETRY_TYPE = Union[SHAPELY_SINGLE_GEOMETRY_TYPE, SHAPELY_MULTI_GEOMETRY_TYPE]