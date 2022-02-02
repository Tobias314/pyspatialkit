from typing import List, Tuple

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, MultiPoint

from ..dataobjects.geoshape import GeoShape

def bounds_from_polygon(polygon: GeoShape):
    bounds = []
    if isinstance(polygon, (MultiLineString, MultiPolygon, MultiPoint)):
        for geom in polygon:
            bounds.append(geom.bounds)
    else:
        bounds = [polygon.bounds]
    return bounds

def raster_bounds2d(bounds: List[Tuple[float, float, float, float]], raster_size: Tuple[float, float],
                    border_size: Tuple[float, float] = (0,0)) -> np.ndarray:
    """
        returns: origins of rasters as np.ndarray of shape (n,2)
    """
    raster_size = np.array(raster_size)
    border_size = np.array(border_size)
    inner_size = raster_size - border_size
    origins = []
    for min_x, min_y, max_x, max_y in bounds:
            xs = np.arange(min_x  - border_size[0], max_x - border_size[0], raster_size[0])
            ys = np.arange(min_y - border_size[1], max_y - border_size[1], raster_size[1])
            origins.append(np.stack(np.meshgrid(xs, ys, indexing='ij'),axis=2).reshape((-1,2)))
    return np.concatenate(origins, axis=0)


def raster_bounds3d(bounds: List[Tuple[float, float, float, float, float, float]], raster_size: Tuple[float, float, float],
                    border_size: Tuple[float, float, float] = (0,0, 0)) -> np.ndarray:
    """
        returns: origins of rasters as np.ndarray of shape (n,3)
    """
    raster_size = np.array(raster_size)
    border_size = np.array(border_size)
    inner_size = raster_size - 2 * border_size
    origins = []
    for min_x, min_y, min_z, max_x, max_y, max_z in bounds:
            xs = np.arange(min_x  - border_size[0], max_x  - border_size[0], raster_size[0])
            ys = np.arange(min_y  - border_size[1], max_y  - border_size[1], raster_size[1])
            zs = np.arange(min_z  - border_size[2], max_z  - border_size[2], raster_size[2])
            origins.append(np.stack(np.meshgrid(xs, ys, zs, indexing='ij'),axis=3).reshape((-1,3)))
    return np.concatenate(origins, axis=0)