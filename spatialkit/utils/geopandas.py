
from typing import Callable

import numpy as np
from geopandas import GeoSeries
import pygeos

def apply(geometries: GeoSeries, transformation: Callable[[np.ndarray], np.ndarray]) -> GeoSeries:
    geoms = pygeos.from_shapely(GeoSeries(geometries))
    return GeoSeries(pygeos.apply(geoms, transformation=transformation))

def projective_transform(geometries: GeoSeries, transformation_matrix: np.ndarray) -> GeoSeries:
    def transform_function(x: np.ndarray) -> np.ndarray:
        tmp = np.concatenate([x, np.ones([x.shape[0], 1])],axis=1) @ transformation_matrix.T
        return tmp[:,:2] / tmp[:,2,np.newaxis]
    return apply(geometries, transform_function)
