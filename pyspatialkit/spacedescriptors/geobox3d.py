from typing import Optional, Tuple, Union, List
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer
from shapely.geometry import Polygon
import sentinelhub

from ..utils.linalg import projective_transform_from_pts

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer


class GeoBox3d:
    """Describing a Volume in 3D space. Described by a min and max point and a crs.
    """

    def __init__(self, min: Tuple[float, float, float], max: Tuple[float, float ,float], crs: GeoCrs = NoneCRS):
        self.crs = crs
        assert len(min)==3 and len(max)==3
        self.min = np.array(min)
        self.max = np.array(max)