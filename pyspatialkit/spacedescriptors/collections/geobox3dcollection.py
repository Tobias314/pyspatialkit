from typing import Optional, Tuple, Union, List, Dict

import numpy as np
from pyproj import CRS

from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.geocrstransformer import GeoCrsTransformer
from ..tiles3dboundingvolume import Tiles3dBoundingVolume
from ...globals import DEFAULT_CRS
from ..geobox3d import GeoBox3d

class GeoBox3dCollection:
    """Describing a collection of volumes in 3D space. Described by min and max points and a crs.
    """

    def __init__(self, mins_maxs: np.ndarray, crs: GeoCrs = DEFAULT_CRS):
        self.crs = crs
        assert mins_maxs.shape[1] == 2 and mins_maxs.shape[2] == 3
        self.mins_maxs = mins_maxs

    @classmethod
    def from_mins_maxs(cls, mins: np.ndarray, maxs: np.ndarray, crs: GeoCrs = DEFAULT_CRS):
        mins_maxs = np.stack([mins, maxs], axis=1)
        return cls(mins_maxs, crs)

    @property
    def mins(self):
        return self.mins_maxs[:, 0]

    @property
    def maxs(self):
        return self.mins_maxs[:, 1]

    @property
    def centers(self):
        return (self.maxs + self.mins) / 2

    def __getitem__(self, index):
        min_max = self.mins_maxs[index]
        return GeoBox3d(min_max[0] , min_max[1], crs = self.crs)

    def __len__(self):
        return self.mins_maxs.shape[0]

    def get_edge_lengths(self) -> np.ndarray:
        return self.maxs - self.mins