from typing import Optional, Tuple, Union, List, Dict

import numpy as np
from pyproj import CRS

from ...crs.geocrs import GeoCrs, NoneCRS
from ...crs.geocrstransformer import GeoCrsTransformer
from ..tiles3dboundingvolume import Tiles3dBoundingVolume
from ...globals import DEFAULT_CRS
from ..geobox3d import GeoBox3d
from ..geobox2d import GeoBox2d
from ...tiling.abstracttiler import AbstractTiler

class GeoBoxCollection(AbstractTiler):
    """Describing a collection of volumes in 3D space. Described by min and max points and a crs.
    """

    def __init__(self, mins_maxs: np.ndarray, crs: GeoCrs = DEFAULT_CRS):
        self.crs = crs
        assert mins_maxs.shape[1] == 2 and (mins_maxs.shape[2] == 3 or mins_maxs.shape[2] == 2)
        self.mins_maxs = mins_maxs

    def partition(self, num_partitions) -> List['GeoBoxCollection']:
        mins_maxs_partitions = np.array_split(self.mins_maxs, num_partitions)
        res = []
        for partition in mins_maxs_partitions:
            res.append(GeoBoxCollection(partition, self.crs))
        return res

    @classmethod
    def from_mins_maxs(cls, mins: np.ndarray, maxs: np.ndarray, crs: GeoCrs = DEFAULT_CRS) -> 'GeoBoxCollection':
        mins_maxs = np.stack([mins, maxs], axis=1)
        return cls(mins_maxs, crs)

    @classmethod
    def from_geobox_list(cls, geoboxes: List[Union[GeoBox2d, GeoBox3d]]) -> 'GeoBoxCollection':
        mins_maxs = []
        crs = geoboxes[0].crs
        for box in geoboxes:
            if box.crs != crs:
                raise ValueError("All GeoBoxes in the list need to have the same crs to combine them to a GeoBoxCollection")
            mins_maxs.append([box.min, box.max])
        mins_maxs = np.array(mins_maxs)
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

    def __iter__(self) -> 'GeoBoxCollectionIterator':
        return GeoBoxCollectionIterator(self)

    def __getitem__(self, index):
        min_max = self.mins_maxs[index]
        if min_max.shape[1] == 2:
            return GeoBox2d(min_max[0] , min_max[1], crs = self.crs)
        else:
            return GeoBox3d(min_max[0] , min_max[1], crs = self.crs)

    def __len__(self):
        return self.mins_maxs.shape[0]

    def get_edge_lengths(self) -> np.ndarray:
        return self.maxs - self.mins


        
class GeoBoxCollectionIterator:

    def __init__(self, tiler: GeoBoxCollection):
        self.tiler = tiler
        self.current = 0

    def __iter__(self) -> 'GeoBoxCollectionTilerIterator':
        return self

    def __next__(self) -> GeoBox2d:
        if self.current >= len(self.tiler):
            raise StopIteration
        tmp = self.tiler[self.current]
        self.current+=1
        return tmp