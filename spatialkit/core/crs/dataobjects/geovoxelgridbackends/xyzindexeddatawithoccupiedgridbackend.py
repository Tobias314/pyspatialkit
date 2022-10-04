from typing import Dict, List, Union, Tuple, Optional, Any

from .abstractbackend import AbstractBackend

import numpy as np
import pandas as pd


class XyzIndexedDataWithOccupiedGridBackend(AbstractBackend):

    def __init__(self):
        super().__init__()
        self._occupied = None
        self._data = None
        self._no_data_value = 0

    @classmethod
    def create_from_grid(cls, occupied: np.ndarray, data: Dict[str, np.ndarray] = {},
                         no_data_value: Any = 0) -> 'XyzIndexedDataWithOccupiedGridBackend':
        b = XyzIndexedDataWithOccupiedGridBackend()
        b._occupied = occupied
        b._emtpy_value = no_data_value
        indices = np.where(occupied != 0)
        md = pd.MultiIndex.from_arrays(indices.transpose(), names=['x', 'y', 'z'])
        for key, array in data:
            data[key] = array[indices[:, 0], indices[:, 1], indices[:, 2]]
        b._data = pd.DataFrame(data, index=md)
        return b

    @classmethod
    def create_from_coordinates(cls, coordinates: np.ndarray, data: Dict[str, np.ndarray] = {},
                                occupied_value: Union[bool, int, float, np.ndarray] = True,
                                dimensions: Optional[np.ndarray] = None,
                                no_data_value: Any = 0) -> 'XyzIndexedDataWithOccupiedGridBackend':
        b = XyzIndexedDataWithOccupiedGridBackend()
        b._no_data_value = no_data_value
        if dimensions is None:
            dimensions = coordinates.max(axis=0) + 1
        if not isinstance(occupied_value, np.ndarray):
            b._occupied = np.zeros(dimensions, dtype=type(occupied_value))
        else:
            b._occupied = np.zeros(dimensions, dtype=occupied_value.dtype)
        b._occupied[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = occupied_value
        md = pd.MultiIndex.from_arrays(coordinates.transpose(), names=['x', 'y', 'z'])
        b._data = pd.DataFrame(data, index=md)
        return b

    def get_grid(self, data_fields: List[str] = []) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        occupied = self._occupied
        data = {}
        for key in data_fields:
            data[key] = np.ones(self.dimensions(), dtype=self._data[key].dtype) * self._no_data_value
        return occupied, data

    def spatial_index_select(self, spatial_indices: np.ndarray, data_fields: List[str] = [], return_occupied=True) \
            -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        data = {}
        occupied = self._occupied[spatial_indices[:, 0], spatial_indices[:, 1], spatial_indices[:, 2]]
        md = pd.MultiIndex.from_arrays(spatial_indices[occupied].transpose(), names=['x', 'y', 'z'])
        for key in data_fields:
            d = self._data[key]
            ar = np.ones(spatial_indices.shape[0], dtype=d.dtype) * self._no_data_value
            ar[occupied] = d[md]
            data[key] = ar
        return occupied, data

    def spatial_index_set(self, spatial_index: Optional[np.ndarray],
                          data: Dict[str, np.ndarray] = {},
                          occupied_value: Union[bool, int, float, np.ndarray] = True):
        if spatial_index is not None:
            self._occupied[spatial_index[:, 0], spatial_index[:, 1], spatial_index[:, 2]] = occupied_value
            for key, array in data.items():
                md = pd.MultiIndex.from_arrays(spatial_index.transpose(), names=['x', 'y', 'z'])
                self._data.loc[md][key] = array
        else:
            self._occupied = occupied_value
            for key, array in data.items():
                self._data[key] = array

    def occupied_spatial_indices(self, return_spatial_indices=True, data_fields: List[str] = [],
                                 indexer_1d_bool: Optional[np.ndarray] = None) \
            ->Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        data = {}
        for key in data_fields:
            ar = self._data[key].to_numpy()
            if indexer_1d_bool is not None:
                ar = ar[indexer_1d_bool]
            data[key] = ar
        if return_spatial_indices:
            spatial_indices = self._data[('x', 'y', 'z')].to_numpy()
            if indexer_1d_bool is not None:
                spatial_indices = spatial_indices[indexer_1d_bool]
            return spatial_indices, data
        else:
            return None, data

    def occupied(self) -> np.ndarray:
        return self._occupied != self._no_data_value

    def dimensions(self) -> np.ndarray:
        return self._occupied.shape

    def data_fields(self) -> List[str]:
        return list(self._data.keys())

    def has_data_field(self, data_field_name: str) -> bool:
        return data_field_name in self._data