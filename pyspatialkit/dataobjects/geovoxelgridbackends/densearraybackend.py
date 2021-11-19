from typing import Dict, List, Union, Optional, Tuple, Any

import numpy as np

from .abstractbackend import AbstractBackend


class DenseArrayBackend(AbstractBackend):

    def __init__(self):
        super().__init__()
        self._occupied: Optional[np.ndarray] = None
        self._data: Dict[str, np.ndarray] = {}

    @classmethod
    def create_from_grid(cls, occupied: np.ndarray, data: Dict[str, np.ndarray] = {},
                         no_data_value: Any = 0) -> 'DenseArrayBackend':
        gb = DenseArrayBackend()
        gb._occupied = occupied
        gb._data = data
        gb._no_data_value = no_data_value
        return gb

    @classmethod
    def create_from_coordinates(cls, coordinates: np.ndarray,
                                occupied_value: Union[bool, int, float, np.ndarray] = True,
                                data: Dict[str, np.ndarray] = {},
                                dimensions: Optional[np.ndarray] = None,
                                no_data_value: Any = 0) -> 'DenseArrayBackend':
        gb = DenseArrayBackend()
        gb._no_data_value = no_data_value
        if dimensions is None:
            dimensions = coordinates.max(axis=0) + 1
        if not isinstance(occupied_value, np.ndarray):
            gb._occupied = np.ones(dimensions, dtype=type(occupied_value)) * no_data_value
        else:
            gb._occupied = np.ones(dimensions, dtype=occupied_value.dtype) * no_data_value
        gb._occupied[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = occupied_value
        gb._data = {}
        for key, array in data.items():
            gb._data[key] = np.ones(dimensions, dtype=array.dtype) * no_data_value
            gb._data[key][coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = array
        return gb

    def get_grid(self, data_fields: List[str] = []) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self._occupied, self._data

    def spatial_index_select(self, spatial_indices: np.ndarray, data_fields: List[str] = [], return_occupied=True) \
            -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        occupied = self._occupied[spatial_indices[:, 0], spatial_indices[:, 1], spatial_indices[:, 2]]
        data = {}
        for key in data_fields:
            data[key] = self._data[key][spatial_indices[:, 0], spatial_indices[:, 1], spatial_indices[:, 2]]
        return occupied, data

    def bool_index_select(self, bool_index: np.ndarray, data_fields: List[str] = [], return_coordinates: bool = True) \
        -> Union[Tuple[np.ndarray, Dict[str, np.ndarray]], Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
        occupied = self._occupied[bool_index]
        data = {}
        for key, array in self._data.items():
            data[key] = self._data[key][bool_index]
        if not return_coordinates:
            return occupied, data
        else:
            return occupied, data, np.stack(np.where(bool_index), axis=1)

    def spatial_index_set(self, spatial_index: Optional[np.ndarray],
                          data: Dict[str, np.ndarray] = {},
                          occupied_value: Union[bool, int, float, np.ndarray] = 1):
        if spatial_index is not None:
            self._occupied[spatial_index[:, 0], spatial_index[:, 1], spatial_index[:, 2]] = occupied_value
            for key, array in data.items():
                self._data[key][spatial_index[:, 0], spatial_index[:, 1], spatial_index[:, 2]] = array
        else:
            self._occupied = occupied_value
            for key, array in data.items():
                self._data[key] = array

    def occupied(self) -> np.ndarray:
        return self._occupied != self._no_data_value

    def dimensions(self):
        return np.array(self._occupied.shape)

    def data_fields(self) -> List[str]:
        return list(self._data.keys())

    def has_data_field(self, data_field_name: str) -> bool:
        return data_field_name in self._data

    def copy(self) -> 'DenseArrayBackend':
        b = DenseArrayBackend()
        b._occupied = self._occupied.copy()
        data = {}
        for key, array in self._data.items():
            data[key] = array.copy()
        b._data = data
        return b