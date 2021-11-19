from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Optional, Any

from ..voxelgridindextransformer import VoxelGridIndexTransformer

import numpy as np


class AbstractBackend(ABC):

    def __init__(self):
        self._index_transformer = VoxelGridIndexTransformer(self)
        self._no_data_value = 0

    @classmethod
    @abstractmethod
    def create_from_grid(cls, occupied: np.ndarray, data: Dict[str, np.ndarray] = {},
                         no_data_value: Any = 0) -> 'AbstractBackend':
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_from_coordinates(cls, coordinates: np.ndarray, data: Dict[str, np.ndarray] = {},
                                occupied_value: Union[bool, int, float, np.ndarray] = True,
                                dimensions: Optional[np.ndarray] = None,
                                no_data_value: Any = 0) -> 'AbstractBackend':
        raise NotImplementedError

    @abstractmethod
    def get_grid(self, data_fields: List[str] = []) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def spatial_index_select(self, spatial_indices: np.ndarray, data_fields: List[str] = [], return_occupied=True) \
            -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        raise NotImplementedError

    def bool_index_select(self, bool_index: np.ndarray, data_fields: List[str] = [], return_coordinates: bool = True) \
            -> Union[Tuple[np.ndarray, Dict[str, np.ndarray]], Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
        spatial_index = np.stack(np.where(bool_index), axis=1)
        if return_coordinates:
            res = self.spatial_index_select(spatial_index, data_fields=data_fields)
            return res[0], res[1], spatial_index
        else:
            return self.spatial_index_select(spatial_index, data_fields=data_fields)

    @abstractmethod
    def spatial_index_set(self, spatial_index: Optional[np.ndarray],
                          data: Dict[str, np.ndarray] = {},
                          occupied_value: Union[bool, int, float, np.ndarray] = True):
        raise NotImplementedError

    @abstractmethod
    def occupied(self) -> np.ndarray:
        raise NotImplementedError

    def occupied_spatial_indices(self, return_spatial_indices=True, data_fields: List[str] = [],
                                 indexer_1d_bool: Optional[np.ndarray] = None) \
            ->Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        spatial_indices = np.stack(np.where(self.occupied() != self._no_data_value), axis=1)
        if indexer_1d_bool is not None:
            spatial_indices = spatial_indices[indexer_1d_bool]
        data = self.spatial_index_select(spatial_indices, data_fields)[1]
        return spatial_indices, data


    @abstractmethod
    def dimensions(self) -> np.ndarray:
        raise NotImplementedError

    def index_transformer(self) -> VoxelGridIndexTransformer:
        return self._index_transformer

    @abstractmethod
    def data_fields(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def has_data_field(self, data_field_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> 'AbstractBackend':
        raise NotImplementedError

    def copy_from_spatial_index(self, spatial_index: np.ndarray, keep_dimensions: bool = False) \
            -> Tuple['AbstractBackend', np.ndarray]:
        si = spatial_index
        oc, data = self.spatial_index_select(si, self.data_fields())
        dimensions = self.dimensions()
        moved_origin = np.zeros(3, dtype=int)
        if not keep_dimensions:
            moved_origin = si.min(axis=0)
            si = si - moved_origin
            dimensions = si.max(axis=0) + 1
        return self.create_from_coordinates(si, data=data, occupied_value=oc, dimensions=dimensions), moved_origin

    def copy_from_bool_index(self, bool_index: np.ndarray, keep_dimensions: bool = False) \
            -> Tuple['AbstractBackend', np.ndarray]:
        oc, data, coords = self.bool_index_select(bool_index, data_fields=self.data_fields(), return_coordinates=True)
        dimensions = self.dimensions()
        moved_origin = np.zeros(3, dtype=int)
        if not keep_dimensions:
            moved_origin = coords.min(axis=0)
            coords = coords - moved_origin
            dimensions = coords.max(axis=0) + 1
        return self.create_from_coordinates(coords, data=data, occupied_value=oc, dimensions=dimensions), moved_origin

