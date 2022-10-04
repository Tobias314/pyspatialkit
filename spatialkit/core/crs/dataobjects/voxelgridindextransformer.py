from typing import Union, Optional

import numpy as np

class VoxelGridIndexTransformer:

    def __init__(self, dimensions_provider):
        self.dim_provider = dimensions_provider

    #def spatial_indices(self, indexer_1d_bool: Optional[np.ndarray] = None) -> np.ndarray:
    #    indices = np.stack(np.where(self.dim_provider.occupied() == 1), axis=1)
    #    if indexer_1d_bool is not None:
    #        indices = indices[indexer_1d_bool]
    #    return indices

    def spatial_index_grid(self) -> np.ndarray:
        dims = self.dim_provider.dimensions()
        x = np.arange(dims[0])
        y = np.arange(dims[1])
        z = np.arange(dims[2])
        return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=3)

    def linear_index_grid(self) -> np.ndarray:
        dims = self.dim_provider.dimensions()
        return np.arange(np.array(dims).prod()).reshape(dims)

    def spatial_to_linear_grid_index(self, spatial_grid_index: np.ndarray):
        dims = self.dim_provider.dimensions()
        fac1 = dims[2]
        fac2 = dims[1] * dims[2]
        if len(spatial_grid_index.shape) == 1:
            return spatial_grid_index[2] + spatial_grid_index[1] * fac1 + spatial_grid_index[0] * fac2
        else:
            return spatial_grid_index[:][2] + spatial_grid_index[:][1] * fac1 + spatial_grid_index[:][0] * fac2

    def linear_to_spatial_grid_index(self, linear_grid_index: Union[int, np.ndarray]):
        dims = self.dim_provider.dimensions()
        fac1 = dims[2]
        fac2 = dims[1] * dims[2]
        if not isinstance(linear_grid_index, np.ndarray):
            return np.array([linear_grid_index // fac2, linear_grid_index % fac2 // fac1, linear_grid_index % fac1])
        else:
            x = linear_grid_index // fac2
            y = linear_grid_index % fac2 // fac1
            z = linear_grid_index % fac1
            return np.stack([x, y, z], axis=1)

    def cumsum_indices(self, invert_occupied: bool = False):
        occupied = self.dim_provider.occupied()
        if invert_occupied:
            occupied = 1 - occupied
        return np.cumsum(occupied.flatten()) - 1

    def indexer_to_1d_bool(self, indexer: np.ndarray) -> np.ndarray:
        if len(indexer.shape) == 1 and indexer.dtype == np.bool:
            return indexer
        #elif len(indexer.shape) == 2 and (indexer.dtype == np.int or indexer.dtype == np.uint64 or indexer.dtype == np.uint32):
        else:
            raise NotImplementedError