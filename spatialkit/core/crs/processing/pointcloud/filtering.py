from typing import TYPE_CHECKING
import numpy as np

# only import type checking, not needed during runtime
if TYPE_CHECKING:
    from ...dataobjects.geopointcloud import GeoPointCloud
    from ...dataobjects.geovoxelgrid import GeoVoxelGrid

def create_filter_from_voxel_grid(point_cloud: 'GeoPointCloud', voxel_grid: 'GeoVoxelGrid') -> np.ndarray:
    s_indices = ((point_cloud.xyz.to_numpy() - voxel_grid.origin) // voxel_grid.voxel_size).astype(int)
    mask = (s_indices[:, 0] >= 0) & (s_indices[:, 0] < voxel_grid.shape[0])
    mask &= (s_indices[:, 1] >= 0) & (s_indices[:, 1] < voxel_grid.shape[1])
    mask &= (s_indices[:, 2] >= 0) & (s_indices[:, 2] < voxel_grid.shape[2])
    valid = s_indices[mask]
    s_indices = np.zeros(point_cloud.shape[0], dtype=bool)
    s_indices[mask] = voxel_grid.occupied()[valid[:, 0], valid[:, 1], valid[:, 2]]
    return s_indices