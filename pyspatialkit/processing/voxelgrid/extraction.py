from typing import Tuple

import numpy as np

from ...dataobjects.geovoxelgrid import GeoVoxelGrid

def compute_height_map(voxel_grid: GeoVoxelGrid, up_axis=2,
                       spatial_index_as_height=False, background_value=-1):
    height_map = voxel_grid.to_image(background_value=-1, projection_axis=up_axis, method='max_s_index')
    if spatial_index_as_height:
        return height_map
    mask = height_map != background_value
    height_map[mask] = height_map[mask] * voxel_grid.voxel_size + voxel_grid.origin[up_axis]
    return height_map

#def top_mask()

def extract_top(voxel_grid: GeoVoxelGrid, up_axis=2, keep_dimensions=True):
    height_map = compute_height_map(voxel_grid=voxel_grid, spatial_index_as_height=True, up_axis=up_axis)
    mask = height_map != -1
    s_indices = np.stack(np.where(mask), axis=1)
    stack = [s_indices[:, 0], s_indices[:, 1]]
    stack.insert(up_axis,  height_map[mask])
    s_indices = np.stack(stack, axis=1)
    return voxel_grid.copy(s_indices, keep_dimensions=keep_dimensions)

def split_by_height_map(voxel_grid: GeoVoxelGrid, height_map:np.ndarray, up_axis=2, keep_dimensions=False)->Tuple[GeoVoxelGrid]:
    s_indices = voxel_grid.occupied_spatial_indices()
    planar_axes = [0,1,2]
    del planar_axes[up_axis]
    hm_ids = s_indices[:, tuple(planar_axes)]
    mask = (s_indices[:, up_axis] >= height_map[hm_ids[:, 0], hm_ids[:, 1]])
    upper = voxel_grid.copy(s_indices[mask], keep_dimensions=keep_dimensions)
    lower = voxel_grid.copy(s_indices[np.logical_not(mask)], keep_dimensions=keep_dimensions)
    return upper, lower

def extract_top_layer(voxel_grid: GeoVoxelGrid, up_axis=2, keep_dimensions=True):
    height_map = compute_height_map(voxel_grid=voxel_grid, spatial_index_as_height=True, up_axis=up_axis)
    nocs = np.cumsum(np.logical_not(voxel_grid.occupied()), axis=up_axis)
    a,b = np.meshgrid(np.arange(height_map.shape[0]),  np.arange(height_map.shape[1]), indexing='ij')
    slicer = [a.flatten(),b.flatten()]
    slicer = (slicer[:up_axis],) + (height_map.flatten(),) + (slicer[up_axis:],)
    shape = (height_map.shape[:up_axis] + (1,) + height_map.shape[up_axis:])
    mask = nocs == nocs[slicer].reshape(shape)
    return voxel_grid.copy(index=mask, keep_dimensions=keep_dimensions)
