from typing import List, Optional, Union, Type, Callable, Tuple
from pathlib import Path

import numpy as np
import open3d as o3d

from . import geopointcloud
from .geovoxelgridbackends.abstractbackend import AbstractBackend
from .geovoxelgridbackends.densearraybackend import DenseArrayBackend

SPATIAL_INDICES_KEY = "__spatial_indices__"
METADATA_KEY = "__metadata___"

EPSILLON = 0.00000000001


class SpatialSlicer:
    def __init__(self, geovoxelgrid: 'GeoVoxelGrid', callback: Callable, keep_dimensions=False):
        self.voxel_grid = geovoxelgrid
        self.callback = callback
        self.keep_dimensions = keep_dimensions

    def __getitem__(self, x):
        if not isinstance(x, tuple):
            raise KeyError("Need 3d slice")
        for s in x:
            if not isinstance(s, slice):
                raise KeyError("Need 3d slice")
        min_x = x[0].start
        max_x = x[0].stop
        min_y = x[1].start
        max_y = x[1].stop
        min_z = x[2].start
        max_z = x[2].stop
        return self.callback(min_x, max_x, min_y, max_y, min_z, max_z, keep_dimensions=self.keep_dimensions)


class GeoVoxelGrid:

    def __init__(self, backend: AbstractBackend, voxel_size: float = 1,
                 origin: Union[List[float], np.ndarray] = [0, 0, 0], rgb_max=1.0):
        self.backend = backend
        if origin is None:
            origin = [0, 0, 0]
        for o in origin:
            if abs(o % voxel_size) > EPSILLON and abs(abs(o % voxel_size) - voxel_size) > EPSILLON:
                raise AttributeError("Origin needs to be a multiple of the voxel size but is" + str(origin))
        self.voxel_size = voxel_size
        self.origin = np.array(origin)
        self.rgb_max = rgb_max

    @classmethod
    def from_vox_file(cls, path) -> 'GeoVoxelGrid':
        from .voxio import read_vox_file #We do it here to avoid cyclic references
        return read_vox_file(path)

    @classmethod
    def from_coordinates_list_file(cls, path: Union[str, Path], backend: Type[AbstractBackend] = DenseArrayBackend):
        path = Path(path)
        s_indices = None
        metadata = None
        data = {}
        with open(path, 'rb') as f:
            d = np.load(f)
            for k in d:
                if k == SPATIAL_INDICES_KEY:
                    s_indices = d[k]
                elif k == METADATA_KEY:
                    metadata = d[k]
                else:
                    data[k] = d[k]
            origin = metadata[:3]
            voxel_size = metadata[3]
            rgb_max = metadata[4]
            backend = backend.create_from_coordinates(s_indices, data=data)
            return GeoVoxelGrid(backend, voxel_size=voxel_size, origin=origin, rgb_max=rgb_max)

    @classmethod
    def from_geopointcloud(cls, cloud: 'GeoPointCloud', voxel_size: float,
                           backend: Type[AbstractBackend] = DenseArrayBackend,
                           bounds: Optional[
                               Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None) -> 'GeoVoxelGrid':
        cloud = cloud.thin_by_grid(voxel_size, inplace=False)
        dimensions = None
        if bounds is None:
            xyz_min = cloud.xyz.min(axis=0).to_numpy()
            xyz = cloud.xyz.to_numpy() - xyz_min + voxel_size / 2
            rgb = cloud.rgb.to_numpy()
        else:
            xyz_min = np.array(bounds[0])
            xyz_max = np.array(bounds[1])
            dimensions = ((xyz_max - xyz_min) / voxel_size).astype(int) + 1
            xyz = cloud.xyz.to_numpy()
            rgb = cloud.rgb.to_numpy()
            mask = np.all(xyz >= xyz_min, axis=1) & np.all(xyz <= xyz_max, axis=1)
            xyz = xyz[mask]
            rgb = rgb[mask]
            xyz = xyz - xyz_min + voxel_size / 2
        indices = (xyz // voxel_size).astype(int)
        backend = backend.create_from_coordinates(indices, data={'r': rgb[:, 0], 'g': rgb[:, 1], 'b': rgb[:, 2]},
                                                  dimensions=dimensions)
        return GeoVoxelGrid(backend, voxel_size=voxel_size, origin=xyz_min, rgb_max=cloud.rgb_max)

    @classmethod
    def from_spatial_indices(cls, indices: np.ndarray, voxel_size: float = 1, origin: List[float] = [0, 0, 0],
                             rgb: Optional[np.ndarray] = None,
                             backend: Type[AbstractBackend] = DenseArrayBackend) -> 'GeoVoxelGrid':
        backend = backend.create_from_coordinates(indices, data={'r': rgb[:, 0], 'g': rgb[:, 1], 'b': rgb[:, 2]})
        rgb_max = 1.0
        if rgb is not None and rgb.dtype != np.float64 and rgb.dtype != np.float32:
            rgb_max = np.iinfo(rgb.dtype).max
        return GeoVoxelGrid(backend, voxel_size=voxel_size, origin=origin, rgb_max=rgb_max)

    @classmethod
    def from_o3d(cls, o3d_voxel_grid: o3d.geometry.VoxelGrid,
                 backend: Type[AbstractBackend] = DenseArrayBackend) -> 'GeoVoxelGrid':
        origin = o3d_voxel_grid.origin
        voxels = o3d_voxel_grid.get_voxels()
        indices = np.asarray([pt.grid_index for pt in voxels])
        rgb = np.asarray([pt.color for pt in voxels])
        backend = backend.create_from_coordinates(indices, data={'r': rgb[:, 0], 'g': rgb[:, 1], 'b': rgb[:, 2]})
        return GeoVoxelGrid(backend, o3d_voxel_grid.voxel_size, origin)

    @classmethod
    def from_others(cls, others: List['GeoVoxelGrid'],
                    backend: Type[AbstractBackend] = DenseArrayBackend) -> 'GeoVoxelGrid':
        assert len(others) >= 2
        vs = others[0].voxel_size
        for o in others[1:]:
            if o.voxel_size != vs:
                raise AttributeError("Cannot merge voxel grids with different voxel sizes at the moment")
        others = [other.to_geopointcloud() for other in others]
        others = geopointcloud.GeoPointCloud.from_others(others)
        return GeoVoxelGrid.from_geopointcloud(others, backend=backend, voxel_size=vs)

    @property
    def shape(self):
        return tuple(self.dimensions())

    def occupied(self) -> np.ndarray:
        return self.backend.occupied()

    def dimensions(self) -> np.ndarray:
        return self.backend.dimensions()

    def occupied_spatial_indices_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        s_ids = self.occupied_spatial_indices()
        return s_ids.min(axis=0), s_ids.max(axis=0)

    def occupied_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        min_s_ids, max_s_ids = self.occupied_spatial_indices_bounds()
        min_s_ids = min_s_ids * self.voxel_size + self.origin
        max_s_ids = max_s_ids * self.voxel_size + self.origin
        return min_s_ids, max_s_ids

    def has_rgb(self) -> bool:
        return self.backend.has_data_field('r') and self.backend.has_data_field('g') and self.backend.has_data_field(
            'b')

    def occupied_spatial_indices(self, indexer_1d_bool: Optional[np.ndarray] = None) -> np.ndarray:
        indices = self.backend.occupied_spatial_indices(return_spatial_indices=True, indexer_1d_bool=indexer_1d_bool)[0]
        return indices

    def linear_index_grid(self) -> np.ndarray:
        return self.backend.index_transformer().linear_index_grid()

    def cumsum_indices(self, invert_occupied: bool = False):
        self.backend.index_transformer().cumsum_indices(invert_occupied=invert_occupied)

    def voxel_coordinates(self, indexer_1d_bool: Optional[np.ndarray] = None) -> np.ndarray:
        return self.occupied_spatial_indices(indexer_1d_bool) * self.voxel_size + self.origin

    def spatial_to_linear_grid_index(self, spatial_index: np.ndarray):
        return self.backend.index_transformer().spatial_to_linear_grid_index(spatial_index)

    def linear_to_spatial_grid_index(self, linear_grid_index: Union[int, np.ndarray]):
        return self.backend.index_transformer().linear_to_spatial_grid_index(linear_grid_index=linear_grid_index)

    def set_data_spatial_indices(self, spatial_indices: np.ndarray, data_field_name: str, data: np.ndarray):
        self.backend.spatial_index_set(spatial_indices, {data_field_name: data})

    def set_rgb_spatial_indices(self, spatial_indices: np.ndarray, rgb: Union[List[float], np.ndarray]):
        if not self.has_rgb():
            raise KeyError("Voxel grid has no rgb")
        rgb = np.array(rgb)
        if len(rgb.shape) == 2:
            self.backend.spatial_index_set(spatial_indices, {'r': rgb[:, 0], 'g': rgb[:, 1], 'b': rgb[:, 2]},
                                           occupied_value=True)
        else:
            self.backend.spatial_index_set(spatial_indices, {'r': rgb[0], 'g': rgb[1], 'b': rgb[2]},
                                           occupied_value=True)

    def delete_voxels_spatial_indices(self, spatial_indices: np.ndarray):
        self.backend.spatial_index_set(spatial_index=spatial_indices, occupied_value=False)

    def __getitem__(self, item):
        return self.copy(index=item)

    def copy(self, index: Optional[np.ndarray] = None,
             new_occupied_value: Optional[Union[bool, float, np.ndarray]] = None,
             keep_dimensions: bool = False) -> 'GeoVoxelGrid':
        moved_origin = np.zeros(3)
        if index is None:
            backend = self.backend.copy()
        else:
            if index.dtype == bool and len(index.shape) == 3:
                backend, moved_origin = self.backend.copy_from_bool_index(bool_index=index,
                                                                          keep_dimensions=keep_dimensions)
            elif ((index.dtype == int or index.dtype == np.uint64 or index.dtype == np.uint32) and len(index.shape) == 2
                  and index.shape[1] == 3):
                backend, moved_origin = self.backend.copy_from_spatial_index(spatial_index=index,
                                                                             keep_dimensions=keep_dimensions)
            else:
                raise KeyError("this form of index is not supported")
        if new_occupied_value:
            backend.spatial_index_set(spatial_index=None, occupied_value=new_occupied_value)
        origin = self.origin + moved_origin * self.voxel_size
        return GeoVoxelGrid(backend, voxel_size=self.voxel_size, origin=origin, rgb_max=self.rgb_max)

    def slice_sindices(self, min_x: int = None, max_x: int = None, min_y: int = None, max_y: int = None,
                       min_z: int = None,
                       max_z: int = None, keep_dimensions=False):
        s_indices = self.occupied_spatial_indices()
        mask = np.ones(s_indices.shape[0], dtype=bool)
        if min_x is not None:
            mask &= s_indices[:, 0] > min_x
        if max_x is not None:
            mask &= s_indices[:, 0] < max_x
        if min_y is not None:
            mask &= s_indices[:, 1] > min_y
        if max_y is not None:
            mask &= s_indices[:, 1] < max_y
        if min_z is not None:
            mask &= s_indices[:, 2] > min_z
        if max_z is not None:
            mask &= s_indices[:, 2] < max_z
        return self.copy(s_indices[mask], keep_dimensions=keep_dimensions)

    def slice_coordinates(self, min_x: float = None, max_x: float = None, min_y: float = None, max_y: float = None,
                          min_z: float = None, max_z: float = None, keep_dimensions=False):
        if (min_x is not None):
            min_x = int((min_x - self.origin[0]) / self.voxel_size)
        if (max_x is not None):
            max_x = int((max_x - self.origin[0]) / self.voxel_size) + 1
        if (min_y is not None):
            min_y = int((min_y - self.origin[1]) / self.voxel_size)
        if (max_y is not None):
            max_y = int((max_y - self.origin[1]) / self.voxel_size) + 1
        if (min_z is not None):
            min_z = int((min_z - self.origin[2]) / self.voxel_size)
        if (max_z is not None):
            max_z = int((max_z - self.origin[2]) / self.voxel_size) + 1
        return self.slice_sindices(min_x, max_x, min_y, max_y, min_z, max_z, keep_dimensions=keep_dimensions)

    def coordinate_slicer(self, keep_dimensions: bool = False):
        return SpatialSlicer(self, self.slice_coordinates, keep_dimensions=keep_dimensions)

    def spatial_indices_slicer(self, keep_dimensions: bool = False):
        return SpatialSlicer(self, self.slice_sindices, keep_dimensions=keep_dimensions)

    def select_from_img_mask(self, img: np.ndarray, axis=2, keep_dimensions=False):
        assert (self.shape[:axis] + self.shape[axis + 1:]) == img.shape
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = np.newaxis
        slicer = tuple(slicer)
        selector = img[slicer]
        selector = selector.repeat(self.shape[axis], axis)
        return self.copy(selector, keep_dimensions=keep_dimensions)

    def to_o3d_pointcloud(self, indexer_1d_bool: Optional[np.ndarray] = None,
                          normalize_rgb=True) -> o3d.geometry.PointCloud:
        data_fields = []
        if self.has_rgb():
            data_fields = ['r', 'g', 'b']
        coords, data = self.backend.occupied_spatial_indices(data_fields=data_fields, indexer_1d_bool=indexer_1d_bool)
        xyz_o3d = o3d.utility.Vector3dVector(coords * self.voxel_size + self.origin + (self.voxel_size / 2))
        pc3d = o3d.geometry.PointCloud()
        pc3d.points = xyz_o3d
        if self.has_rgb():
            rgb = np.stack(list(data.values()), axis=1)
            if normalize_rgb:
                rgb /= self.rgb_max
            rgb_o3d = o3d.utility.Vector3dVector(rgb)
            pc3d.colors = rgb_o3d
        return pc3d

    def to_geopointcloud(self, indexer_1d_bool: Optional[np.ndarray] = None) -> geopointcloud.GeoPointCloud:
        return geopointcloud.GeoPointCloud.from_o3d(self.to_o3d_pointcloud(indexer_1d_bool, normalize_rgb=False),
                                                    rgb_max=self.rgb_max)

    def to_o3d(self, indexer_1d_bool: Optional[np.ndarray] = None) -> o3d.geometry.VoxelGrid:
        return o3d.geometry.VoxelGrid.create_from_point_cloud(self.to_o3d_pointcloud(indexer_1d_bool),
                                                              voxel_size=self.voxel_size)

    def to_image(self, data_value: Optional[str] = None, background_value: float = 0, projection_axis: int = 2,
                 method='max'):
        """
        data_value=None uses occupied as pixel value
        """
        if data_value is None:
            grid, _ = self.backend.get_grid()
        else:
            _, grid = self.backend.get_grid(data_fields=[data_value])
        if projection_axis != 2:
            axes = [0, 1, 2]
            axes[projection_axis] = 2
            axes[2] = projection_axis
            grid = np.transpose(grid, axes)
        if method == "max":
            return grid.max(axis=2)
        elif method == "max_s_index":
            mask = grid != 0
            val = grid.shape[2] - np.flip(mask, axis=2).argmax(axis=2) - 1
            return np.where(mask.any(axis=2), val, background_value)
        elif method == "min_s_index":
            mask = grid != 0
            val = mask.argmax(axis=2)
            return np.where(mask.any(axis=2), val, background_value)
        elif method == "sum":
            return grid.sum(axis=2)
        else:
            raise AttributeError("method: " + method + " not supported!")
        return None

    def save_coordinates_list(self, path: Union[str, Path]):
        path = Path(path)
        s_indices, data = self.backend.occupied_spatial_indices(return_spatial_indices=True,
                                                                data_fields=self.backend.data_fields())
        if SPATIAL_INDICES_KEY in data.keys():
            raise KeyError("Cannot save data which has a " + SPATIAL_INDICES_KEY + " key")
        keys = [METADATA_KEY, SPATIAL_INDICES_KEY] + list(data.keys())
        meta_data_array = np.array([self.origin[0], self.origin[1], self.origin[2], self.voxel_size, self.rgb_max],
                                   dtype=np.float64)
        arrays = [meta_data_array, s_indices]
        for a in data.values():
            arrays.append(a)
        d = dict(zip(keys, arrays))
        with open(path, 'wb') as f:
            np.savez(f, **d)

    def _indexer_1d_bool_from_indexer_and_bounds(self, indexer_1d_bool: Optional[np.ndarray] = None, min_x=None,
                                                 max_x=None,
                                                 min_y=None, max_y=None, min_z=None, max_z=None):
        if min_x is not None or max_x is not None or min_y is not None or max_y is not None or min_z is not None or max_z is not None:
            vc = self.voxel_coordinates()
            mask = np.ones(vc.shape[0], dtype=bool)
            if min_x is not None:
                mask &= vc[:, 0] > min_x
            if max_x is not None:
                mask &= vc[:, 0] < max_x
            if min_y is not None:
                mask &= vc[:, 0] > min_y
            if max_y is not None:
                mask &= vc[:, 0] < max_y
            if min_z is not None:
                mask &= vc[:, 0] > min_z
            if max_z is not None:
                mask &= vc[:, 0] < max_z
            if indexer_1d_bool is not None:
                indexer_1d_bool &= mask
            else:
                indexer_1d_bool = mask
        return indexer_1d_bool

    def plot_o3d(self, indexer_1d_bool: Optional[np.ndarray] = None, min_x=None, max_x=None, min_y=None, max_y=None,
                 min_z=None, max_z=None):
        from open3d.visualization import draw_geometries
        
        indexer_1d_bool = self._indexer_1d_bool_from_indexer_and_bounds(indexer_1d_bool, min_x, max_x, min_y, max_y,
                                                                        min_z, max_z)
        draw_geometries([self.to_o3d(indexer_1d_bool=indexer_1d_bool)])

    def plot_o3dj(self, indexer_1d_bool: Optional[np.ndarray] = None, min_x=None, max_x=None, min_y=None, max_y=None,
                  min_z=None, max_z=None):
        from open3d.web_visualizer import draw as draw_jupyter
        indexer_1d_bool = self._indexer_1d_bool_from_indexer_and_bounds(indexer_1d_bool, min_x, max_x, min_y, max_y,
                                                                        min_z, max_z)
        draw_jupyter([self.to_o3d(indexer_1d_bool=indexer_1d_bool)])

    def to_vox_file(self, path: Union[str, Path], vox_color: int = 10):
        from .voxio import write_vox_file #We do it here to avoid cyclic references
        write_vox_file(self, path, vox_color)
