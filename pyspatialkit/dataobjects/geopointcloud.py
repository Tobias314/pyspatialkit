import json
from typing import Union, Tuple, List, Optional, Type, TypeVar
from pathlib import Path

from crs.geocrs import NoneCRS
from dataobjects.georaster import GeoRaster
from spacedescriptors.georect import GeoRect

T = TypeVar('T', bound='TrivialClass')

import open3d as o3d
import numpy as np
import pandas as pd
import pylas
import pdal
from sklearn.decomposition import PCA
from matplotlib import cm

from . import geomesh
from ..processing.pointcloud.utils import points3d_to_image
from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer


class GeoPointCloud:

    def __init__(self):
        self.data = pd.DataFrame(columns=['x', 'y', 'z'])
        self.rgb_max = 1.0
        self.crs = NoneCRS
        self._reset_cached()

    @classmethod
    def from_numpy_arrays(self, xyz: np.ndarray, crs: GeoCrs=NoneCRS, rgb: Union[np.ndarray, None] = None, rgb_max=1.0,
                          normals_xyz: Union[np.ndarray, None] = None, curvature: Union[np.ndarray, None] = None,
                          data_axis_is_first=False):
        cloud = GeoPointCloud()
        cloud.crs = crs
        if data_axis_is_first:
            xyz = xyz.transpose()
        cloud.data['x'] = xyz[:, 0]
        cloud.data[['y', 'z']] = xyz[:, 1:]
        if rgb is not None:
            if data_axis_is_first:
                rgb = rgb.transpose()
            cloud.data[['r', 'g', 'b']] = rgb
        cloud.rgb_max = rgb_max
        if normals_xyz is not None:
            if data_axis_is_first:
                normals_xyz = normals_xyz.transpose()
            cloud.data[['n_x', 'n_y', 'n_z']] = normals_xyz
        if curvature is not None:
            if data_axis_is_first:
                curvature = curvature.transpose()
            cloud.data['curvature'] = curvature
        return cloud

    @classmethod
    def create_empty(cls):
        return GeoPointCloud()

    @classmethod
    def from_pandas(cls, data_frame: pd.DataFrame, crs: GeoCrs=NoneCRS(), rgb_max=1.0):
        if 'x' not in data_frame or 'y' not in data_frame or 'z' not in data_frame:
            raise KeyError("DataFrame needs at least columns x,y,z.")
        cloud = GeoPointCloud()
        cloud.data = data_frame
        cloud.crs = crs
        cloud.rgb_max = rgb_max
        return cloud

    @classmethod
    def from_structured_array(cls, structured_array: np.ndarray, crs: GeoCrs=NoneCRS(), rgb_max=1.0):
        x = structured_array['X']
        y = structured_array['Y']
        z = structured_array['Z']
        xyz = np.stack((x, y, z))
        fields = structured_array.dtype.fields
        rgb = None
        if 'Red' in fields and 'Green' in fields and 'Blue' in fields:
            rgb = np.stack((structured_array['Red'], structured_array['Green'], structured_array['Blue']))
        normals_xyz = None
        if 'NormalX' in fields and 'NormalY' in fields and 'NormalZ' in fields:
            normals_xyz = np.stack((structured_array['NormalX'], structured_array['NormalY'],
                                    structured_array['NormalZ']))
        curvature = None
        if 'Curvature' in fields:
            curvature = structured_array['Curvature']
        return GeoPointCloud.from_numpy_arrays(xyz, crs=crs, rgb=rgb, rgb_max=rgb_max, normals_xyz=normals_xyz,
                                               curvature=curvature, data_axis_is_first=True)

    @classmethod
    def from_o3d(cls, o3d_point_cloud: o3d.geometry.PointCloud, crs: GeoCrs = NoneCRS(), rgb_max=255):
        if (not o3d_point_cloud.has_points()):
            raise ValueError("Open3d point cloud needs to have points")
        xyz = np.array(o3d_point_cloud.points)
        normals = None
        if (o3d_point_cloud.has_normals()):
            normals = np.array(o3d_point_cloud.normals)
            if (normals.shape[0] != xyz.shape[0]):
                normals = None
        rgb = None
        if (o3d_point_cloud.has_colors()):
            rgb = np.array(o3d_point_cloud.colors)
            if (rgb.shape[0] != xyz.shape[0]):
                rgb = None
        return GeoPointCloud.from_numpy_arrays(xyz, crs=crs, rgb=rgb, normals_xyz=normals, rgb_max=rgb_max)

    @classmethod
    def from_las(cls, file_path, crs:GeoCrs = NoneCRS()):
        pipeline_description = [str(file_path)]
        pipeline = pdal.Pipeline(json.dumps(pipeline_description))
        count = pipeline.execute()
        arrays = pipeline.arrays
        metadata = json.loads(pipeline.metadata)
        if isinstance(crs, NoneCRS):
            crs = GeoCrs(metadata['metadata']['readers.las']['spatialreference'])
        return GeoPointCloud.from_structured_array(arrays[0], crs=crs, rgb_max=np.iinfo(np.uint16).max)

    @classmethod
    def from_xyz_file(cls, file_path: Union[str, Path], crs:GeoCrs=NoneCRS(), read_normals=False):
        file_path = Path(file_path)
        names = ('x', 'y', 'z')
        if read_normals:
            names = ('x', 'y', 'z', 'n_x', 'n_y', 'n_z')
        points = pd.read_csv(file_path, sep=' ', header=0, names=names)
        xyz = np.stack([points['x'].to_numpy(), points['y'].to_numpy(), points['z'].to_numpy()], axis=0)
        normals = None
        if read_normals:
            normals = np.stack([points['n_x'].to_numpy(), points['n_y'].to_numpy(), points['n_z'].to_numpy()], axis=0)
        return GeoPointCloud.from_numpy_arrays(xyz, crs=crs, normals_xyz=normals, data_axis_is_first=True)

    @classmethod
    def from_geomesh(cls: Type[T], mesh: geomesh.GeoMesh, point_density: float = 1,
                     number_of_points: Optional[int] = None, crs:GeoCrs=NoneCRS()):
        mo3d = mesh.to_o3d()
        if number_of_points is None:
            area = mo3d.get_surface_area()
            number_of_points = int(area * point_density)
        return cls.from_o3d(mo3d.sample_points_uniformly(number_of_points=number_of_points), crs=crs)

    @classmethod
    def from_others(cls, others: List['GeoPointCloud'], keep_others=True) -> 'GeoPointCloud':
        if keep_others:
            result = others[0].copy()
        else:
            result = others[0]
        if len(others) == 1:
            return result
        result.data = pd.concat([other.data for other in others])
        return result

    def split_by_class(self, class_column: str) -> List['GeoPointCloud']:
        return [GeoPointCloud.from_pandas(x, rgb_max=self.rgb_max) for _, x in
                self.data.groupby(self.data[class_column])]

    @property
    def xyz(self) -> pd.DataFrame:
        return self.data[['x', 'y', 'z']]

    @xyz.setter
    def xyz(self, xyz: np.ndarray):
        assert xyz.shape == self.xyz.shape
        self.data[['x', 'y', 'z']] = xyz

    @property
    def x(self) -> pd.Series:
        return self.data['x']

    @x.setter
    def x(self, x: np.ndarray):
        assert x.shape == self.x.shape
        self.data['x'] = x

    @property
    def y(self) -> pd.Series:
        return self.data['z']

    @y.setter
    def y(self, y: np.ndarray):
        assert y.shape == self.y.shape
        self.data['y'] = y

    @property
    def z(self) -> pd.Series:
        return self.data['z']

    @z.setter
    def z(self, z: np.ndarray):
        assert z.shape == self.z.shape
        self.data['z'] = z

    @property
    def rgb(self) -> Optional[pd.DataFrame]:
        if 'r' in self.data and 'g' in self.data and 'b' in self.data:
            return self.data[['r', 'g', 'b']]
        else:
            return None

    @rgb.setter
    def rgb(self, rgb: np.ndarray):
        assert rgb.shape == self.rgb.shape
        self.data[['r', 'g', 'b']] = rgb

    def has_normals(self) -> bool:
        return 'n_x' in self.data and 'n_y' in self.data and 'n_z' in self.data

    @property
    def normals_xyz(self) -> Optional[pd.DataFrame]:
        if self.has_normals():
            return self.data[['n_x', 'n_y', 'n_z']]
        else:
            return None

    @normals_xyz.setter
    def normals_xyz(self, normals: np.ndarray) -> Optional[pd.DataFrame]:
        assert normals.shape == self.normals_xyz.shape
        self.data[['n_x', 'n_y', 'n_z']] = normals

    @property
    def curvature(self) -> Optional[pd.Series]:
        if 'curvature' in self.data:
            return self.data['curvature']
        else:
            return None

    @property
    def confidence(self) -> Optional[pd.Series]:
        if 'confidence' in self.data:
            return self.data['confidence']
        else:
            return None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def _reset_cached(self):
        self._bounds = None
        self._principle_components = None
        self._cov = None
        self._aab_dims = None
        self._center = None
        if 'n_x' in self.data:
            del self.data['n_x']
        if 'n_y' in self.data:
            del self.data['n_y']
        if 'n_z' in self.data:
            del self.data['n_z']

    def _dtypes(self):
        dtypes = {}
        for key, value in self.data.items():
            dtypes[key] = value.dtype

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace=True) -> 'GeoPointCloud':
        if crs_transformer is None:
            crs_transformer = GeoCrsTransformer(self.crs, new_crs)
        if inplace:
            self.xyz = np.stack(crs_transformer.transform(self.x, self.y, self.z), axis=1)
            self.crs = new_crs
            self._reset_cached()
            return self
        else:
            res = self.copy()
            res.to_crs(new_crs = new_crs, crs_transformer=crs_transformer)
            return res

    def to_o3d(self, normalize_rgb=True):
        xyz_o3d = o3d.utility.Vector3dVector(self.xyz.to_numpy())
        pc3d = o3d.geometry.PointCloud()
        pc3d.points = xyz_o3d
        if self.normals_xyz is not None:
            normals_xyz_o3d = o3d.utility.Vector3dVector(self.normals_xyz.to_numpy())
            pc3d.normals = normals_xyz_o3d
        if self.rgb is not None:
            rgb = self.rgb.to_numpy().astype(float)
            if normalize_rgb:
                rgb /= self.rgb_max
            rgb_o3d = o3d.utility.Vector3dVector(rgb)
            pc3d.colors = rgb_o3d
        return pc3d

    def to_structured_array(self) -> np.ndarray:
        sub_arrays = []
        dtypes = []
        dt = self.xyz.dtype
        sub_arrays.append(('X', self.xyz[0]))
        sub_arrays.append(('Y', self.xyz[1]))
        sub_arrays.append(('Z', self.xyz[2]))
        dtypes.extend([dt, dt, dt])
        if self.rgb is not None:
            dt = self.rgb.dtype
            sub_arrays.append(('Red', self.rgb[0]))
            sub_arrays.append(('Green', self.rgb[1]))
            sub_arrays.append(('Blue', self.rgb[2]))
        dtypes.extend([dt, dt, dt])
        if self.normals_xyz is not None:
            dt = self.normals_xyz.dtype
            sub_arrays.append(('NormalX', self.normals_xyz[0]))
            sub_arrays.append(('NormalY', self.normals_xyz[1]))
            sub_arrays.append(('NormalZ', self.normals_xyz[2]))
        dtypes.extend([dt, dt, dt])
        if self.curvature is not None:
            sub_arrays.append(('Curvature', self.curvature))
            dtypes.append(self.curvature.dtype)
        result = np.empty(self.size, np.dtype([(a[0], dtypes[i]) for i, a in enumerate(sub_arrays)]))
        for a in sub_arrays:
            result[a[0]] = a[1]
        return result

    def to_las(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        point_format_id = 0
        if self.rgb is not None:
            point_format_id = 3
        las_data = pylas.create(point_format_id=point_format_id, file_version="1.3")
        if self.size:
            las_data.x = self.xyz['x'].to_numpy()
            las_data.y = self.xyz['y'].to_numpy()
            las_data.z = self.xyz['z'].to_numpy()
        if self.rgb is not None:
            rgb = self.rgb
            if self.rgb_max != np.iinfo(np.uint16).max:
                rgb = rgb / self.rgb_max * np.iinfo(np.uint16).max
            las_data.red = rgb['r'].to_numpy().astype(np.uint16)
            las_data.green = rgb['g'].to_numpy().astype(np.uint16)
            las_data.blue = rgb['b'].to_numpy().astype(np.uint16)
        las_data.update_header()
        file_path.parents[0].mkdir(parents=True, exist_ok=True)
        las_data.write(str(file_path))

    def apply_pdal_pipeline(self, pipeline_json: str, return_raw_output=False):
        pipeline = pdal.Pipeline(pipeline_json, [self.to_structured_array()])
        pipeline.validate()
        count = pipeline.execute()
        if return_raw_output:
            return pipeline.arrays, pipeline.metadata, pipeline.log
        else:
            return [GeoPointCloud.from_structured_array(a, crs=self.crs) for a in pipeline.arrays]

    def filter_by_voxel_grid(self, voxel_grid: 'GeoVoxelGrid') -> 'GeoPointCloud':
        s_indices = ((self.xyz.to_numpy() - voxel_grid.origin) // voxel_grid.voxel_size).astype(int)
        mask = (s_indices[:, 0] >= 0) & (s_indices[:, 0] < voxel_grid.shape[0])
        mask &= (s_indices[:, 1] >= 0) & (s_indices[:, 1] < voxel_grid.shape[1])
        mask &= (s_indices[:, 2] >= 0) & (s_indices[:, 2] < voxel_grid.shape[2])
        valid = s_indices[mask]
        s_indices = np.zeros(self.shape[0], dtype=bool)
        s_indices[mask] = voxel_grid.occupied()[valid[:, 0], valid[:, 1], valid[:, 2]]
        return self[s_indices]

    def to_image(self, pixel_size: float, up_axis: int = 1, value_field: Optional[str] = None, empty_value=0,
                 ufunc: Optional[np.ufunc] = None) -> Tuple[np.ndarray, np.ndarray]:
        xyz = self.xyz.to_numpy()
        if value_field is None:
            values = 1
        elif value_field == 'height':
            values = self.xyz.to_numpy()[:, up_axis]
            if ufunc is None:
                ufunc = np.maximum()
        else:
            values = self.data[value_field]
        return points3d_to_image(pixel_size=pixel_size, xyz=xyz, values=values, up_axis=up_axis,
                                 empty_value=empty_value, ufunc=ufunc)

    def to_georaster(self, pixel_size: float, value_field: Optional[str] = None, empty_value=0,
                     ufunc: Optional[np.ufunc] = None) -> GeoRaster:
        if self.crs.is_geocentric:
            raise ValueError("Point clouds in geocentric coordinates cannot projected down along the z-axis to flatten them to earth surface")
        img, origin = self.to_image(pixel_size=pixel_size, up_axis=2, value_field=value_field, empty_value=empty_value,
                                     ufunc=ufunc)
        georect = GeoRect(origin, (origin + img.shape[1], origin + img.shape[0]), crs=self.crs)
        return GeoRaster(georect, img)

    def filter_by_image(self, image: np.ndarray, image_origin: Tuple[float, float] = (0, 0),
                        pixel_size: float = 1, up_axis: int = 2) -> 'GeoPointCloud':
        plane_axis = [0, 1, 2]
        del plane_axis[up_axis]
        s_indices = self.xyz.to_numpy()[:, (plane_axis[0], plane_axis[1])]
        s_indices = ((s_indices - np.array(image_origin)) / pixel_size).astype(int)
        mask = (s_indices[:, 0] >= 0) & (s_indices[:, 0] < image.shape[0])
        mask &= (s_indices[:, 1] >= 0) & (s_indices[:, 1] < image.shape[1])
        valid = s_indices[mask]
        s_indices = np.zeros(self.shape[0], dtype=bool)
        s_indices[mask] = image[valid[:, 0], valid[:, 1]]
        return self[s_indices]

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            return self.copy(indices=item)

    def copy(self, indices: Union[np.ndarray, None] = None):
        if indices is not None:
            if indices.dtype == np.bool:
                result = GeoPointCloud.from_pandas(self.data[indices], crs=self.crs)
            elif indices.dtype == int or indices.dtype == np.long or indices.dtype == np.int:
                mask = np.zeros(self.shape[0], dtype=bool)
                mask[indices] = 1
                result = GeoPointCloud.from_pandas(self.data[mask], crs=self.crs)
        else:
            result = GeoPointCloud.from_pandas(self.data.copy(), crs=self.crs)
        result._bounds = self._bounds
        result._principle_components = self._principle_components
        result._cov = self._cov
        result._aab_dims = self._aab_dims
        result._center = self._center
        result.rgb_max = self.rgb_max
        return result

    def column_descriptor(self):
        return dict([(name, column.dtype) for name, column in self.data.items()])

    def extend(self, other: 'GeoPointCloud', keep_other=True):
        if self.column_descriptor() != other.column_descriptor():
            raise ValueError("The point clouds have different attribute names or types")
        self.data = pd.concat([self.data, other.data])
        if not keep_other:
            del other

    @property
    def size(self):
        return self.xyz.shape[1]

    def __len__(self):
        return self.size

    @property
    def cov(self):
        if self._cov is None:
            self._cov = np.cov((self.xyz - self.xyz.mean(axis=1, keepdims=True)) /
                               self.xyz.std(axis=1, keepdims=True))
        return self._cov

    @property
    def principle_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (eigenvalues, eigenvectors)
        """
        if self._principle_components is None:
            pca = PCA(n_components=3)
            pca.fit(self.xyz)
            self._principle_components = (pca.singular_values_, pca.components_)
        return self._principle_components

    def thin_by_grid(self, grid_size: float, inplace=True):
        if inplace:
            cloud = self
        else:
            cloud = self.copy()

        cloud.data[['x', 'y', 'z']] //= grid_size
        data = cloud.data.groupby(['x', 'y', 'z'], as_index=False).mean()
        data[['x', 'y', 'z']] *= grid_size
        cloud.data[['x', 'y', 'z']] += np.full(cloud.data[['x', 'y', 'z']].shape, grid_size / 2)

        cloud.data = data
        return cloud
        # self._reset_cached() TODO: think about whether we should be super accurate here and reset the cache

    def apply_jitter(self, max_jitter=0.001):
        s = np.random.normal(0, max_jitter, self.shape[0] * 3).reshape((self.shape[0], 3))
        xyz = self.xyz.to_numpy() + s
        self.xyz = xyz

    @property
    def center(self) -> np.ndarray:
        if self._center is None:
            self._center = self.xyz.mean(axis=1)
        return self._center

    def make_axis_aligned(self):
        transform = self.principle_components[1]
        self.xyz = transform @ (self.xyz - self.xyz.mean(axis=1, keepdims=True))

    def estimate_normals(self, knn_k=100, fast_normal_computation=True):
        o3d_pc = self.to_o3d()
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn_k), fast_normal_computation)
        normals = np.array(o3d_pc.normals)
        self.data[['n_x', 'n_y', 'n_z']] = normals

    def subsample_random(self, num_points) -> 'GeoPointCloud':
        if num_points > self.shape[0]:
            return self.copy()
        indices = np.random.choice(self.shape[0], num_points, replace=False)
        return GeoPointCloud.from_pandas(self.data.iloc[indices], rgb_max=self.rgb_max)

    @property
    def aab_dims(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (aab_dim_lengths, aab_dim_vectors) sorted big to small (largest component first)
        """
        if self._aab_dims is None:
            transform = self.principle_components[1]
            xyz = self.xyz.to_numpy()
            xyz = (xyz - xyz.mean(axis=0, keepdims=True)) @ transform
            self._aab_dims = (xyz.max(axis=0) - xyz.min(axis=0), self.principle_components[1])
        return self._aab_dims

    @property
    def bounds(self):
        if self._bounds is None:
            mins = self.xyz.min()
            maxs = self.xyz.max()
            self._bounds = [*mins[:3], *maxs[:3]]
        return self._bounds

    # TODO: Deprecated, remove!
    def visualize_pptk(self):
        import warnings
        warnings.warn(".visualize_pptk got renamed use .plot_pptk() instead", DeprecationWarning)
        self.plot_pptk()

    def _categorical_colors(self, attribute_name: str, color_map=cm.tab20):
        colors = np.array(color_map.colors)
        return colors[self.data[attribute_name].to_numpy() % colors.shape[0]]

    def plot_pptk(self, categorical_colors_attribute: Optional[str] = None, color_map=cm.tab20):
        import pptk  # import it here so that it does not fail in case it is not installed and not used
        if self.rgb is None and categorical_colors_attribute is None:
            v = pptk.viewer(self.xyz.to_numpy())
        else:
            print("rendering with RGB")
            if categorical_colors_attribute is not None:
                rgb = self._categorical_colors(categorical_colors_attribute, color_map=color_map)
            else:
                rgb = self.rgb.to_numpy()
                if self.rgb_max != 255:
                    rgb = rgb.astype(float) / self.rgb_max
            v = pptk.viewer(self.xyz.to_numpy(), rgb)
        return v

    def _plot_o3d(self, function, categorical_colors_attribute: Optional[str] = None, color_map=cm.tab20, *args,
                  **kwargs):
        xyz_o3d = o3d.utility.Vector3dVector(self.xyz.to_numpy())
        pc3d = o3d.geometry.PointCloud()
        pc3d.points = xyz_o3d
        if categorical_colors_attribute is not None:
            rgb = self._categorical_colors(categorical_colors_attribute, color_map=color_map)
            pc3d.colors = o3d.utility.Vector3dVector(rgb)
        elif self.rgb is not None:
            rgb = self.rgb.to_numpy().astype(float)
            rgb /= self.rgb_max
            pc3d.colors = o3d.utility.Vector3dVector(rgb)
        if self.has_normals():
            pc3d.normals = o3d.utility.Vector3dVector(self.normals_xyz.to_numpy())
        function([pc3d], *args, **kwargs)

    def plot_o3d(self, categorical_colors_attribute: Optional[str] = None, color_map=cm.tab20, *args, **kwargs):
        self._plot_o3d(o3d.visualization.draw_geometries, categorical_colors_attribute, color_map, *args, **kwargs)

    def plot_o3dj(self, categorical_colors_attribute: Optional[str] = None, color_map=cm.tab20, *args, **kwargs):
        from open3d.web_visualizer import draw as draw_jupyter
        self._plot_o3d(draw_jupyter, categorical_colors_attribute, color_map, *args, **kwargs)