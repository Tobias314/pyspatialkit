from typing import Type, TypeVar, Tuple, Sequence

#import kaolin as kal
#import torch
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
import trimesh
from trimesh.creation import triangulate_polygon

T = TypeVar('T', bound='TrivialClass')

class Geo3dMeshData:
    def __init__(self):
        self.vertices = None
        self.faces = None


class GeoMesh:

    @classmethod
    def from_kalmesh(cls: Type[T], kalmesh) -> T:
        mesh = GeoMesh(trimesh.Trimesh(kalmesh.vertices.numpy(), kalmesh.faces.numpy()))
        mesh.cached_kalmesh = kalmesh
        return mesh

    @classmethod
    def from_o3d_mesh(cls: Type[T], o3dmesh) -> T:
        tmesh = trimesh.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles)
        mesh = cls(tmesh)
        return mesh

    @classmethod
    def from_trimesh(cls: Type[T], trimesh: trimesh.Trimesh) -> T:
        return cls(trimesh)

    @classmethod
    def from_shapely(cls: Type[T], polygon: Polygon, third_dim: int = 2, third_dim_value: float = 0):
        vert, faces = triangulate_polygon(polygon, engine='earcut') #we only use earcut because it has the better license
        vert = np.insert(vert, third_dim, third_dim_value, axis=1)
        return cls.from_vertices_faces(vert, faces)

    @classmethod
    def from_vertices_faces(cls: Type[T], vertices: np.ndarray, faces: np.ndarray):
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return cls(tmesh)

    @classmethod
    def from_others(cls: Type[T], others: Sequence['GeoMesh']) -> T:
        vertices = []
        faces = []
        count = 0
        for other in others:
            vertices.append(other.vertices)
            faces.append(other.faces + count)
            count += other.vertices.shape[0]
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        return cls.from_vertices_faces(vertices, faces)

    def __init__(self, trimesh):
        self.tmesh = trimesh

        self.cached_kalmesh = None

    def __getstate__(self):
        data = Geo3dMeshData()
        data.vertices = self.vertices()
        data.faces = self.faces()
        return data

    def __setstate__(self, state : Geo3dMeshData):
        tmesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
        self.__init__(tmesh)

    def center_and_scale(self, center: Tuple[float, float, float] = (0,0,0), max_bbx_side: float = 1):
        self.tmesh = self.tmesh.apply_transform(
            trimesh.transformations.scale_matrix(max_bbx_side / self.tmesh.extents.max()))
        center = np.array(center)
        shift_mat = trimesh.transformations.scale_and_translate(
            scale=1, translate=center - 1/2 * self.tmesh.bounds.sum(axis=0))
        self.tmesh = self.tmesh.apply_transform(shift_mat)
        self._invalidate_caches()

    def to_trimesh(self):
        return self.tmesh

    def to_o3d(self):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        triangles = o3d.utility.Vector3iVector(self.faces)
        return o3d.geometry.TriangleMesh(vertices, triangles)

    # def as_kmesh(self):
    #    if not self.cached_kalmesh:
    #        self.cached_kalmesh = kal.rep.TriangleMesh(vertices=torch.from_numpy(self.vertices()),
    #                                                   faces=torch.from_numpy(self.faces()))
    #    return self.cached_kalmesh

    def plot_o3d(self, *args, **kwargs):
        mesh = self.to_o3d()
        mesh.compute_triangle_normals()
        o3d.visualization.draw_geometries([mesh], *args, **kwargs)


    def plot_o3dj(self, *args, **kwargs):
        from open3d.web_visualizer import draw as draw_jupyter
        
        mesh = self.to_o3d()
        mesh.compute_triangle_normals()
        draw_jupyter([mesh], *args, **kwargs)

    @property
    def vertices(self):
        return self.tmesh.vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        assert vertices.shape == self.vertices.shape
        self.tmesh.vertices = vertices

    @property
    def faces(self):
        return self.tmesh.faces

    @property
    def vertex_normals(self):
        return self.tmesh.vertex_normals

    def _invalidate_caches(self):
        pass

    @classmethod
    def get_box_mesh(cls: Type[T]) -> T:
        o3dmesh = o3d.geometry.TriangleMesh.create_box()
        return cls.from_o3d_mesh(o3dmesh)