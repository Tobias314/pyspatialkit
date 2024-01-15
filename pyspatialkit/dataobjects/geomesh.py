from typing import Type, TypeVar, Tuple, Sequence, Optional
from pathlib import Path

#import kaolin as kal
#import torch
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
import trimesh
from trimesh import Trimesh
from trimesh.creation import triangulate_polygon
import fcl
from trimesh.collision import mesh_to_BVH

from ..crs.geocrs import GeoCrs
from ..crs.geocrstransformer import GeoCrsTransformer
from ..globals import get_default_crs
from ..storage.bboxstorage.bboxstorage import BBoxStorageObjectInterface

T = TypeVar('T', bound='TrivialClass')

class Geo3dMeshData:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.crs_dict = None


class GeoMesh(BBoxStorageObjectInterface):

    @classmethod
    def from_kalmesh(cls: Type[T], kalmesh, crs:Optional[GeoCrs]=None) -> T:
        mesh = GeoMesh(trimesh.Trimesh(kalmesh.vertices.numpy(), kalmesh.faces.numpy()),crs=crs)
        mesh.cached_kalmesh = kalmesh
        return mesh

    @classmethod
    def from_o3d_mesh(cls: Type[T], o3dmesh, crs:Optional[GeoCrs]=None) -> T:
        tmesh = trimesh.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles)
        mesh = cls(tmesh,crs=crs)
        return mesh

    @classmethod
    def from_trimesh(cls: Type[T], trimesh: trimesh.Trimesh, crs:Optional[GeoCrs]=None) -> T:
        return cls(trimesh,crs=crs)

    @classmethod
    def from_shapely(cls: Type[T], polygon: Polygon, crs:Optional[GeoCrs]=None, third_dim: int = 2, third_dim_value: float = 0):
        vert, faces = triangulate_polygon(polygon, engine='earcut') #we only use earcut because it has the better license
        vert = np.insert(vert, third_dim, third_dim_value, axis=1)
        return cls.from_vertices_faces(vert, faces,crs=crs)

    @classmethod
    def from_vertices_faces(cls: Type[T], vertices: np.ndarray, faces: np.ndarray, crs:Optional[GeoCrs]=None):
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return cls(tmesh,crs=crs)

    @classmethod
    def from_others(cls: Type[T], others: Sequence['GeoMesh']) -> T:
        vertices = []
        faces = []
        count = 0
        for other in others:
            if other.crs != other[0].crs:
                raise ValueError("All meshes need to have the same crs to so that they can be combined into one mesh!")
            vertices.append(other.vertices)
            faces.append(other.faces + count)
            count += other.vertices.shape[0]
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        return cls.from_vertices_faces(vertices, faces, crs=others[0].crs)

    def __init__(self, trimesh: Trimesh, crs:Optional[GeoCrs]=None):
        self.tmesh = trimesh
        self.cached_kalmesh = None
        self.crs = crs
        if self.crs is None:
            self.crs = get_default_crs()

    def to_file(self, path: Path):
        crs_str = self.crs.to_str()
        np.savez(path, vertices=self.tmesh.vertices, faces=self.tmesh.faces, crs=np.array([crs_str], dtype=str))

    @classmethod
    def from_file(cls, path: Path) -> 'GeoMesh':
        data = np.load(str(path) + '.npz')
        crs = GeoCrs.from_str(data['crs'])
        tmesh = Trimesh(vertices=data['vertices'], faces=data['faces'])
        return cls(trimesh=tmesh, crs=crs)

    def __getstate__(self):
        data = Geo3dMeshData()
        data.vertices = self.vertices
        data.faces = self.faces
        data.crs_dict = self.crs.to_dict()
        return data

    def __setstate__(self, state : Geo3dMeshData):
        tmesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
        self.__init__(tmesh, crs=GeoCrs.from_dict(state.crs_dict))

    def center_and_scale(self, center: Tuple[float, float, float] = (0,0,0), max_bbx_side: float = 1):
        self.tmesh = self.tmesh.apply_transform(
            trimesh.transformations.scale_matrix(max_bbx_side / self.tmesh.extents.max()))
        center = np.array(center)
        shift_mat = trimesh.transformations.scale_and_translate(
            scale=1, translate=center - 1/2 * self.tmesh.bounds.sum(axis=0))
        self.tmesh = self.tmesh.apply_transform(shift_mat)
        self._invalidate_caches()

    def copy(self) -> 'GeoMesh':
        return GeoMesh(self.tmesh, )

    def to_trimesh(self):
        return self.tmesh

    def to_o3d(self) -> o3d.geometry.TriangleMesh:
        #vertices = o3d.utility.Vector3dVector(self.vertices)
        #triangles = o3d.utility.Vector3iVector(self.faces)
        #return o3d.geometry.TriangleMesh(vertices, triangles)
        return self.tmesh.as_open3d

    def to_crs(self, new_crs: GeoCrs, crs_transformer: Optional[GeoCrsTransformer] = None, inplace:bool=False):
        if new_crs is None:
            if crs_transformer is None:
                raise AttributeError("You need to either specify a new_crs or a crs_transformer!")
            new_crs = crs_transformer.to_crs
        if crs_transformer is None:
            crs_transformer = GeoCrsTransformer(self._crs, new_crs)
        if inplace:
            res = self
        else:
            res = self.copy()
        x,y,z = np.split(res.tmesh.vertices, 3, axis=1)
        res.tmesh.vertices = np.stack(crs_transformer.transform(x.flatten(), y.flatten(), z.flatten()), axis=1)
        return res

    def is_intersecting_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
        size = bbox_max - bbox_min
        bbox = fcl.Box(*size)
        tf = fcl.Transform(bbox_min)
        bbox = fcl.CollisionObject(bbox, tf)
        mesh = fcl.CollisionObject(mesh_to_BVH(self.tmesh), fcl.Transform())
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        ret = fcl.collide(mesh, bbox, request, result)
        return ret > 0

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

    @faces.setter
    def faces(self, faces: np.ndarray):
        self.tmesh.faces = faces

    @property
    def vertex_normals(self):
        return self.tmesh.vertex_normals

    @property
    def bounds(self):
        self.get_bounds()

    def get_bounds(self):
        return np.array(self.tmesh.bounds).flatten()

    def _invalidate_caches(self):
        pass

    @classmethod
    def get_box_mesh(cls: Type[T]) -> T:
        o3dmesh = o3d.geometry.TriangleMesh.create_box()
        return cls.from_o3d_mesh(o3dmesh)