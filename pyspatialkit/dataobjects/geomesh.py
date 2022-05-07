from typing import Type, TypeVar, Tuple, Sequence, Optional
from pathlib import Path
from io import BytesIO
import json

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

from ..crs.geocrs import GeoCrs, NoneCRS
from ..crs.geocrstransformer import GeoCrsTransformer
from ..globals import get_default_crs, TILE3D_CRS
from ..storage.bboxstorage.bboxstorage import BBoxStorageObjectInterface
from .geodataobjectinterface import GeoDataObjectInterface
from .tiles3d.tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType
from ..spacedescriptors.geobox3d import GeoBox3d

T = TypeVar('T', bound='TrivialClass')

B3DM_VERSION = 1
B3DM_VERSION_HEADER_FIELD = B3DM_VERSION.to_bytes(4, 'little')
Z_UP_Y_UP_TRANSFORM = np.array([[ 1.,0.,0.,0.],[ 0.,0.,1.,0.],[0.,-1.,0.,0.],[ 0.,0.,0.,1.]])

class Geo3dMeshData:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.crs_dict = None


class GeoMesh(BBoxStorageObjectInterface, GeoDataObjectInterface, Tiles3dContentObject):

    # @classmethod
    # def from_kalmesh(cls: Type[T], kalmesh, crs:Optional[GeoCrs]=None) -> T:
    #     mesh = GeoMesh(trimesh.Trimesh(kalmesh.vertices.numpy(), kalmesh.faces.numpy()),crs=crs)
    #     mesh.cached_kalmesh = kalmesh
    #     return mesh

    @classmethod
    def from_o3d_mesh(cls: Type[T], o3dmesh, crs:Optional[GeoCrs]=None) -> T:
        visuals = None
        if o3dmesh.vertex_colors is not None:
            visuals = trimesh.visual.ColorVisuals(vertex_colors=o3dmesh.vertex_colors)
        tmesh = trimesh.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles, visual=visuals)
        mesh = cls(tmesh,crs=crs)
        return mesh

    @classmethod
    def from_trimesh(cls: Type[T], trimesh: trimesh.Trimesh, crs:Optional[GeoCrs]=None) -> T:
        return cls(trimesh,crs=crs)

    @classmethod
    def from_shapely(cls: Type[T], polygon: Polygon, crs:Optional[GeoCrs]=NoneCRS(), third_dim: int = 2, third_dim_value: float = 0):
        vert, faces = triangulate_polygon(polygon, engine='earcut') #we only use earcut because it has the better license
        vert = np.insert(vert, third_dim, third_dim_value, axis=1)
        return cls.from_vertices_faces(vert, faces,crs=crs)

    @classmethod
    def from_vertices_faces(cls: Type[T], vertices: np.ndarray, faces: np.ndarray, crs: Optional[GeoCrs]=None,
                            vertex_colors: Optional[np.ndarray] = None):
        visuals = None
        if vertex_colors is not None:
            visuals = trimesh.visual.color.ColorVisuals(vertex_colors=vertex_colors[:,:3].astype(np.int64))
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, process=False)
        return cls(tmesh,crs=crs)

    @classmethod
    def from_others(cls: Type[T], others: Sequence['GeoMesh']) -> T:
        tmeshes = []
        crs = others[0].crs
        for other in others:
            if other.crs != crs:
                raise ValueError("All meshes need to have the same crs to so that they can be combined into one mesh!")
            tmeshes.append(other.tmesh)
        res = trimesh.util.concatenate(tmeshes)
        return cls(trimesh=trimesh, crs=crs)
        #TODO: delete if no longer needed:
        # vertices = []
        # faces = []
        # vertex_colors = []
        # count = 0
        # for other in others:
        #     if other.crs != others[0].crs:
        #         raise ValueError("All meshes need to have the same crs to so that they can be combined into one mesh!")
        #     vertices.append(other.vertices)
        #     faces.append(other.faces + count)
        #     vertex_colors.append(other.tmesh.visual.vertex_colors)
        #     count += other.vertices.shape[0]
        # vertices = np.concatenate(vertices, axis=0)
        # faces = np.concatenate(faces, axis=0)
        # vertex_colors = np.concatenate(vertex_colors, axis=0)
        # return cls.from_vertices_faces(vertices, faces, crs=others[0].crs, vertex_colors=vertex_colors)
    
    @classmethod
    def from_bytes(cls, binary_rep: bytes)-> Optional['GeoMesh']:
        crs_str_length = int.from_bytes(binary_rep[:4], 'big')
        crs = GeoCrs(binary_rep[4:4+crs_str_length].decode('utf-8'))
        binary_io = BytesIO(binary_rep[4+crs_str_length:])
        tmesh = trimesh.exchange.load.load(binary_io, file_type='glb')
        if len(tmesh.geometry) == 1:
            tmesh = next(iter(tmesh.geometry.values()))
        else:
            tmesh = trimesh.util.concatenate(list(tmesh.geometry.values()))
        return cls(trimesh=tmesh, crs=crs)

    def __init__(self, trimesh: Trimesh, crs:Optional[GeoCrs]=None):
        self.tmesh = trimesh
        self.cached_kalmesh = None
        self.crs = crs
        if self.crs is None:
            self.crs = get_default_crs()

    def to_gmsh_file(self, path: Path):
        crs_str = self.crs.to_str()
        np.savez(path, vertices=self.tmesh.vertices, faces=self.tmesh.faces, crs=np.array([crs_str], dtype=str))

    def to_bytes(self)-> bytes:
        crs_str = self.crs.to_str().encode('utf-8')
        crs_str_length = (len(crs_str)).to_bytes(4, 'big')
        binary_mesh = self.tmesh.export(file_type='glb')
        return crs_str_length + crs_str + binary_mesh

    @classmethod
    def from_gmsh_file(cls, path: Path) -> 'GeoMesh':
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
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.faces))
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.tmesh.visual.vertex_colors[:,:3].astype(float) / 255)
        return o3d_mesh
        #return self.tmesh.as_open3d

    def to_crs(self, new_crs: Optional[GeoCrs] = None, crs_transformer: Optional[GeoCrsTransformer] = None, inplace:bool=False):
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
        import open3d as o3d
        mesh = self.to_o3d()
        mesh.compute_triangle_normals()
        o3d.visualization.draw_geometries([mesh], *args, **kwargs)

    def plot_o3dj(self, *args, **kwargs):
        from open3d.web_visualizer import draw as draw_jupyter
        
        mesh = self.to_o3d()
        mesh.compute_triangle_normals()
        draw_jupyter([mesh], *args, **kwargs)
    
    def plot_trimesh(self):
        return self.to_trimesh().show()

    def plot(self):
        return self.plot_trimesh()

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

    @classmethod
    def get_content_type_tile3d(self) -> Tiles3dContentType:
        return Tiles3dContentType.MESH

    def to_bytes_tiles3d(self, crs_transformer: Optional[GeoCrsTransformer] = None) -> bytes:
        print(self.vertices)
        if TILE3D_CRS == self.crs:
            crs_transformer = None
        else:
            if crs_transformer is None:
                crs_transformer = GeoCrsTransformer(self.crs, TILE3D_CRS)
            else:
                if crs_transformer.to_crs != TILE3D_CRS:
                    raise ValueError('Transformer must transform to:' + str(TILE3D_CRS))
        original_vertices = self.tmesh.vertices.copy()
        if crs_transformer is not None:
            x,y,z = np.split(self.tmesh.vertices, 3, axis=1)
            self.tmesh.vertices = np.stack(crs_transformer.transform(x.flatten(), y.flatten(), z.flatten()), axis=1)
            center = self.tmesh.vertices.mean(axis=0)
            self.tmesh.vertices -= center
        magic = b'b3dm'
        version = B3DM_VERSION_HEADER_FIELD
        batch_table_json_byte_length = (0).to_bytes(4, 'little')
        batch_table_binary_byte_length = (0).to_bytes(4, 'little')
        feature_table_json = {
            'BATCH_LENGTH': 0,
            'RTC_CENTER': list(center)
        }
        feature_table_json = json.dumps(feature_table_json)
        feature_table_json_padding = (8 - (28 + len(feature_table_json)) % 8) % 8
        feature_table_json = bytes(feature_table_json + ' ' * feature_table_json_padding, 'utf-8') 
        feature_table = b''
        tscene = trimesh.scene.Scene()
        tscene.add_geometry(self.tmesh, transform=Z_UP_Y_UP_TRANSFORM)
        binary_gltf = tscene.export(file_type='glb')
        #binary_gltf = self.tmesh.export(file_type='glb')
        byte_length = (28 + len(feature_table_json) + len(feature_table) + len(binary_gltf)).to_bytes(4, 'little')
        feature_table_json_byte_length = np.array([len(feature_table_json)], dtype='<u4').tobytes()
        feature_table_binary_byte_length = np.array([len(feature_table)], dtype='<u4').tobytes()
        res = magic + version + byte_length + feature_table_json_byte_length + feature_table_binary_byte_length + \
               batch_table_json_byte_length + batch_table_binary_byte_length + feature_table_json + feature_table + binary_gltf
        self.tmesh.vertices = original_vertices
        return res

    @property
    def bounding_volume_tiles3d(self) -> GeoBox3d:
        return GeoBox3d.from_bounds(self.bounds)