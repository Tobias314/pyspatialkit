from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from ...spacedescriptors.tiles3dboundingvolume import Tiles3dBoundingVolume
from ...crs.geocrs import GeoCrs
from ...crs.geocrstransformer import GeoCrsTransformer

#TODO: add all types
class Tiles3dContentType(Enum):
    POINT_CLOUD = 1
    MESH = 2

TILES3D_CONTENT_TYPE_TO_FILE_ENDING = {
    Tiles3dContentType.POINT_CLOUD.value : '.pnts',
    Tiles3dContentType.MESH.value: '.b3dm'
} #TODO: add file endings for remaining types

class Tiles3dContentObject(ABC):

    @classmethod
    @abstractmethod
    def get_content_type_tile3d(self)->Tiles3dContentType:
        raise NotImplementedError

    @abstractmethod
    def to_bytes_tiles3d(self)->bytes:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounding_volume_tiles3d(self)->Tiles3dBoundingVolume:
        raise NotImplementedError

    @abstractmethod
    def to_crs(self, new_crs: GeoCrs, transformer: Optional[GeoCrsTransformer] = None):
        raise NotImplementedError()