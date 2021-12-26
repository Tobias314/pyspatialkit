


from abc import ABC, abstractmethod
from enum import Enum

from ...spacedescriptors.tiles3dboundingvolume import Tiles3dBoundingVolume

#TODO: add all types
class Tiles3dContentType(Enum):
    POINT_CLOUD = 1

TILES3D_CONTENT_TYPE_TO_FILE_ENDING = {'POINT_CLOUD': '.pnts'} #TODO: add file endings for remaining types

class Tiles3dContentObject(ABC):
    @property
    @abstractmethod
    def content_type_tile3d(self)->Tiles3dContentType:
        raise NotImplementedError

    @abstractmethod
    def to_bytes_tiles3d(self)->bytes:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounding_volume_tiles3d(self)->Tiles3dBoundingVolume:
        raise NotImplementedError