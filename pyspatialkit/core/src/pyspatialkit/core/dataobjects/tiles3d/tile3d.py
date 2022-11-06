from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod
from enum import Enum

from ...spacedescriptors.tiles3dboundingvolume import Tiles3dBoundingVolume
from ...dataobjects.tiles3d.tiles3dcontentobject import Tiles3dContentObject, Tiles3dContentType, TILES3D_CONTENT_TYPE_TO_FILE_ENDING
from ...spacedescriptors.geobox3d import GeoBox3d

# only import type for type checking, do not import during runtime
if TYPE_CHECKING:
    from . import tileset3d


class RefinementType(Enum):
    ADD = 1
    REPLACE = 2


class Tile3d(ABC):
    def __init__(self, tileset: 'tileset3d.Tileset3d'):
        self._reset_cached()
        self.tileset = tileset

    def _reset_cached(self):
        self._bounding_volume: Optional[Tiles3dBoundingVolume] = None
        self._geometric_error: Optional[float] = None
        self._refine: Optional[RefinementType] = None
        self._content: Optional[Tiles3dContentObject] = None
        self._is_content_initialized = False
        self._content_bytes = None
        self._content_type: Optional[Tiles3dContentType] = None
        self._content_bounding_volume: Optional[Tiles3dBoundingVolume] = None
        self._children: Optional[List[Tile3d]] = None
        self._identifier: Optional[object] = None
        self._cost: Optional[float] = None

###Hooks which sub classes have to override###

    @abstractmethod
    def get_bounding_volume(self) -> Tiles3dBoundingVolume:
        raise NotImplementedError

    @abstractmethod
    def get_geometric_error(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_refine(self) -> RefinementType:
        raise NotImplementedError

    @abstractmethod
    def get_content(self) -> Tiles3dContentObject:
        raise NotImplementedError

    @abstractmethod
    def get_content_type(self) -> Tiles3dContentType:
        raise NotImplementedError

    @abstractmethod
    def get_children(self) -> List['Tile3d']:
        raise NotImplementedError

    @abstractmethod
    def get_identifier(self) -> object:
        raise NotImplementedError

    def content_to_bytes(self, content: Tiles3dContentObject) -> bytes:
        return content.to_bytes_tiles3d()

    def get_cost(self) -> float:
        return 1

    def get_content_bounding_volume(self) -> Tiles3dBoundingVolume:
        return self.bounding_volume
        #return self.content.bounding_volume_tiles3d

###Properties###
    @property
    def bounding_volume(self) -> Tiles3dBoundingVolume:
        if self._bounding_volume is None:
            self._bounding_volume = self.get_bounding_volume()
        return self._bounding_volume

    @property
    def geometric_error(self) -> float:
        if self._geometric_error is None:
            self._geometric_error = self.get_geometric_error()
        return self._geometric_error

    @property
    def refine(self) -> RefinementType:
        if self._refine is None:
            self._refine = self.get_refine()
        return self._refine

    @property
    def content(self) -> Optional[Tiles3dContentObject]:
        if self._is_content_initialized is False:
            self._content = self.get_content()
            self._is_content_initialized = True
        if self._content is not None and self.content_type != self._content.get_content_type_tile3d():
            raise TypeError("The type of the content object does not match the content type set for this tile!")
        return self._content

    @property
    def content_bytes(self) -> bytes:
        if self._content_bytes is None:
            content = self.content
            if content is None:
                self._content_bytes = bytes()
            else:
                self._content_bytes = self.content_to_bytes(content)

    @property
    def content_type(self) -> Tiles3dContentType:
        if self._content_type is None:
            self._content_type = self.get_content_type()
        return self._content_type

    @property
    def content_bounding_volume(self) -> Tiles3dContentType:
        if self._content_bounding_volume is None:
            self._content_bounding_volume = self.get_content_bounding_volume()
        return self._content_bounding_volume

    @property
    def children(self) -> List['Tile3d']:
        if self._children is None:
            self._children = self.get_children()
        return self._children

    @property
    def identifier(self) -> object:
        if self._identifier is None:
            self._identifier = self.get_identifier()
        return self._identifier

    @property
    def cost(self) -> float:
        if self._cost is None:
            self._cost = self.get_cost()
        return self._cost

###Methods###

    #TODO fix cost/max_cost calculation
    def materialize(self, tile_uri_generator: Callable[['Tile3d'], str],
                    current_depth: int, accumulated_cost: float,
                    tile_content_uri_generator: Optional[Callable[['Tile3d'], str]] = None,
                    max_depth: Optional[int] = None, max_cost: Optional[float] = None,
                    callback: Optional[Callable[['Tile3d'], None]] = None) -> Tuple[Dict[str, Union[str, float, int, Dict, List, Tuple]], List['Tile3d']]:
        tile_uri = tile_uri_generator(self)
        if tile_content_uri_generator is None:
            tile_content_uri = tile_uri + '_content' + TILES3D_CONTENT_TYPE_TO_FILE_ENDING[self.content_type.value]
        else:
            tile_content_uri = tile_content_uri_generator(self)
        if max_depth is not None and current_depth > max_depth is not None:
            return (self._generate_link_proxy_tile_dict(tile_uri), [self])
        else:
            current_depth+=1
        if max_cost is not None and accumulated_cost + self.cost > max_cost:
            return (self._generate_link_proxy_tile_dict(tile_uri), [self])
        else:
            accumulated_cost += self.cost
        end_points = []
        children_dicts = []
        for c in self.children:
            child_dict, e_pts = c.materialize(tile_uri_generator=tile_uri_generator, tile_content_uri_generator=tile_content_uri_generator,
                                               current_depth=current_depth, accumulated_cost=accumulated_cost, max_depth=max_depth,
                                               max_cost=max_cost, callback=callback)
            children_dicts.append(child_dict)
            end_points += e_pts
        res_dict = self._generate_tile_dict(tile_content_uri=tile_content_uri, children=children_dicts)
        if callback is not None:
            callback(self)
        return res_dict, end_points

    def _generate_tile_dict_header(self, children: List[Dict]) -> Dict[str, Union[str, float, int, Dict, List, Tuple]]:
        res = {
            'boundingVolume': self.bounding_volume.to_tiles3d_bounding_volume_dict(),
            'geometricError':self.geometric_error,
            'refine':self.refine.name,
            'children': children,
        }
        return res

    def _generate_tile_dict(self, tile_content_uri: str, children: List[Dict]) -> Dict[str, Union[str, float, int, Dict, List, Tuple]]:
        res = self._generate_tile_dict_header(children)
        if self.content is not None:
            res['content'] = {'boundingVolume': self.content_bounding_volume.to_tiles3d_bounding_volume_dict(),
                              'uri': tile_content_uri}
        return res

    def _generate_link_proxy_tile_dict(self, tile_uri: str) -> Dict[str, Union[str, float, int, Dict, List, Tuple]]:
        res = self._generate_tile_dict_header([])
        res['content'] = {'boundingVolume': self.bounding_volume.to_tiles3d_bounding_volume_dict(),
                          'uri': tile_uri}
        return res