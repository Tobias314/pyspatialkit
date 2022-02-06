from typing import Union, Dict, Optional, TYPE_CHECKING, Tuple, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path

from .tile3d import Tile3d

TILES3D_VERSION = '1.0'

class Tileset3d(ABC):

    def __init__(self):
        self._root: Optional[Tile3d] = None

    @property
    def tileset_version(self) -> Union[float, str]:
        return 1.0

    @property
    def properties(self) -> Dict[str, Dict[str, float]]:
        return {}

    @property
    def geometric_error(self) -> float:
        return 0

    @property
    def root(self) -> Tile3d:
        if self._root is None:
            self._root = self.get_root()
        return self._root

    @abstractmethod
    def get_root(self) -> Tile3d:
        raise NotImplementedError

    @abstractmethod
    def get_tile_by_identifier(self,identifier: object)-> Tile3d:
        raise NotImplementedError

    #TODO fix cost/max_cost calculation
    def materialize(self, tile_uri_generator: Callable[['Tile3d'], str],
                     tile_content_uri_generator: Optional[Callable[['Tile3d'], str]] = None,
                     root_tile: Optional[Tile3d] = None,
                     max_depth: Optional[int] = None, max_cost: Optional[float] = None,
                     callback: Optional[Callable[['Tile3d'], None]] = None) -> Tuple[Dict[str, Union[str, float, int, Dict, List, Tuple]], List['Tile3d']]:
        if root_tile is None:
            root_tile = self.root
            geometric_error = self.geometric_error
        else:
            geometric_error = root_tile.geometric_error
        tile_dict, end_list = root_tile.materialize(tile_uri_generator=tile_uri_generator, tile_content_uri_generator=tile_content_uri_generator,
                                                     current_depth=0, accumulated_cost=0, max_depth=max_depth, max_cost=max_cost,
                                                     callback=callback)
        res_dict = self._generate_tileset_dict(tileset_version=self.tileset_version, geometric_error=geometric_error, root=tile_dict,
                                            properties=self.properties)
        return res_dict, end_list

    def _generate_tileset_dict(self, tileset_version: str, geometric_error: float,
                                 root: Dict, properties: Dict = {}) -> Dict[str, Union[str, float, int, Dict, List, Tuple]]:
        res = {
            'asset': {'version':TILES3D_VERSION, 'tilesetVersion': tileset_version},
            'properties': properties,
            'geometricError': geometric_error,
            'root': root
        }
        return res



