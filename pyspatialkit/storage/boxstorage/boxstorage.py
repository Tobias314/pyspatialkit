from pathlib import path
from typing import Union, Tuple
from threading import Lock

import numpy as np
from cachetools import LRUCache




class BoxStorageTileIndex:
    
    def __init__(self, storage: 'BoxStorage', identifier: Union[Tuple[int, int], Tuple[int,int,int]],
                 index_boxes: np.ndarray, index_ids: np.ndarray, max_item: int = 0):
        self.storage = storage
        self.identifier = identifier
        self.index_boxes = index_boxes
        self.index_ids = index_ids
        self.max_item = np.array([max_item], dtype=np.uint32)
        self.dims = len(identifier)

    def persist_to_file(self):
        np.savez(max_item=self.max_item, index_boxes=self.index_boxes, index_ids=self.index_ids)

    @classmethod
    def load_from_file(cls, storage: 'BoxStorage', identifier: Union[Tuple[int,int], Tuple[int,int,int]]):
        data = np.load(storage.get_tile_directory(identifier))
        max_item = data['max_item'][0]
        return cls(storage=storage, identifier=identifier, index_boxes=data['index_boxes'],
                   index_ids=data['index_ids'], max_item=max_item)

    def filter_boxes_by_bounds(self, bounds: np.ndarray) -> np.ndarray:
        requ_min, requ_max = bounds[:self.dims], bounds[self.dims:]
        index_mins = self.index_boxes[:, :self.dims]
        index_maxs = self.index_boxes[:, self.dims:]
        mask = np.logical_not((requ_min > index_maxs).any(axis=1) | (requ_max < index_mins).any(axis=1))
        return self.index_ids[mask]


class BoxStorage:

    def __init__(self, directory_path: Path, bounds: Union[Tuple[float,float,float,float], Tuple[float,float,float,float,float,float]],
                 tile_size: Union[Tuple[float, float], Tuple[float,float,float]],
                 tile_cache_size: int = 100, object_cache_size: int = 1000):
        self.directory_path= directory_path
        self.bounds = np.array(bounds)
        self.tile_size = np.array(tile_size)
        if len(self.tile_size) * 2 != len(self.bounds) * 2:
            raise AttributeError("Dimensionality of tile size does not match dimensionality of bounds")
        self.tile_cache_size = tile_cache_size
        self.object_cache_size = object_cache_size
        self.tile_cache = LRUCache(maxsize=self.tile_cache_size)
        self.tile_cache_read_lock = Lock()
        self.tile_cache_write_lock = Lock()
        self.object_cache = LRUCache(maxsize=self.object_cache_size)
        self.object_cache_read_lock = Lock()
        self.object_cache_write_lock = Lock()

    def persist_to_file(self):
        config = {}
        config['bounds'] = tuple(self.bounds)
        config['tile_size'] = tuple(self.tile_size)
        config['tile_cache_size'] = self.tile_cache_size
        config['object_cache_size'] = self.object_cache_size
        json.dump

    def get_tile_directory(self, identifier: Union[Tuple[int, int], Tuple[int, int, int]]):
        return self.directory_path / '_'.join([str(i) for i in identifier]) + '.index'

    @classmethod
    def load_from_file(cls, directory_path: Path):
         with open(directory_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            return cls(directory_path=directory_path, bounds=config['bounds'], tile_size=config['tile_size'],
                       tile_cache_size=config['tile_cache_size'])

    def get_tile(self, identifier: Tuple):
        try:
            self.tile_cache_read_lock.acquire(blocking=False)
            if not self.tile_cache_write_lock.locked():
                if identifier in self.tile_cache:
                    return self.tile_cache[identifier]
                else:
                    self.tile_cache[identifier] = self._load_tile_from_file(identifier)
        finally:
            self.tile_cache_read_lock.release()

    def write_tile(self, tile: BoxStorageTile):
        #TODO
        pass

    def get_tiles_for_bounds(self, bounds: np.ndarray):
        #TODO
        pass

