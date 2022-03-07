from pathlib import Path
from typing import Union, Tuple, Type, List, Optional
from threading import Lock
import json
from abc import ABC, abstractmethod, abstractclassmethod

import numpy as np
from cachetools import LRUCache

from ...utils.threading import RWLock


class BoxStorageObjectInterface(ABC):

    @abstractmethod
    def to_file(self, path: Path):
        raise NotImplementedError()

    @abstractclassmethod
    def from_file(cls, path: Path) -> Optional['BoxStorageObjectInterface']:
        raise NotImplementedError()

    @abstractmethod
    def is_intersecting_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_bounds(self)-> np.ndarray:
        raise NotImplementedError()


class BoxStorageTileIndex:
    
    def __init__(self, storage: 'BoxStorage', identifier: Union[Tuple[int, int], Tuple[int,int,int]],
                 own_index_boxes: np.ndarray, own_index_ids: np.ndarray,
                 foreign_index_boxes: np.ndarray, foreign_index_ids: np.ndarray,
                 max_item: int = 0):
        self.storage = storage
        self.identifier = identifier
        self.own_index_boxes = own_index_boxes
        self.foreign_index_boxes = foreign_index_boxes
        self.own_index_ids = own_index_ids
        self.foreign_index_ids = foreign_index_ids
        self.unflushed_changes = False
        self.unflushed_own_index_boxes = []
        self.unflushed_foreign_index_boxes = []
        self.unflushed_own_index_ids = []
        self.unflushed_foreign_index_ids = []
        self.max_item = np.array([max_item], dtype=np.uint32)
        self.dims = len(identifier)

    def persist_to_file(self):
        path = self.storage.get_tile_directory(self.identifier) / '.index'
        np.savez(file=path, max_item=self.max_item, own_index_boxes=self.own_index_boxes, own_index_ids=self.own_index_ids,
                 foreign_index_boxes=self.foreign_index_boxes, foreign_index_ids=self.foreign_index_ids)

    @classmethod
    def load_from_file(cls, storage: 'BoxStorage', identifier: Union[Tuple[int,int], Tuple[int,int,int]]):
        data = np.load(storage.get_tile_directory(identifier) / '.index')
        max_item = data['max_item'][0]
        return cls(storage=storage, identifier=identifier, 
                   own_index_boxes=data['own_index_boxes'],own_index_ids=data['own_index_ids'],
                   foreign_index_boxes=data['foreign_index_boxes'],foreign_index_ids=data['foreign_index_ids'],
                   max_item=max_item)

    def filter_boxes_by_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> np.ndarray:
        index_mins = self.own_index_boxes[:, :self.dims]
        index_maxs = self.own_index_boxes[:, self.dims:]
        mask = np.logical_not((requ_min > index_maxs).any(axis=1) | (requ_max < index_mins).any(axis=1))
        res1 = self.own_index_ids[mask]
        res1 = np.concatenate([np.repeat([self.identifier], res1.shape[0], axis=0), res1[:, np.newaxis]])
        ndex_mins = self.foreign_index_boxes[:, :self.dims]
        index_maxs = self.foreign_index_boxes[:, self.dims:]
        mask = np.logical_not((requ_min > index_maxs).any(axis=1) | (requ_max < index_mins).any(axis=1))
        res2 = self.foreign_index_ids[mask]
        return np.concatenate([res1, res2], axis=0)

    def write_own_object(self, bounds: np.ndarray, obj: BoxStorageObjectInterface, flush=True) -> int:
        assert bounds.shape[0] == 2 * self.dims
        self.max_item += 1
        while (self.own_index_ids == self.max_item).any():
            self.max_item += 1
        self.storage.write_object_to_file(obj, self.identifier, self.max_item)
        self.unflushed_own_index_boxes.append(bounds)
        self.unflushed_own_index_ids.append(self.max_item)
        if flush:
            self.flush()
        else:
            self.unflushed_changes = True
        return self.max_item
    
    def write_foreign_object(self, bounds: np.ndarray, obj: BoxStorageObjectInterface, foreign_tile_identifier, foreign_object_identifier,
                             flush=True) -> int:
        assert bounds.shape[0] == 2 * self.dims
        self.unflushed_foreign_index_boxes.append(bounds)
        self.unflushed_foreign_index_ids.append((*foreign_tile_identifier, foreign_object_identifier))
        if flush:
            self.flush()
        else:
            self.unflushed_changes = True
        return self.max_item

    def flush(self):
        if self.unflushed_changes:
            if self.unflushed_foreign_index_boxes:
                self.foreign_index_boxes = np.concatenate([self.foreign_index_boxes, self.unflushed_foreign_index_boxes])
                self.foreign_index_ids = np.concatenate([self.foreign_index_ids, self.unflushed_foreign_index_ids])
            if self.unflushed_own_index_boxes:
                self.own_index_boxes = np.concatenate([self.own_index_boxes, self.unflushed_own_index_boxes])
                self.own_index_ids = np.concatenate([self.own_index_ids, self.unflushed_own_index_ids])
        self.persist_to_file()
        self.unflushed_changes = False


class BoxStorage:

    def __init__(self, directory_path: Path, bounds: Union[Tuple[float,float,float,float], Tuple[float,float,float,float,float,float]],
                 tile_size: Union[int, float, Tuple[float, float], Tuple[float,float,float]], object_type: BoxStorageObjectInterface,
                 tile_cache_size: int = 100, object_cache_size: int = 1000):
        self.directory_path= directory_path
        self.bounds = np.array(bounds)
        self.tile_size = np.array(tile_size)
        if len(tile_size) == 0:
            self.tile_size = np.array([tile_size for i in range(len(self.bounds) / 2)])
        if len(self.tile_size) * 2 != len(self.bounds):
            raise AttributeError("Dimensionality of tile size does not match dimensionality of bounds")
        self.dims = len(self.tile_size)
        self.tile_cache_size = tile_cache_size
        self.object_cache_size = object_cache_size
        self.tile_cache = LRUCache(maxsize=self.tile_cache_size)
        self.tile_cache_lock = Lock()
        self.object_cache = LRUCache(maxsize=self.object_cache_size)
        self.object_cache_lock = Lock()
        self.object_type = object_type
        self.rw_lock = RWLock()

    def persist_to_file(self):
        config = {}
        config['bounds'] = tuple(self.bounds)
        config['tile_size'] = tuple(self.tile_size)
        config['tile_cache_size'] = self.tile_cache_size
        config['object_cache_size'] = self.object_cache_size
        json.dump

    def get_tile_directory(self, identifier: Union[Tuple[int, int], Tuple[int, int, int]]):
        return self.directory_path / '_'.join([str(i) for i in identifier])

    @classmethod
    def load_from_file(cls, directory_path: Path, object_type: Type):
         with open(directory_path / 'config.json') as json_file:
            config = json_file.read()
            config = json.loads(config)
            return cls(directory_path=directory_path, bounds=config['bounds'], tile_size=config['tile_size'],
                       tile_cache_size=config['tile_cache_size'], object_type=object_type)

    def get_tile(self, identifier: Tuple):
        with self.tile_cache_lock:
            if identifier in self.tile_cache:
                return self.tile_cache[identifier]
            else:
                tile = BoxStorageTileIndex.load_from_file(self, identifier)
                self.tile_cache[identifier] = tile
                return tile

    def write_object_to_file(self, obj: BoxStorageObjectInterface, tile_identifier: Tuple, object_identifier: int):
        tile_identifier = tuple(tile_identifier)
        object_identifier = int(object_identifier)
        with self.object_cache_lock:
            identifier = (tile_identifier, object_identifier)
            path = self.get_tile_directory(tile_identifier) / 'data' / str(object_identifier) + '.geoobj'
            obj.to_file(path)
            self.object_cache[identifier] = obj

    def get_object_for_identifier(self, tile_identifier: Tuple, object_identifier: int) -> Optional[BoxStorageObjectInterface]:
        tile_identifier = tuple(tile_identifier)
        object_identifier = int(object_identifier)
        with self.object_cache_lock:
            identifier = (tile_identifier, object_identifier)
            if identifier in self.object_cache:
                return self.object_cache[identifier]
            else:
                path = self.get_tile_directory(tile_identifier) / 'data' / str(object_identifier) + '.geoobj'
                res =  self.object_type.from_file(path)
                self.object_cache[identifier] = res
                return res

    def get_tile_identifiers_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> np.ndarray:
        requ_min = ((requ_min - self.bounds[:self.dims]) / self.tile_size).astype(int)
        requ_max = np.ceil((requ_max - self.bounds[self.dims:]) / self.tile_size).astype(int)
        ranges = [np.arange(axis_min, axis_max) for axis_min, axis_max in zip(requ_min, requ_max)]
        tiles = np.stack([a.flatten() for a in np.meshgrid(*ranges, indexing='ij')], axis=1)
        return tiles

    def get_tiles_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> List[BoxStorageTileIndex]:
        tiles = self.get_tile_identifiers_for_bbox(requ_min, requ_max)
        res = []
        for tile_index in tiles:
            res.append(self.get_tile(tile_index))
        return res

    def get_objects_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> List[BoxStorageObjectInterface]:
        with self.rw_lock.r_locked():
            tiles = self.get_tiles_for_bbox(requ_min, requ_max)
            candidates  = []
            for tile_index in tiles:
                candidates.append(self.get_tile(tile_index).filter_boxes_by_bbox(requ_min, requ_max))
            candidates = np.unique(np.concatenate(candidates, axis=0), axis=0)
            candidates = np.split(candidates, np.unique(candidates[:, :self.dims], return_index=True, axis=0)[1][1:])
            res = []
            for candidate in candidates:
                obj = self.get_object_for_identifier(candidate[:self.dims], candidate[-1])
                if obj.is_intersecting_bbox(requ_min, requ_max):
                    res.append(obj)
            return res

    def write_objecs(self, objects: List[BoxStorageObjectInterface]):
        with self.rw_lock.w_locked():
            bounds = np.array([obj.get_bounds() for obj in objects])
            assert bounds.shape[1] == 2 * self.dims
            bbox_mins, bbox_maxs = bounds[:,:self.dims], bounds[:,self.dims:]
            centers = (((bbox_maxs - bbox_mins) / 2 - self.bounds[:self.dims]) / self.tile_size).astype(int)
            written_tiles = set()
            for i, (bbox_min, bbox_max, center, obj) in enumerate(zip(bbox_mins, bbox_maxs, centers, objects)):
                obj_tiles = self.get_tile_identifiers_for_bbox(bbox_min, bbox_max)
                identifier = tuple(center)
                tile = self.get_tile(identifier)
                obj_id = tile.write_own_object([*bbox_min, *bbox_max], obj)
                written_tiles.add(tile)
                for tile_id in obj_tiles:
                    if tile_id!=identifier:
                        tile = self.get_tile(tile_id)
                        tile.write_foreign_object([*bbox_min, *bbox_max], obj, tile_id, obj_id)
                        written_tiles.add(tile)
            for tile in written_tiles:
                tile.flush()

    def invalidate_cache(self):
        self.tile_cache = LRUCache(self.tile_cache_size)
        self.object_cache = LRUCache(self.object_cache_size)


