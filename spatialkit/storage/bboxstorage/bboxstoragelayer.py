from pathlib import Path
from typing import Union, Tuple, Type, List, Optional, Dict, TYPE_CHECKING
from threading import Lock
import json
from abc import ABC, abstractmethod, abstractclassmethod
import shutil

import numpy as np
import numpy.typing as npt
from cachetools import LRUCache
import fasteners
import csv

from ...utils.threading import RWLock

if TYPE_CHECKING:
    from .bboxstorage import BBoxStorage

LOCK_FILE_NAME = 'lock.lck'
INDEX_FILE_NAME = 'index'


class BBoxStorageObjectInterface(ABC):

    @abstractmethod
    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    @abstractclassmethod
    def from_bytes(cls, bytes) -> Optional['BBoxStorageObjectInterface']:
        raise NotImplementedError()

    @abstractmethod
    def is_intersecting_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_bounds(self)-> np.ndarray:
        raise NotImplementedError()


class BBoxStorageTileIndex:
    
    def __init__(self, storage_layer: 'BBoxStorageLayer', identifier: Union[Tuple[int, int], Tuple[int,int,int]],
                 own_index_bounds: npt.NDArray[float], own_index_ids: npt.NDArray[int],
                 foreign_index_bounds: npt.NDArray[float], foreign_index_ids: npt.NDArray[int], own_bbox: npt.NDArray[float],
                 max_item: int = 0):
        self.storage_layer = storage_layer
        self.identifier = identifier
        self.own_index_bounds = own_index_bounds
        self.foreign_index_bounds = foreign_index_bounds
        self.own_index_ids = own_index_ids
        self.foreign_index_ids = foreign_index_ids
        self.id_to_bounds: Dict[int, npt.NDArray[float]] = {}
        for bounds, object_id in zip(self.own_index_bounds, self.own_index_ids):
            self.id_to_bounds[object_id] = bounds
        self.unflushed_changes = False
        self.unflushed_own_index_bounds = []
        self.unflushed_foreign_index_bounds = []
        self.unflushed_own_index_ids = []
        self.unflushed_foreign_index_ids = []
        self.max_item = np.array([max_item], dtype=np.uint32)
        self.dims = len(identifier)
        self.own_bbox = own_bbox
        self._file_lock = None
        self._directory_path = None

    def persist_to_file(self):
        path = self.directory_path / INDEX_FILE_NAME
        with self.file_lock.write_lock():
            np.savez(file=path, max_item=self.max_item, own_index_bounds=self.own_index_bounds, own_index_ids=self.own_index_ids,
                    foreign_index_bounds=self.foreign_index_bounds, foreign_index_ids=self.foreign_index_ids)

    @classmethod
    def load_from_file(cls, storage_layer: 'BBoxStorageLayer', identifier: Union[Tuple[int,int], Tuple[int,int,int]]):
        try:
            directory_path = storage_layer.get_tile_directory(identifier)
            file_lock = fasteners.InterProcessReaderWriterLock(storage_layer.get_tile_directory(identifier) / LOCK_FILE_NAME)
            with file_lock.read_lock():
                data = np.load(directory_path / (INDEX_FILE_NAME + '.npz'))
            max_item = data['max_item'][0]
            own_bbox = data['own_bbox']
            return cls(storage_layer=storage_layer, identifier=identifier, 
                    own_index_bounds=data['own_index_bounds'],own_index_ids=data['own_index_ids'],
                    foreign_index_bounds=data['foreign_index_bounds'],foreign_index_ids=data['foreign_index_ids'],
                    own_bbox=own_bbox, max_item=max_item)
        except FileNotFoundError:
            bboxes_index_shape = (0,storage_layer.dims * 2)
            (storage_layer.get_tile_directory(identifier) / 'data').mkdir(exist_ok=True, parents=True)
            own_bbox = storage_layer.get_bounds_for_identifier(identifier)
            return cls(storage_layer=storage_layer, identifier=identifier, own_index_bounds = np.empty(bboxes_index_shape, dtype=float),
                       own_index_ids=np.empty(0, dtype=int), foreign_index_bounds=np.empty(bboxes_index_shape, dtype=float),
                       foreign_index_ids=np.empty((0,4), dtype=int), own_bbox=own_bbox, max_item=0)

    def get_bounds_for_index(self, index: int) -> npt.NDArray[float]:
        return self.id_to_bounds[index]

    def filter_boxes_by_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> np.ndarray:
        index_mins = self.own_index_bounds[:, :self.dims]
        index_maxs = self.own_index_bounds[:, self.dims:]
        mask = np.logical_not((requ_min > index_maxs).any(axis=1) | (requ_max < index_mins).any(axis=1))
        res1 = self.own_index_ids[mask]
        res1 = np.concatenate([np.repeat([self.identifier], res1.shape[0], axis=0), res1[:, np.newaxis]], axis=1)
        index_mins = self.foreign_index_bounds[:, :self.dims]
        index_maxs = self.foreign_index_bounds[:, self.dims:]
        mask = np.logical_not((requ_min > index_maxs).any(axis=1) | (requ_max < index_mins).any(axis=1))
        res2 = self.foreign_index_ids[mask]
        return np.concatenate([res1, res2], axis=0)

    def write_own_object(self, bounds: np.ndarray, obj: BBoxStorageObjectInterface, flush=True) -> int:
        assert len(bounds) == 2 * self.dims
        self.max_item += 1
        if self.max_item==0: #check for max_item overflow, in this case we need to flus to prevent conflicts with unflushed ids
            self.flush()
        while (self.own_index_ids == self.max_item.item()).any():
            self.max_item += 1
        self.storage_layer.write_object_to_file(obj, self.identifier, self.max_item.item())
        self.id_to_bounds[self.max_item.item()] = bounds
        self.unflushed_own_index_bounds.append(bounds)
        self.unflushed_own_index_ids.append(self.max_item.item())
        self.unflushed_changes = True
        self.update_own_bbox(bounds) #TODO: in worst case we might do this quite often, causing many flushs of parent tiles, maybe make batched write_own_objects methdo
        if flush:
            self.flush()
        return self.max_item.item()

    def update_own_bbox(self, bounds: np.ndarray):
        dims = self.dims
        if (bounds[:dims] < self.own_bbox[:dims]).any() or (bounds[dims:] > self.own_bbox[dims:]).any():
            self.own_bbox[:dims] = np.minimum(self.own_bbox[:dims], bounds[:dims])
            self.own_bbox[dims:] = np.minimum(self.own_bbox[dims:], bounds[dims:])
            parent_tile = self.storage.get_parent_tile_index_for_identifier(identifier)
            if parent_tile is not None:
                parent_tile.update_own_bbox(bounds)
                parent_tile.flush()
    
    def write_foreign_object(self, bounds: np.ndarray, obj: BBoxStorageObjectInterface, foreign_tile_identifier, foreign_object_identifier,
                             flush=True) -> int:
        assert bounds.shape[0] == 2 * self.dims
        self.unflushed_foreign_index_bounds.append(bounds)
        self.unflushed_foreign_index_ids.append((*foreign_tile_identifier, foreign_object_identifier))
        if flush:
            self.flush()
        else:
            self.unflushed_changes = True

    def flush(self):
        if self.unflushed_changes:
            if self.unflushed_foreign_index_bounds:
                self.foreign_index_bounds = np.concatenate([self.foreign_index_bounds, self.unflushed_foreign_index_bounds])
                self.foreign_index_ids = np.concatenate([self.foreign_index_ids, self.unflushed_foreign_index_ids])
                self.unflushed_foreign_index_bounds = []
                self.unflushed_foreign_index_ids = []
            if self.unflushed_own_index_bounds:
                self.own_index_bounds = np.concatenate([self.own_index_bounds, self.unflushed_own_index_bounds], axis=0)
                self.own_index_ids = np.concatenate([self.own_index_ids, self.unflushed_own_index_ids])
                self.unflushed_own_index_bounds = []
                self.unflushed_own_index_ids = []
        self.persist_to_file()
        self.unflushed_changes = False

    def get_valid_object_ids(self) -> List[int]:
        return list(self.id_to_bounds.keys())

    @property
    def directory_path(self):
        if self._directory_path is None:
            self._directory_path = self.storage.get_tile_directory(self.identifier)
        return self._directory_path

    @property
    def file_lock(self):
        if self._file_lock is None:
            self._file_lock = fasteners.InterProcessReaderWriterLock(self.directory_path / LOCK_FILE_NAME)
        return self._file_lock

class BBoxStorageLayer:

    def __init__(self, directory_path: Path, bounds: Union[Tuple[float,float,float,float], Tuple[float,float,float,float,float,float]],
                 tile_size: Union[Tuple[float, float], Tuple[float,float,float]], object_type: BBoxStorageObjectInterface,
                 tile_cache_size: int = 100, object_cache_size: int = 1000,
                 storage: Optional['BBoxStorage'] = None, parent_layer: Optional['BBoxStorageLayer'] = None,
                 tile_bboxes: Dict[Tuple,Tuple[Tuple,Tuple]] = {}):
        self.directory_path= directory_path
        self.bounds = np.array(bounds)
        if len(tile_size) * 2 != len(self.bounds):
            raise AttributeError("Dimensionality of tile size does not match dimensionality of bounds")
        self.tile_size = np.array(tile_size)
        self.dims = len(self.tile_size)
        self.tile_cache_size = tile_cache_size
        self.object_cache_size = object_cache_size
        self.tile_cache = LRUCache(maxsize=self.tile_cache_size)
        self.tile_cache_lock = Lock()
        self.object_cache = LRUCache(maxsize=self.object_cache_size)
        self.object_cache_lock = Lock()
        self.object_type = object_type
        self.rw_lock = RWLock()
        self.storage = storage
        self.parent_layer = parent_layer
        self.tile_bboxes = tile_bboxes
        self._file_lock = None
        self.has_pyramid = False #TODO: change when implementing pyramids

    @property
    def file_lock(self):
        if self._file_lock is None:
            self._file_lock =  fasteners.InterProcessReaderWriterLock(self.directory_path / LOCK_FILE_NAME)
        return self._file_lock

    def persist_to_file(self,bbox_tiles_only=False):
        with self.file_lock.write_lock():
            if not bbox_tiles_only:
                config = {}
                config['bounds'] = [float(b) for b in self.bounds]
                config['tile_size'] = [float(s) for s in self.tile_size]
                config['tile_cache_size'] = int(self.tile_cache_size)
                config['object_cache_size'] = int(self.object_cache_size)
                with open(self.directory_path / 'config.json', 'w') as json_file:
                    json.dump(config, json_file)
            with open(self.directory_path / 'tile_bboxes.csv', 'w') as tile_bboxes_file:
                    csv_writer = csv.writer(tile_bboxes_file, delimiter=',')
                    for key, bbox in self.tile_bboxes.items():
                        csv_writer.writerow(list(key) + list(bbox[0]) + list(bbox[1]))
            

    def get_tile_directory(self, identifier: Union[Tuple[int, int], Tuple[int, int, int]]):
        return self.directory_path / '_'.join([str(i) for i in identifier])

    @classmethod
    def load_from_file(cls, directory_path: Path, object_type: Type):
        file_lock =  fasteners.InterProcessReaderWriterLock(directory_path / LOCK_FILE_NAME)
        tile_bboxes = {}
        with file_lock.read_lock():
            with open(directory_path / 'tile_bboxes.csv') as tile_bboxes_file:
                csv_reader = csv.reader(tile_bboxes_file, delimiter=',',)
                for row in csv_reader:
                    tile_bboxes[(row[0], row[1], row[2])] = (row[3:6], row[6:])
            with open(directory_path / 'config.json') as json_file:
                config = json_file.read()
                config = json.loads(config)
                res =  cls(directory_path=directory_path, bounds=config['bounds'], tile_size=config['tile_size'],
                           tile_cache_size=config['tile_cache_size'], object_type=object_type, tile_bboxes = tile_bboxes)
        res.persist_to_file(bbox_tiles_only=True)

    def get_tile(self, identifier: Tuple):
        with self.tile_cache_lock:
            if identifier in self.tile_cache:
                return self.tile_cache[identifier]
            else:
                tile = BBoxStorageTileIndex.load_from_file(self, identifier)
                self.tile_cache[identifier] = tile
                return tile

    def get_parent_tile(self, identifier: Tuple) -> Optional[BBoxStorageTileIndex]:
        if self.storage is None:
            return None
        identifier = tuple((np.array(identifier) / self.storage.chunk_size).astype(int))
        if self.parent_layer is None:
            return None
        return self.parent_layer.get_tile(identifier)


    def _get_object_path(self, tile_indices: Tuple, object_identifier: int) -> Path:
        return self.get_tile_directory(tile_indices) / 'data' / (str(object_identifier) + '.geoobj')

    def write_object_to_file(self, obj: BBoxStorageObjectInterface, tile_indices: Tuple, object_identifier: int):
        tile_indices = tuple(tile_indices)
        object_identifier = int(object_identifier)
        with self.object_cache_lock:
            identifier = (tile_indices, object_identifier)
            b = obj.to_bytes()
            with open(self._get_object_path(tile_indices, object_identifier), 'wb') as f:
                f.write(b)
            self.object_cache[identifier] = obj

    def get_object_ids_for_tile(self, tile_indices: Tuple) -> List[int]:
        tile = self.get_tile(tile_indices)
        with self.rw_lock.r_locked():
            return tile.get_valid_object_ids()
    
    def get_bounds_for_identifiert(self, tile_indices: Tuple, object_identifier: int) -> npt.NDArray[float]:
        tile = self.get_tile(tile_indices)
        with self.rw_lock.r_locked():
            return tile.get_bounds_for_index(object_identifier)

    def get_object_for_identifier(self, tile_indices: Tuple, object_identifier: int) -> Optional[BBoxStorageObjectInterface]:
        tile_indices = tuple(tile_indices)
        object_identifier = int(object_identifier)
        with self.object_cache_lock:
            identifier = (tile_indices, object_identifier)
            if identifier in self.object_cache:
                return self.object_cache[identifier]
            else:
                with open(self._get_object_path(tile_indices, object_identifier), 'rb') as f:
                    res = self.object_type.from_bytes(f.read())
                self.object_cache[identifier] = res
                return res

    def get_bounds_for_identifier(self, identifier: npt.NDArray[int]) -> npt.NDArray[float]:
        bounds = np.zeros(2 * self.dims)
        bounds[:self.dims] = self.bounds[:self.dims] + identifier * self.tile_size
        bounds[self.dims:] = bounds[:self.dims] + self.tile_size
        if (identifier < 0).any() or (bounds[self.dims:] > self.bounds[self.dims:]).any():
            raise IndexError("Invalid identifier!")
        return bounds

    def get_tile_identifiers_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> np.ndarray:
        requ_min = ((requ_min - self.bounds[:self.dims]) / self.tile_size).astype(int)
        requ_max = np.ceil((requ_max - self.bounds[:self.dims]) / self.tile_size).astype(int)
        ranges = [np.arange(axis_min, axis_max) for axis_min, axis_max in zip(requ_min, requ_max)]
        tiles = np.stack([a.flatten() for a in np.meshgrid(*ranges, indexing='ij')], axis=1)
        return tiles

    def _get_tiles_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> List[BBoxStorageTileIndex]:
        tiles = self.get_tile_identifiers_for_bbox(requ_min, requ_max)
        res = []
        for tile_index in tiles:
            res.append(self.get_tile(tuple(tile_index)))
        return res

    def get_objects_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray) -> List[BBoxStorageObjectInterface]:
        with self.rw_lock.r_locked():
            tiles = self._get_tiles_for_bbox(requ_min, requ_max)
            candidates  = []
            for tile in tiles:
                candidates.append(tile.filter_boxes_by_bbox(requ_min, requ_max))
            candidates = np.unique(np.concatenate(candidates, axis=0), axis=0)
            candidates = np.split(candidates, np.unique(candidates[:, :self.dims], return_index=True, axis=0)[1][1:])
            res = []
            for tile_candidates in candidates:
                for candidate in tile_candidates:
                    obj = self.get_object_for_identifier(candidate[:self.dims], candidate[-1])
                    #if obj.is_intersecting_bbox(requ_min, requ_max):
                    res.append(obj)
            return res

    def write_objecs(self, objects: List[BBoxStorageObjectInterface]):
        with self.rw_lock.w_locked():
            bounds = np.array([obj.get_bounds() for obj in objects])
            assert bounds.shape[1] == 2 * self.dims
            bbox_mins, bbox_maxs = bounds[:,:self.dims], bounds[:,self.dims:]
            centers = (((bbox_maxs - bbox_mins) / 2 - self.bounds[:self.dims]) / self.tile_size).astype(int)
            written_tiles = set()
            new_tile_bboxes = {}
            for i, (bbox_min, bbox_max, center, obj) in enumerate(zip(bbox_mins, bbox_maxs, centers, objects)):
                obj_tiles = self.get_tile_identifiers_for_bbox(bbox_min, bbox_max)
                identifier = tuple(center)
                tile = self.get_tile(identifier)
                if identifier not in self.tile_bboxes:
                    self.tile_bboxes[identifier] = (bbox_min, bbox_max)
                    new_tile_bboxes[identifier] = (bbox_min, bbox_max)
                else:
                    old_min, old_max = self.tile_bboxes[identifier]
                    old_min, old_max = np.array(old_min), np.array(old_max)
                    new_min = np.minimum(old_min, bbox_min)
                    new_max = np.maximum(old_max, bbox_max)
                    if (new_min!=old_min).any() or (new_max != old_max).any():
                        self.tile_bboxes[identifier] = (new_min, new_max)
                        new_tile_bboxes[identifier] = (new_min, new_max)
                object_bounds = np.array([*bbox_min, *bbox_max])
                obj_id = tile.write_own_object(object_bounds, obj, flush=False)
                written_tiles.add(tile)
                for tile_id in obj_tiles:
                    tile_id = tuple(tile_id)
                    if tile_id!=identifier:
                        tile = self.get_tile(tile_id)
                        tile.write_foreign_object(object_bounds, obj, identifier, obj_id, flush=False)
                        written_tiles.add(tile)
            for tile in written_tiles:
                tile.flush()
            with self.file_lock.write_lock():
                with open(self.directory_path / 'tile_bboxes.csv', 'a') as tile_bboxes_file:
                    csv_writer = csv.writer(tile_bboxes_file, delimiter=',')
                    for key, bbox in new_tile_bboxes.items():
                        csv_writer.writerow(list(key) + list(bbox[0]) + list(bbox[1]))

    def invalidate_cache(self):
        self.tile_cache = LRUCache(self.tile_cache_size)
        self.object_cache = LRUCache(self.object_cache_size)

    def delete_permanently(self):
        shutil.rmtree(self.directory_path)


