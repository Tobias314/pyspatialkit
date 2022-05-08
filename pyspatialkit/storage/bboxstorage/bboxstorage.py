from typing import List, Dict, Tuple
from pathlib import Path

#TODO: finish this
#   - load layers from file system (create layer folders if not exist)
#   - implement interface for storing and retrieving data
class BBoxStorageLayer:
    def __init__(self, directory_path: Path, bounds:  List[float], tile_size: List[float], object_type: BBoxStorageObjectInterface,
                 chunk_size: List[int] = 2, tile_cache_size: int = 100, object_cache_size: int = 1000,
                 tile_bboxes: Dict[Tuple,Tuple[Tuple,Tuple]] = {}):
        self.directory_path = directory_path
        self.bounds = np.array(bounds)
        if len(tile_size) * 2 != len(self.bounds) or len(chunk_size) != len(tile_size):
            raise AttributeError("Dimensionality of tile_size, chunk_size, and bounds need to match")
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
        res.persist_to_file(bbox_tiles_only=True)

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

    def get_tile_identifiers_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray, level: Optional[int] = None) -> np.ndarray:
        requ_min = ((requ_min - self.bounds[:self.dims]) / self.tile_size).astype(int)
        requ_max = np.ceil((requ_max - self.bounds[:self.dims]) / self.tile_size).astype(int)
        ranges = [np.arange(axis_min, axis_max) for axis_min, axis_max in zip(requ_min, requ_max)]
        tiles = np.stack([a.flatten() for a in np.meshgrid(*ranges, indexing='ij')], axis=1)
        return tiles

    def _get_tiles_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray, level: Optional[int] = None) -> List[BBoxStorageTileIndex]:
        tiles = self.get_tile_identifiers_for_bbox(requ_min, requ_max, level)
        res = []
        for tile_index in tiles:
            res.append(self.get_tile(tuple(tile_index)))
        return res

    def get_objects_for_bbox(self, requ_min: np.ndarray, requ_max: np.ndarray, level: Optional[int] = None) -> List[BBoxStorageObjectInterface]:
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

    def write_objecs(self, objects: List[BBoxStorageObjectInterface], level: Optional[int] = None):
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

    def invalidate_cache(self, levels: Optional[List[int]] = None):
        self.tile_cache = LRUCache(self.tile_cache_size)
        self.object_cache = LRUCache(self.object_cache_size)

    def delete_permanently(self):
        shutil.rmtree(self.directory_path)