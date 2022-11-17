from typing import Tuple, Union, Optional
from pathlib import Path
import math
import time
import shutil
from threading import Thread

import numpy as np
import tiledb
from tqdm.auto import tqdm

from pyspatialkit.core.utils.numpy import next_bigger_dtype
from ..utils.tiledb import consolidate_and_vacuume

class TileDbDenseBackend:
    """Backend for storing raster data in (several) TileDB databases.
    
    The TileDB can be thought of as very big array with [col,row] indexing

    """

    def __init__(self, bounds: Tuple[float, float, float, float], num_bands: int, dtype: np.dtype, directory_path: Path,
                 pixel_size_xy: Tuple[float, float] = (1,1), tile_size: Tuple[int, int] = (1000, 1000), fill_value = 0,
                 build_pyramid: bool = True, num_pyramid_layers: Union[int, str] = 'all') -> None:
        self.bounds = np.array(bounds)
        self.dims = np.array((bounds[3] - bounds[1] / pixel_size_xy[1], bounds[2] - bounds[0] / pixel_size_xy[0]))
        self.pixel_size_xy = np.array(pixel_size_xy)
        self.tile_size = np.array(tile_size, dtype=int)
        directory_path = Path(directory_path)
        assert(directory_path.is_dir)
        directory_path.mkdir(parents=True, exist_ok=True)
        self.directory_path = directory_path
        self.build_pyramid = build_pyramid
        if self.build_pyramid :
            if num_pyramid_layers == 'all':
                self.num_pyramid_layers = min(math.ceil(math.log2(self.dims[0])), math.ceil(math.log2(self.dims[1])))
            else:
                self.num_pyramid_layers = num_pyramid_layers
        else:
            self.num_pyramid_layers = 0
        self.layers = [None,] * (self.num_pyramid_layers + 1)
        self.num_bands = num_bands
        self.dtype = dtype
        self.next_bigger_dtype = next_bigger_dtype(self.dtype)
        self.band_attribute_dtype = np.dtype(','.join([np.dtype(self.dtype).str] * self.num_bands))
        self.fill_value = fill_value
        band_attribute = tiledb.Attr(dtype=self.band_attribute_dtype, fill=fill_value)
        for i, layer in enumerate(self.layers):
            path = str(self.directory_path / ("layer_" + str(i)))
            if not Path(path).exists():
                dim = np.ceil(self.dims / 2**i).astype(int)
                tile_size = np.stack([np.ceil(dim / 2), self.tile_size], axis=1).min(axis=1)
                max32 = np.iinfo(np.int32).max
                dtypes = (np.int32 if self.dims[0] > max32 else np.int64,  np.int32 if self.dims[1] > max32 else np.int64)
                dimy = tiledb.Dim(name="dimy" + str(i), domain=(0, dim[0]-1), tile=tile_size[0], dtype=dtypes[0])
                dimx = tiledb.Dim(name="dimx" + str(i), domain=(0, dim[1]-1), tile=tile_size[1], dtype=dtypes[1])
                dom = tiledb.Domain(dimy, dimx)
                schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=[band_attribute,])
                schema.check()
                tiledb.Array.create(path, schema)
            self.layers[i] = [path, None]
        if self.build_pyramid:
            self.dirty_regions = set()
        self._consolidation_thread = None

    def _bounds_to_indexes(self, bounds: Tuple[float, float, float, float]) -> np.ndarray:
        index_bounds = np.array(bounds)
        index_bounds[:2] = (index_bounds[:2] - self.bounds[:2]) / self.pixel_size_xy
        index_bounds[2:4] = (index_bounds[2:4] - self.bounds[:2]) / self.pixel_size_xy
        index_bounds = np.array([self.dims[0] - index_bounds[3], index_bounds[0], self.dims[0] - index_bounds[1], index_bounds[2]], dtype=int)
        #print("got bounds:" + str(bounds))
        #print("to Indexes:" + str(index_bounds))
        return index_bounds

    def _read_array_region(self, db_read: tiledb.DenseArray, ax0_range: Tuple[int, int], ax1_range: Tuple[int, int]) -> np.ndarray:
        ax0_range = np.array(ax0_range, dtype=int)
        ax1_range = np.array(ax1_range, dtype=int)
        ax0_clipped_range = np.clip(ax0_range, 0, db_read.shape[0])
        ax1_clipped_range = np.clip(ax1_range, 0, db_read.shape[1])
        if (ax0_range!=ax0_clipped_range).any() or (ax1_range!=ax1_clipped_range).any():
            res_shape = (ax0_range[1] - ax0_range[0], ax1_range[1] - ax1_range[0], self.num_bands)
            res = np.full(res_shape, self.fill_value, dtype=self.dtype)
            tmp = db_read[ax0_clipped_range[0]:ax0_clipped_range[1], ax1_clipped_range[0]:ax1_clipped_range[1]]
            tmp = tmp.view((self.dtype, (self.num_bands,)))
            ax0_clipped_range -= ax0_range[0]
            ax1_clipped_range -= ax1_range[0]
            res[ax0_clipped_range[0]:ax0_clipped_range[1], ax1_clipped_range[0]:ax1_clipped_range[1]] = tmp
        else:
            res = db_read[ax0_clipped_range[0]:ax0_clipped_range[1], ax1_clipped_range[0]:ax1_clipped_range[1]]
            res = res.view((self.dtype, (self.num_bands,)))
        return res

    def _write_array_region(self, db_write: tiledb.DenseArray, ax0_range: Tuple[int, int], ax1_range: Tuple[int, int], data: np.ndarray) -> None:
        ax0_range = np.array(ax0_range, dtype=int)
        ax1_range = np.array(ax1_range, dtype=int)
        ax0_clipped = np.clip(ax0_range, 0, db_write.shape[0])
        ax1_clipped = np.clip(ax1_range, 0, db_write.shape[1])
        d = np.ascontiguousarray(data)
        d = d.view(self.band_attribute_dtype)
        if (ax0_range!=ax0_clipped).any() or (ax1_range!=ax1_clipped).any():
            ax0_clipped_local = ax0_clipped - ax0_range[0]
            ax1_clipped_local = ax1_clipped - ax1_range[0]
            db_write[ax0_clipped[0]: ax0_clipped[1], ax1_clipped[0]: ax1_clipped[1]] = d[ax0_clipped_local[0]: ax0_clipped_local[1], ax1_clipped_local[0]: ax1_clipped_local[1]]
        else:
            db_write[ax0_range[0]: ax0_range[1], ax1_range[0]: ax1_range[1]] = d


    def write_data(self, bounds: Tuple[float, float, float, float], data: np.ndarray) -> None:
        #TODO check whether bounds in layer bounds
        assert data.shape[-1] == self.num_bands
        assert data.dtype == self.dtype
        data = data.squeeze()
        indexes = self._bounds_to_indexes(bounds)
        path = self.layers[0][0]
        layer = self.layers[0][1]
        if layer is not None:
            layer.close()
        with tiledb.DenseArray(path, mode='w') as db:
            self._write_array_region(db, (indexes[0],indexes[2]), (indexes[1],indexes[3]), data=data)
            self.layers[0][1] = tiledb.DenseArray(path, mode='r')
            if self.build_pyramid:
                self.dirty_regions.add(tuple(indexes))
            db.close()

    #TODO: merge bounds at every level to increase performance of batch updates
    def update_pyramid(self, print_progress: bool = False) -> None:
        print("UPDATING PYRAMIDS")
        dirty_regions = self.dirty_regions
        layer_iterator = range(1, self.num_pyramid_layers+1)
        if print_progress:
            layer_iterator = tqdm(layer_iterator)
        for layer in layer_iterator:
            write_db_path = self.layers[layer][0]
            new_dirty_regions = []
            if self.layers[layer][1] is not None:
                self.layers[layer][1].close()
                self.layers[layer][1] = None
            with tiledb.DenseArray(write_db_path, mode='w') as db:
                lower_layer = self.layers[layer-1]
                if  lower_layer[1] is None:
                    lower_layer[1] = tiledb.DenseArray(lower_layer[0], mode='r')
                dirty_regions_iterator = tqdm(dirty_regions)
                if print_progress:
                    layer_iterator = tqdm(dirty_regions_iterator)
                for index_bounds in dirty_regions_iterator:
                    index_bounds = np.array(index_bounds)
                    upper_level_index_bounds = np.concatenate([np.floor(index_bounds[:2] / 2), np.ceil(index_bounds[2:4] / 2)]).astype(int)
                    index_bounds = upper_level_index_bounds * 2
                    #print(index_bounds)
                    img = self._read_array_region(lower_layer[1], (index_bounds[0],index_bounds[2]), (index_bounds[1],index_bounds[3]))
                    #print(img.shape)
                    img = img.astype(self.next_bigger_dtype)
                    img_low_res = ((img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]) / 4).astype(self.dtype)
                    self._write_array_region(db, (upper_level_index_bounds[0],upper_level_index_bounds[2]), (upper_level_index_bounds[1],upper_level_index_bounds[3]), data=img_low_res)
                    #db[upper_level_index_bounds[0]:upper_level_index_bounds[2], upper_level_index_bounds[1]:upper_level_index_bounds[3]] = img_low_res.view(self.band_attribute_dtype)
                    new_dirty_regions.append(upper_level_index_bounds)
                dirty_regions = new_dirty_regions
        self.dirty_regions = []

    def get_data(self, bounds: Tuple[float, float, float, float], resolution: Optional[Tuple[int,int]]=None) -> np.ndarray:
        indexes = self._bounds_to_indexes(bounds)
        dims = np.array([indexes[2] - indexes[0], indexes[3] - indexes[1]])
        if resolution is None:
            layer = 0
        else:
            layer = np.clip(math.floor(math.log2((dims / np.array(resolution)).min())), 0, self.num_pyramid_layers)
        indexes = (indexes / 2**layer).astype(int)
        t1 = time.time()
        layer = self.layers[layer]
        if layer[1] is None:
            layer[1] = tiledb.DenseArray(layer[0], mode='r')
        res = self._read_array_region(layer[1], (indexes[0],indexes[2]),(indexes[1],indexes[3]))
        #print("db request took: {}".format(time.time() - t1))
        return res     

    def consolidate_and_vacuum(self):
        uris = []
        for layer in self.layers:
            if layer[1] is not None:
                layer[1].close()
            uris.append(layer[0])
        if self._consolidation_thread is None or not self._consolidation_thread.is_alive():
            self._consolidation_thread = Thread(target = consolidate_and_vacuume, args=(uris,))
            self._consolidation_thread.start()
    
    def invalidate_cache(self):
        for layer in self.layers:
            if layer[1] is not None:
                layer[1].close()
                layer[1] = None
    
    def delete_permanently(self):
        for layer in self.layers:
            if layer[1] is not None:
                layer[1].close()
        shutil.rmtree(self.directory_path)


