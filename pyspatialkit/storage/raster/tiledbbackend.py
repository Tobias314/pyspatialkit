from typing import Tuple, Union
from pathlib import Path
import math

import numpy as np
import tiledb

class TileDbBackend:

    def __init__(self, bounds: Tuple[float, float, float, float], num_bands: int, dtype: np.dtype, directory_path: Path,
                 pixel_size: Tuple[float, float] = (1,1), tile_size: Tuple[int, int] = (1000, 1000), fill_value = 0,
                 build_pyramid: bool = True, num_pyramid_layers: Union[int, str] = 'all') -> None:
        self.bounds = np.array(bounds)
        self.dims = np.array((bounds[2] - bounds[0] / pixel_size[0], bounds[3] - bounds[1] / pixel_size[1]))
        self.pixel_size = np.array(pixel_size)
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
        self.layers = [None,] * (self.num_pyramid_layers + 1)
        self.num_bands = num_bands
        self.dtype = dtype
        self.band_attribute_dtype = np.dtype(','.join([np.dtype(self.dtype).str] * self.num_bands))
        band_attribute = tiledb.Attr(dtype=self.band_attribute_dtype, fill=fill_value)
        for i, layer in enumerate(self.layers):
            path = str(self.directory_path / ("layer_" + str(i)))
            if not Path(path).exists():
                dim = np.ceil(self.dims / 2**i).astype(int)
                tile_size = np.stack([np.ceil(dim / 2), self.tile_size], axis=1).min(axis=1)
                max32 = np.iinfo(np.int32).max
                dtypes = (np.int32 if self.dims[0] > max32 else np.int64,  np.int32 if self.dims[1] > max32 else np.int64)
                dimx = tiledb.Dim(name="dimx" + str(i), domain=(0, dim[0]), tile=tile_size[0], dtype=dtypes[0])
                dimy = tiledb.Dim(name="dimy" + str(i), domain=(0, dim[1]), tile=tile_size[1], dtype=dtypes[1])
                dom = tiledb.Domain(dimx, dimy)
                schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=[band_attribute,])
                schema.check()
                tiledb.Array.create(path, schema)
            self.layers[i] = [path, tiledb.DenseArray(path, mode='r')]
        if self.build_pyramid:
            self.dirty_bounds = []

    def _bounds_to_indices(self, bounds: Tuple[float, float, float, float]) -> np.ndarray:
        bounds = np.array(bounds)
        bounds[:2] = (bounds[:2] - self.bounds[:2]) / self.pixel_size
        bounds[2:4] = (bounds[2:4] - self.bounds[:2]) / self.pixel_size
        return bounds

    def write_data(self, bounds: Tuple[float, float, float, float], data: np.ndarray) -> None:
        assert data.shape[-1] == self.num_bands
        assert data.dtype == self.dtype
        data = data.squeeze()
        bounds = self._bounds_to_indices(bounds)
        path = self.layers[0][0]
        with tiledb.DenseArray(path, mode='w') as db:
            d = data.view(self.band_attribute_dtype)
            db[bounds[0]:bounds[2], bounds[1]:bounds[3]] = d
            self.layers[0][1] = tiledb.DenseArray(path, mode='r')
            if self.build_pyramid:
                self.dirty_bounds.append(bounds)

    def update_pyramid(self) -> None:
        dirty_bounds = self.dirty_bounds
        for layer in range(1, self.num_pyramid_layers+1):
            write_db_path = self.layers[layer][0]
            new_dirty_bounds = []
            with tiledb.DenseArray(write_db_path, mode='w') as db:
                for bounds in dirty_bounds:
                    upper_level_bounds = np.concatenate([np.floor(bounds[:2] / 2), np.ceil(bounds[2:4] / 2)]).astype(int)
                    bounds = upper_level_bounds * 2
                    img = self.layers[layer-1][1][bounds[0]:bounds[2], bounds[1]:bounds[3]]
                    img = img.view((self.dtype, self.num_bands))
                    img_low_res = (img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]) / 4
                    db[upper_level_bounds[0]:upper_level_bounds[2], upper_level_bounds[1]:upper_level_bounds[3]] = img_low_res.view(self.band_attribute_dtype)
                    self.layers[layer][1] = tiledb.DenseArray(write_db_path, mode='r')
                    new_dirty_bounds.append(upper_level_bounds)
                dirty_bounds = new_dirty_bounds

    def get_data(self, bounds: Tuple[float, float, float, float], resolution: Tuple[int, int]) -> np.ndarray:
        bounds = self._bounds_to_indices(bounds)
        dims = np.array([bounds[2] - bounds[0], bounds[3] - bounds[1]])
        layer = math.floor(math.log2((dims / np.array(resolution)).min()))
        bounds = (bounds / 2**layer).astype(int)
        db = self.layers[layer][1]
        res = db[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        res = res.view((self.dtype, self.num_bands))
        return res

    def close_all_db_connections(self):
        for layer in self.layers:
            layer[1].close()

