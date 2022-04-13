from typing import Tuple, Union, Optional, Dict, List
from pathlib import Path
import math
import time
import shutil
from collections import OrderedDict
from threading import Thread

import numpy as np
from numpy.lib.shape_base import tile
import tiledb
import pandas as pd

from ...utils.numpy import next_bigger_dtype
from ...utils.bounds import bounds3d_half_surface_area, bounds3d_edge_lengths, bounds3d_volume
from ...utils.logging import dbg, raise_warning
from ...utils.tiledb import consolidate_and_vacuume

POINT_PYRAMID_REDUCTION_FACTOR = 5
AXIS_NAMES = ['x', 'y', 'z']
AXIS_NAMES_SET = set(AXIS_NAMES)


class TileDbSparseBackend:
    """Backend for storing point cloud data in (several) TileDB databases.

    The TileDB can be thought of as very big sparse array with [x,y,z] indexing
    """

    def __init__(self, bounds: Tuple[float, float, float, float, float, float], directory_path: Path, data_scheme: Dict[str, np.dtype],
                 space_tile_size: Tuple[float, float] = (2, 2, 2), data_tile_capacity=1000,
                 build_pyramid: bool = True, base_point_density=0.01, num_pyramid_layers: Optional[int] = None) -> None:
        self.bounds = np.array(bounds)
        self.extent = self.bounds[3:] - self.bounds[:3]
        dbg("extent:" + str(self.extent))
        self.space_tile_size = np.array(space_tile_size)
        self.data_tile_capacity = data_tile_capacity
        if not (AXIS_NAMES[0] in data_scheme and AXIS_NAMES[1] in data_scheme and AXIS_NAMES[2] in data_scheme):
            raise ValueError("At least {},{},{} must be in the data scheme!".format(
                AXIS_NAMES[0], AXIS_NAMES[1], AXIS_NAMES[2]))
        self.data_scheme = data_scheme
        self.build_pyramid = build_pyramid
        if self.build_pyramid:
            max_pyramid_layers = int(np.log2(self.extent / np.array(self.space_tile_size)).max())
            if num_pyramid_layers is None:
                self.num_pyramid_layers = max_pyramid_layers
            else:
                self.num_pyramid_layers = min(num_pyramid_layers, max_pyramid_layers)
            self.base_point_density = base_point_density
        else:
            self.num_pyramid_layers = 0
        self.space_tile_size = np.array(space_tile_size)
        npts_dim1 = self.space_tile_size[0] / base_point_density
        npts_dim2 = self.space_tile_size[1] / base_point_density
        npts_dim3 = self.space_tile_size[2] / base_point_density
        self.num_points_base_tile_estimate: float = npts_dim1 * npts_dim2 + npts_dim1 * npts_dim3 + npts_dim2 * npts_dim3
        self.levels: List[tiledb.Array] = [None, ] * (self.num_pyramid_layers + 1)
        directory_path = Path(directory_path)
        assert(directory_path.is_dir)
        self.directory_path = directory_path
        if not directory_path.exists():
            directory_path.mkdir(parents=True)
            tiledb.group_create(str(self.directory_path))
        self.array_attributes = OrderedDict()
        for a, dt in self.data_scheme.items():
            if a not in AXIS_NAMES_SET:
                self.array_attributes[a] = tiledb.Attr(name=a, dtype=dt)
        if len(self.array_attributes) == 0:
            self._no_attributes = True
            self.array_attributes['t'] = tiledb.Attr(name='t', dtype=np.uint8)
        else:
            self._no_attributes = False
        for i, layer in enumerate(self.levels):
            path = str(self.directory_path / ("level_" + str(i)))
            if not Path(path).exists():
                dbg("size" + str(self.space_tile_size * 2**i))
                tile_extent = self.space_tile_size[0] * 2**i
                tile_extent = np.minimum(tile_extent, self.extent)
                dim1 = tiledb.Dim(name=AXIS_NAMES[0], domain=(self.bounds[0], self.bounds[3]), tile=tile_extent[0],
                                  dtype=self.data_scheme[AXIS_NAMES[0]])
                dim2 = tiledb.Dim(name=AXIS_NAMES[1], domain=(self.bounds[1], self.bounds[4]), tile=tile_extent[1],
                                  dtype=self.data_scheme[AXIS_NAMES[1]])
                dim3 = tiledb.Dim(name=AXIS_NAMES[2], domain=(self.bounds[2], self.bounds[5]), tile=tile_extent[2],
                                  dtype=self.data_scheme[AXIS_NAMES[2]])
                dom = tiledb.Domain(dim1, dim2, dim3)
                schema = tiledb.ArraySchema(
                    domain=dom, sparse=True, capacity=self.data_tile_capacity, attrs=self.array_attributes.values())
                schema.check()
                tiledb.Array.create(path, schema)
            self.levels[i] = [path, None]
        if self.build_pyramid:
            self.dirty_boxes = []
        self._consolidation_thread: Optional[Thread] = None

    def write_data(self, data: pd.DataFrame) -> None:
        columns = data.columns
        if not (AXIS_NAMES[0] in columns and AXIS_NAMES[1] in columns and AXIS_NAMES[2] in columns):
            raise ValueError("data needs {},{},{} columns".format(
                AXIS_NAMES[0], AXIS_NAMES[1], AXIS_NAMES[2]))
        attributes = {}
        for column in columns:
            if column not in AXIS_NAMES_SET:
                if column not in self.array_attributes:
                    raise ValueError(
                        'Attribute {} does not exist!'.format(column))
                attributes[column] = data[column].to_numpy()
        if self._no_attributes:
            attributes = np.full(data.shape[0], 1, dtype=np.uint8)
        if self.levels[0][1] is not None:
            self.levels[0][1].close()
        with tiledb.SparseArray(self.levels[0][0], mode='w') as db:
            x = data[AXIS_NAMES[0]].to_numpy()
            y = data[AXIS_NAMES[1]].to_numpy()
            z = data[AXIS_NAMES[2]].to_numpy()
            try:
                db[x, y, z] = attributes
            except tiledb.TileDBError as error:
                #TODO: make sure whether it has to be x<=bounds[3] ... or x<bounds[3]
                mask = (x>=self.bounds[0]) & (y>=self.bounds[1]) & (z>=self.bounds[2]) & (x<=self.bounds[3]) & (y<=self.bounds[4]) & (z<=self.bounds[5])
                if not mask.all():
                    raise_warning("Out of bounds write, writing only values inside of layer bounds...")
                    db[x[mask], y[mask], z[mask]] = attributes[mask]
                else:
                    raise error
            if self.build_pyramid:
                bounds = (x.min(), y.min(), z.min(), x.max(), y.max(), z.max())
                self.dirty_boxes.append(bounds)
        #self.levels[0][1] = tiledb.SparseArray(self.levels[0][0], mode='r')

    # TODO: merge bounds at every level to increase performance of batch updates
    def update_pyramid(self) -> None:
        print("UPDATING PYRAMIDS...")
        dirty_regions = self.dirty_boxes
        for level in range(1, self.num_pyramid_layers+1):
            write_db_path = self.levels[level][0]
            new_dirty_regions = []
            self.levels[level][1].close()
            with tiledb.SparseArray(write_db_path, mode='w') as db:
                lower_level = self.levels[level-1]
                if lower_level[1] is None:
                    lower_level[1] = tiledb.SparseArray(lower_level[0], mode='r')
                for bounds in dirty_regions:
                    df = lower_level[1].query(coords=True, use_arrow=False).df[bounds[0]:bounds[3],
                                                    bounds[1]:bounds[4], bounds[2]:bounds[5]]
                    area_estimate = bounds3d_half_surface_area(bounds)
                    density = (2**(level - 1) * self.base_point_density)
                    points_lower_estimate = area_estimate / density**2
                    points_upper_bound = bounds3d_volume(bounds) / density**3
                    num_points_target = int(np.clip(len(
                        df) / POINT_PYRAMID_REDUCTION_FACTOR, points_lower_estimate, points_upper_bound))
                    if num_points_target < len(df):
                        df = df.sample(n=num_points_target)
                    attributes = {}
                    for key in df.columns:
                        if key not in AXIS_NAMES_SET:
                            # TODO: to_numpy() might not be needed
                            attributes[key] = df[key].to_numpy()
                    #db[df[AXIS_NAMES[0]], df[AXIS_NAMES[1]],
                    #    df[AXIS_NAMES[2]]] = attributes
                    new_dirty_regions.append(bounds)
                dirty_regions = new_dirty_regions
            #self.levels[level][1] = tiledb.SparseArray(write_db_path, mode='r')
        self.dirty_boxes = []

    def get_data(self, bounds: Tuple[float, float, float, float, float, float], attributes: Optional[Tuple[str]] = None) -> pd.DataFrame:
        return self.get_data_for_level(bounds=bounds, level=0, attributes=attributes)

    # def get_data_with_count(self, bounds: Tuple[float, float, float, float, float, float], attributes: Optional[Tuple[str]] = None, min_num_points: Optional[int] = None) -> pd.DataFrame:
    #     if min_num_points is None:
    #         return self.get_data_for_level(bounds=bounds, level=0, attributes=attributes)
    #     area_estimate = bounds3d_half_surface_area(bounds)
    #     density_estimate = math.sqrt(area_estimate / min_num_points)
    #     level = np.clip(1 + math.floor(math.log(density_estimate /
    #                     self.base_point_density, 2)), 0, self.num_pyramid_layers)
    #     est_size = self.levels[level][1].query(
    #         return_incomplete=True).multi_index[bounds[0]:bounds[3], bounds[1]:bounds[4], bounds[2]:bounds[5]].est_result_size()
    #     if est_size < min_num_points:
    #         level += 1
    #         while level+1 <= self.num_pyramid_layers and self.levels[level][1].query(return_incomplete=True).multi_index[bounds[0]:bounds[3], bounds[1]:bounds[4], bounds[2]:bounds[5]].est_result_size() < min_num_points:
    #             level += 1
    #     elif est_size > min_num_points:
    #         while level-1 >= 0 and self.levels[level+1][1].query(return_incomplete=True).multi_index[bounds[0]:bounds[3], bounds[1]:bounds[4], bounds[2]:bounds[5]].est_result_size() > min_num_points:
    #             level -= 1
    #     return self.get_data_for_level(bounds=bounds, level=level, attributes=attributes)

    def get_data_for_level(self, bounds: Tuple[float, float, float, float, float, float], level: int, attributes: Optional[Tuple[str]] = None) -> pd.DataFrame:
        level = self.levels[level]
        if level[1] is None:
            level[1] = tiledb.SparseArray(level[0], mode='r')
        res = level[1].query(attrs=attributes, coords=True, use_arrow=False).df[bounds[0]:bounds[3], bounds[1]:bounds[4], bounds[2]:bounds[5]]
        if self._no_attributes:
            res.drop(columns='t', inplace=True)
        return res

    def invalidate_cache(self):
        for level in self.levels:
            if level[1] is not None:
                level[1].close()
                level[1] = None
                
    def consolidate_and_vacuum(self):
        uris = []
        for layer in levels:
            if layer[1] is not None:
                layer[1].close()
            uris.append(layer[0])
        if self._consolidation_thread is None or not self._consolidation_thread.is_alive():
            self._consolidation_thread = Thread(target = consolidate_and_vacuume, args=(uris,))
            self._consolidation_thread.start()            

    def delete_permanently(self):
        for level in self.levels:
            if level[1] is not None:
                level[1].close()
        shutil.rmtree(self.directory_path)
