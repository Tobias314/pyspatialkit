from typing import Optional, List, Tuple, Type
from pathlib import Path
import uuid

import sqlalchemy as sa
from fs import open_fs
import numpy as np
import pandas as pd

from pyspatialkit.core.interfaces import BBoxSerializable

R_TREE_TABLE_NAME = "r_tree"
OBJECT_ID_COLUMN_NAME = 'object_id'
DATA_FILE_ENDING = '.data'

class PersistentRTree():

    def __init__(self, root_path: str, object_type: Type[BBoxSerializable], dimensions: int = 3):
        self.root_path = root_path
        self.object_typ = object_type
        if root_path == ':memory:':
            self.engine = sa.create_engine(f"sqlite+pysqlite:///{root_path}", echo=True)
            self.data_fs = open_fs('mem://').makedirs('data')
        else:
            self.engine = sa.create_engine(f"sqlite+pysqlite:///{root_path}/tree.sqlite", echo=True)
            self.data_fs = open_fs(root_path)
            if self.data_fs.isdir('data'):
                self.data_fs = self.data_fs.opendir('data')
            else:
                self.data_fs = self.data_fs.makedirs('data')
        self.dimensions = dimensions
        self.axis_names = []
        for d in range(self.dimensions):
            c = chr(d+65)
            self.axis_names.append(f'min{c}')
            self.axis_names.append(f'max{c}')
        if not sa.inspect(self.engine).has_table(R_TREE_TABLE_NAME):
            self._initialize_r_tree()

    def __getitem__(self, indexer) -> List[BBoxSerializable]:
        #TODO, fix multidimension
        if isinstance(indexer, (slice, tuple)):
            if isinstance(indexer, slice):
                indexer = (indexer,)
            range = []
            for ind in indexer:
                range.append((ind.start, ind.stop))
            self.query_bbox(range)
        else:
            self.query_point(indexer)

    def query_bbox(self, bbox: np.ndarray) -> List[BBoxSerializable]: #TODO
        indices = [self.axis_names[d] for d in range(0, len(self.axis_names), 2)]
        indices += [self.axis_names[d] for d in range(1, len(self.axis_names), 2)]
        query = f"SELECT {','.join(indices)},{OBJECT_ID_COLUMN_NAME} FROM {R_TREE_TABLE_NAME} WHERE "
        constraints = []
        for d in range(self.dimensions):
            constraints.append(f'{self.axis_names[2*d]}<={bbox[self.dimensions+d]}')
            constraints.append(f'{self.axis_names[2*d+1]}>={bbox[d]}')
        query += ' AND '.join(constraints)
        df = pd.read_sql(sql=query, con=self.engine)
        bboxes = df.loc[:, indices].to_numpy()
        results = []
        for i, object_id in enumerate(df[OBJECT_ID_COLUMN_NAME]):
            with self.data_fs.open(object_id + DATA_FILE_ENDING, mode='rb') as f:
                data = f.read()
            results.append(self.object_typ.from_bytes(data=data, bbox=bboxes[i]))
        return results

    def query_point(self, point: List[float]) -> List[BBoxSerializable]: #TODO
        raise NotImplementedError()

    def insert(self, objects: List[BBoxSerializable]):
        bounds_df = np.stack(list(map(lambda obj: obj.get_bounds(), objects)), axis=0)
        #reshuffle bbox coordinates to get into sqlite r-tree format (minX,maxX,minY,maxY...)
        indices = np.stack([np.arange(self.dimensions), np.arange(start=self.dimensions, stop= 2 * self.dimensions)], axis=1)
        indices = indices.flatten()
        bounds_df = bounds_df[:, indices]
        bounds_df = pd.DataFrame(data=bounds_df, columns=self.axis_names)
        uuids = [str(uuid.uuid4()) for i in range(bounds_df.shape[0])]
        bounds_df[OBJECT_ID_COLUMN_NAME] = uuids
        for i,obj in enumerate(objects):
            data = obj.to_bytes()
            with self.data_fs.open(f'{uuids[i]}.data', mode='wb') as f:
                f.write(data)
        bounds_df.to_sql(name=R_TREE_TABLE_NAME, con=self.engine, if_exists='append', index=False)
        

    def _initialize_r_tree(self):
        sql = f"CREATE VIRTUAL TABLE {R_TREE_TABLE_NAME} USING rtree(\n id"
        for d in range(0,self.dimensions*2,2):
            sql += f",\n {self.axis_names[d]}, {self.axis_names[d+1]}"
        sql += f",\n +{OBJECT_ID_COLUMN_NAME} TEXT);"
        with self.engine.begin() as connection:
            connection.execute(sa.text(sql))