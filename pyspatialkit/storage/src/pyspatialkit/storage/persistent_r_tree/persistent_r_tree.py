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

class RTreeNode:
    def __init__(self, tree: 'PersistentRTree', id: int, objects: List[BBoxSerializable] = []):
        self.tree = tree
        self.id = id
        #self.child_node_ids = child_node_ids
        self.objects = objects

    def get_child_nodes(self)->List['RTreeNode']:
        return self.tree.get_child_nodes(parent_node_id=self.id)

class PersistentRTree():

    def __init__(self, tree_path: str, object_type: Type[BBoxSerializable], dimensions: int = 3,
                 data_path: Optional[str]=None, tree_name: str = 'tree.sqlite'):
        self.object_typ = object_type
        if tree_path == ':memory:':
            self.engine = sa.create_engine(f"sqlite+pysqlite:///:memory:", echo=True)
        else:
            self.engine = sa.create_engine(f"sqlite+pysqlite:///{tree_path}/{tree_name}", echo=True)
        if data_path is None:
            if tree_path == ':memory:':
                self.data_fs = open_fs('mem://').makedirs('data')
            else:
                self.data_fs = open_fs(tree_path).makedir('data')
        else:
            self.data_fs = open_fs(data_path)
        self.dimensions = dimensions    
        self.axis_names = []
        for d in range(dimensions):
            c = chr(d+65)
            self.axis_names.append(f'min{c}')
            self.axis_names.append(f'max{c}')
        if not sa.inspect(self.engine).has_table(R_TREE_TABLE_NAME):
            self._initialize_r_tree()

    def __getitem__(self, indexer) -> List[BBoxSerializable]:
        if not isinstance(indexer, tuple):
            indexer = (indexer,)
        starts = []
        stops = []
        for ind in indexer:
            if not isinstance(ind, slice):
                starts.append(ind)
                stops.append(ind)
            else:
                starts.append(ind.start)
                stops.append(ind.stop)
        return self.query_bbox(np.array(starts + stops))

    def query_bbox(self, bbox: np.ndarray) -> List[BBoxSerializable]: #TODO
        indices = [self.axis_names[d] for d in range(0, len(self.axis_names), 2)]
        indices += [self.axis_names[d] for d in range(1, len(self.axis_names), 2)]
        query = f"SELECT {','.join(indices)},{OBJECT_ID_COLUMN_NAME} FROM {R_TREE_TABLE_NAME} WHERE "
        constraints = []
        for d in range(self.dimensions):
            constraints.append(f'{self.axis_names[2*d]}<={bbox[self.dimensions + d]}')
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

    def get_root_node_id(self)->int:
        return 1

    def get_root_node(self):
        return self.get_node(node_id=self.get_root_node_id())

    def get_node(self, node_id: int)->RTreeNode:
        raise NotImplementedError()
        
    def get_node(self, node_id: int) -> RTreeNode:
        sql = f"SELECT data FROM {R_TREE_TABLE_NAME}_node WHERE nodeno={node_id}"
        with self.engine.connect() as con:
            res = [row.data for row in con.execute(sa.text(sql))]
            assert len(res)==1
            data = res[0]
            sql = f"SELECT nodeno FROM {R_TREE_TABLE_NAME}_parent WHERE parentnode = {int(node_id)}"
            child_node_ids = [row.nodeno for row in con.execute(sa.text(sql))]
            num_entries = int.from_bytes(data[2:4], 'big')
            entry_size = 8 + 4 * self.dimensions * 2
            data = data[4 : 4 + entry_size * num_entries]
            buffer = np.frombuffer(data, dtype=np.uint8)
            buffer = buffer.reshape((num_entries, entry_size))
            dt_float32 = np.dtype(np.float32).newbyteorder('>')
            bounds = np.frombuffer(buffer[:,8:].tobytes(), dtype=dt_float32).reshape((num_entries, self.dimensions * 2))
            dt_int64 = np.dtype(np.int64).newbyteorder('>')
            ids = np.frombuffer(buffer[:, :8].tobytes(), dtype=dt_int64)
        #TODO
        raise NotImplementedError()
        
    def _initialize_r_tree(self):
        sql = f"CREATE VIRTUAL TABLE {R_TREE_TABLE_NAME} USING rtree(\n id"
        for d in range(0,self.dimensions*2,2):
            sql += f",\n {self.axis_names[d]}, {self.axis_names[d+1]}"
        sql += f",\n +{OBJECT_ID_COLUMN_NAME} TEXT);"
        sql2 = f'\n CREATE INDEX "parent_index" ON "{R_TREE_TABLE_NAME}_parent" ("parentnode")'
        with self.engine.begin() as connection:
            connection.execute(sa.text(sql))
            connection.execute(sa.text(sql2))