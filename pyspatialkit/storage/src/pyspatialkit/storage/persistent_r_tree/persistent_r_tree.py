from typing import Optional, List, Tuple

import sqlalchemy as sa

from pyspatialkit.core.interfaces import BBoxSerializable

R_TREE_TABLE_NAME = "r_tree"

class PersistentRTree():

    def __init__(self, path: str, dimensions: int = 3):
        self.path = path
        self.dimensions = dimensions
        self.engine = sa.create_engine(f"sqlite+pysqlite:///{self.path}", echo=True)
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
            self.query_range(range)
        else:
            self.query_point(indexer)

    def query_range(self, range: List[Tuple[float, float]]) -> List[BBoxSerializable]: #TODO
        raise NotImplementedError()

    def query_point(self, point: List[float]) -> List[BBoxSerializable]: #TODO
        raise NotImplementedError()

    def _initialize_r_tree(self):
        sql = f"""CREATE VIRTUAL TABLE {R_TREE_TABLE_NAME} USING rtree(
                    id"""
        for d in range(self.dimensions):
            c = chr(d+65)
            sql += f"""
            ,min{c}, max{c}"""
        sql += ');'
        with self.engine.begin() as connection:
            connection.execute(sa.text(sql))