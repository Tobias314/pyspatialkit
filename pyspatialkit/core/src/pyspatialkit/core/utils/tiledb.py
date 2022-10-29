from typing import List

import tiledb
from .logging import dbg


def consolidate_and_vacuume(uris: List[str]):
    dbg('start consolidating')
    for uri in uris:
        dbg('consoldating tiledb array at {} ...'.format(uri))
        tiledb.consolidate(uri)
        tiledb.vacuum(uri)
    dbg('done consolidating')