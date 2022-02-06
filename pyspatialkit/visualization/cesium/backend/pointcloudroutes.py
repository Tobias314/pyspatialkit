from logging import exception
from pathlib import Path
import os
from io import BytesIO
import time
import re
from typing import Tuple

import numpy as np
from fastapi import APIRouter, Response, Request
from fastapi.responses import StreamingResponse, FileResponse
from genshi.template import MarkupTemplate
from pyproj import CRS as PyprojCRS
import imageio
import cv2 as cv

from ....crs.geocrs import GeoCrs
from ....spacedescriptors.georect import GeoRect
from ....storage.geostorage import GeoStorage
from ....dataobjects.tiles3d.pointcloud.geopointcloudtile3d import GeoPointCloudTileIdentifier
from ....dataobjects.tiles3d.pointcloud.geopointcloudtile3d import GeoPointCloudTile3d
from ....dataobjects.tiles3d.tiles3dcontentobject import TILES3D_CONTENT_TYPE_TO_FILE_ENDING
from ....dataobjects.tiles3d.tile3d import Tile3d
from ....dataobjects.tiles3d.tileset3d import Tileset3d
from ....storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer


FILE_PATTERN = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)\..+')
CONTENT_ENDING = TILES3D_CONTENT_TYPE_TO_FILE_ENDING['POINT_CLOUD']

router = APIRouter(
    prefix='/backend',
    tags=["pointcloud"],
    responses={404: {"description": "Not found"}},
)


def _uri_generator(tile: GeoPointCloudTile3d) -> str:
    level = tile.identifier.level
    indices = tile.identifier.tile_indices
    return str("{}_{}_{}_{}.json".format(level, indices[0], indices[1], indices[2]))

def _content_uri_generator(tile: GeoPointCloudTile3d) -> str:
    tile_uri = _uri_generator(tile)
    return 'content/' + tile_uri + CONTENT_ENDING

def _get_tileset(request: Request, layer_name: str) -> Tileset3d:
    geostorage: GeoStorage = request.app.geostorage
    layer: GeoPointCloudLayer = geostorage.get_layer(layer_name)
    return layer.visualizer_tileset

def _get_tileset_and_tile(request: Request, layer_name: str, tile_descriptor: str) -> Tuple[Tileset3d, Tile3d]:
    m = [int(val) for val in re.search(FILE_PATTERN, tile_descriptor).groups()]
    tile_identifier = GeoPointCloudTileIdentifier(m[0], m[1:4])
    tileset = _get_tileset(request, layer_name)
    tile = tileset.get_tile_by_identifier(tile_identifier)
    return tileset, tile

@router.get("/{layer}/tiles/content/{tile_descriptor}")
async def get_point_cloud_root_tile(request: Request, layer: str, tile_descriptor: str):
    _, tile = _get_tileset_and_tile(request, layer, tile_descriptor)
    serialized_bytes = tile.content.to_bytes_tiles3d()
    return StreamingResponse(BytesIO(serialized_bytes), media_type="application/octet-stream")

@router.get("/{layer}/tiles/{tile_descriptor}")
async def get_point_cloud_content(request: Request, layer: str, tile_descriptor: str):
    if tile_descriptor == 'root.json':
        tileset = _get_tileset(request, layer)
        tile = tileset.get_root()
    else:
        tileset, tile = _get_tileset_and_tile(request, layer, tile_descriptor)
    json_dict, _ = tileset.materialize(tile_uri_generator=_uri_generator, tile_content_uri_generator=_content_uri_generator,
                                       root_tile=tile, max_depth=1)
    return json_dict