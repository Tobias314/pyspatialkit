from typing import Optional, List, TYPE_CHECKING
import numpy as np
import sys
import json

from ...utils.numpy import to_endianess
if TYPE_CHECKING:
    from ...dataobjects.geopointcloud import GeoPointCloud

PNTS_VERSION = 1
PNTS_VERSION_HEADER_FIELD = to_endianess(np.array([PNTS_VERSION], dtype=np.uint32), '<').tobytes()


# TODO implement full 3dtiles pnts specification (batch_ids, batch_table ...)
def geopointcloud_to_3dtiles_pnts(pcl: 'GeoPointCloud', rgb: bool = True, normals: bool = False) -> bytes:
    magic = b'pnts'
    version = PNTS_VERSION_HEADER_FIELD
    batch_table_json_byte_length = np.array([0], dtype='<u4').tobytes()
    batch_table_binary_byte_length = np.array([0], dtype='<u4').tobytes()
    num_pts = pcl.shape[0]
    center = pcl.center
    feature_table_json = {
        'POINTS_LENGTH': num_pts,
        'POSITION': {'byteOffset': 0},
        'RTC_CENTER': list(center),
    }
    offset = num_pts * 12
    arrays: List[np.ndarray] = []
    xyz_array = (pcl.xyz.to_numpy() - center).astype(np.float32)
    arrays.append(xyz_array.flatten())
    if normals:
        feature_table_json['NORMAL'] = {'byteOffset': offset}
        normal_array = pcl.normals_xyz.to_numpy().astype(np.float32)
        arrays.append(normal_array.flatten())
        offset += num_pts * 12
    if rgb:
        if not pcl.has_rgb:
            z = pcl.z.to_numpy()
            brightness = 100 + (np.abs(z) % 50) * 3
            brightness = brightness.astype(np.uint8)
            rgb = np.repeat(z[:,np.newaxis], repeats=3, axis=1)
            pcl.rgb = rgb
        feature_table_json['RGB'] = {'byteOffset': offset}
        rgb_array = pcl.rgb.to_numpy()
        if pcl.rgb_max != 255:
            rgb_array = 255 * rgb_array / pcl.rgb_max
        rgb_array = rgb_array.astype(np.uint8)
        arrays.append(rgb_array.flatten())
        offset += num_pts * 3
    feature_table_json = json.dumps(feature_table_json)
    feature_table_json_padding = (8 - (28 + len(feature_table_json)) % 8) % 8
    feature_table_json = bytes(feature_table_json + ' ' * feature_table_json_padding, 'utf-8') 
    if sys.byteorder == 'big':
        for arr in arrays:
            arr.byteswap()
    feature_table = b''
    for arr in arrays:
        feature_table += arr.tobytes()
    feature_table_padding = (8 - len(feature_table) % 8) % 8
    feature_table += b' ' * feature_table_padding
    byte_length = np.array([28 + len(feature_table_json) + len(feature_table)], dtype='<u4').tobytes()
    feature_table_json_byte_length = np.array([len(feature_table_json)], dtype='<u4').tobytes()
    feature_table_binary_byte_length = np.array([len(feature_table)], dtype='<u4').tobytes()
    res = magic + version + byte_length + feature_table_json_byte_length + feature_table_binary_byte_length + \
           batch_table_json_byte_length + batch_table_binary_byte_length + feature_table_json + feature_table
    return res
