from typing import Optional, List, TYPE_CHECKING
import numpy as np
import sys
import json

from ...utils.numpy import to_endianess
print("TYPECHECKING" + str(TYPE_CHECKING))
if TYPE_CHECKING:
    from ...dataobjects.geopointcloud import GeoPointCloud

PNTS_VERSION = 1
PNTS_VERSION_HEADER_FIELD = to_endianess(np.array([PNTS_VERSION], dtype=np.uint32), '<').tobytes()


# TODO implement full 3dtiles pnts specification (batch_ids, batch_table ...)
def geopointcloud_to_3dtiles_pnts(pcl: 'GeoPointCloud', rgb: bool = True, normals: bool = False) -> bytes:
    magic = b'pnts'
    version = PNTS_VERSION_HEADER_FIELD
    batch_table_json_byte_length = np.array([0], dtype=np.uint32)
    batch_table_binary_byte_length = np.array([0], dtype=np.uint32)
    num_pts = pcl.shape[0]
    feature_table_json = {
        'POINTS_LENGTH': num_pts,
        'POSITION': {'byteOffset': 0},
    }
    offset = num_pts * 12
    center = pcl.center
    arrays: List[np.ndarray] = []
    xyz_array = (pcl.xyz.to_numpy() - center).astype(np.float32)
    arrays.append(xyz_array.flatten())
    if rgb:
        feature_table_json['RGB'] = {'byteOffset': offset}
        rgb_array = pcl.rgb.to_numpy()
        if pcl.rgb_max != 255:
            rgb_array = 255 * rgb_array / pcl.rgb_max
        rgb_array = rgb_array.astype(np.uint8)
        arrays.append(rgb_array.flatten())
        offset += num_pts * 3
    if normals:
        feature_table_json['NORMAL'] = {'byteOffset': offset}
        normal_array = pcl.normals_xyz.to_numpy().astype(np.float32)
        arrays.append(normal_array.flatten())
        offset += num_pts * 12
    feature_table_json = json.dumps(feature_table_json)
    feature_table_json = bytes(feature_table_json, 'utf-8')
    if sys.byteorder == 'big':
        for arr in arrays:
            arr.byteswap()
    feature_table = b''
    for arr in arrays:
        feature_table += arr.tobytes()
    byte_length = np.array([28 + len(feature_table_json) + len(feature_table)], dtype='<u32').tobytes()
    feature_table_json_byte_length = np.array([len(feature_table_json)], dtype='<u32').tobytes()
    feature_table_binary_byte_length = np.array([len(feature_table)], dtype='<u32').tobytes()
    res = magic + version + byte_length + feature_table_json_byte_length + feature_table_binary_byte_length + \
           batch_table_json_byte_length + batch_table_binary_byte_length + feature_table_json + feature_table
    return res
