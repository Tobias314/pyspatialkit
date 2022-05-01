from typing import Optional, List, TYPE_CHECKING
import numpy as np
import sys
import json

from ...utils.numpy import to_endianess
if TYPE_CHECKING:
    from ...dataobjects.geopointcloud import GeoPointCloud
from ...crs.geocrstransformer import GeoCrsTransformer
from ...processing.pointcloud.normals import compute_normals_from_xyz
from ...globals import TILE3D_CRS

PNTS_VERSION = 1
PNTS_VERSION_HEADER_FIELD = PNTS_VERSION.to_bytes(4, 'little')


# TODO implement full 3dtiles pnts specification (batch_ids, batch_table ...)
# TODO create a copy of the point cloud with the relevant attributes only and do crs transformation on this copy using the
#       default to_crs() method previnting code duplication and reimplementation to crs transformation here
def geopointcloud_to_3dtiles_pnts(pcl: 'GeoPointCloud', crs_transformer: Optional[GeoCrsTransformer] = None, rgb: bool = True,
                                  normals: bool = False) -> bytes:
    do_crs_transform = TILE3D_CRS != pcl.crs
    if do_crs_transform:
        if crs_transformer is None:
            crs_transformer = GeoCrsTransformer(pcl.crs, TILE3D_CRS)
        else:
            if crs_transformer.to_crs != TILE3D_CRS:
                raise ValueError('Transformer must transform to:' + str(TILE3D_CRS))
    magic = b'pnts'
    version = PNTS_VERSION_HEADER_FIELD
    batch_table_json_byte_length = (0).to_bytes(4, 'little')
    batch_table_binary_byte_length =(0).to_bytes(4, 'little')
    num_pts = pcl.shape[0]
    if do_crs_transform:
        if normals:
            normal_array = xyz_array + pcl.normals_xyz.to_numpy()
        xyz_array = np.stack(crs_transformer.transform(pcl.x, pcl.y, pcl.z), axis=1)
        if normals:
            normal_array = np.stack(crs_transformer.transform(normal_array[:,0], normal_array[:,1], normal_array[:,2]), axis=1)
            normal_array = normal_array - xyz_array
            normal_array = normal_array / np.linalg.norm(normal_array, axis=1)[:, np.newaxis]
    else:
        xyz_array = pcl.xyz.to_numpy()
        if normals:
            normal_array = pcl.normals.to_numpy()
    center = np.array(crs_transformer.transform(*pcl.center))
    feature_table_json = {
        'POINTS_LENGTH': num_pts,
        'POSITION': {'byteOffset': 0},
        'RTC_CENTER': list(center),
    }
    offset = num_pts * 12
    arrays: List[np.ndarray] = []
    xyz_array = (xyz_array - center).astype(np.float32)
    arrays.append(xyz_array.flatten())
    if normals:
        feature_table_json['NORMAL'] = {'byteOffset': offset}
        arrays.append(normal_array.astype(np.float32).flatten())
        offset += num_pts * 12
    if rgb:
        if not pcl.has_rgb:
            z = pcl.z.to_numpy()
            brightness = 100 + (np.abs(z) % 50) * 3
            brightness = brightness.astype(np.uint8)
            rgb_array = np.repeat(z[:,np.newaxis], repeats=3, axis=1)
        else:
            rgb_array = pcl.rgb.to_numpy()
            if pcl.rgb_max != 255:
                rgb_array = 255 * rgb_array / pcl.rgb_max
        feature_table_json['RGB'] = {'byteOffset': offset}
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
