from typing import Optional, Tuple
from .codequality import deprecated

import numpy as np
from skimage.transform import warp
from matplotlib import pyplot as plt

def _create_mask(img: np.ndarray, no_data_value: float, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is not None:
        assert img.shape[0]==mask.shape[0] and img.shape[1] == mask.shape[1]
        if no_data_value is not None:
                mask = mask & img == no_data_value
    elif no_data_value is not None:
        mask = img == no_data_value
    else:
        mask = None
    return mask


#TODO: do profiling to optimize performence because those functions might be critical for the overall performance


def project_skimage(img_in: np.ndarray, img_out: np.ndarray, inv_mat: np.ndarray,
                     in_no_data_value: Optional[float] = None, in_mask: Optional[np.ndarray] = None,
                     out_no_data_value: Optional[float] = None, out_mask: np.ndarray = None) -> None:
    assert img_in.shape[2] == img_out.shape[2]
    assert img_in.dtype == img_out.dtype
    warped = warp(img_in, inv_mat, output_shape=img_out.shape[:2], cval=np.nan, preserve_range=True)
    nan_mask = np.isnan(warped)
    warped[nan_mask] = img_out[nan_mask]
    if in_no_data_value is not None or in_mask is not None or out_no_data_value is not None or out_mask is not None:
        in_mask = _create_mask(img_in, in_no_data_value, in_mask)
        out_mask = _create_mask(img_out, out_no_data_value, out_mask)
        if in_mask is not None:
            in_mask = warp(in_mask, inv_mat, output_shape=img_out.shape[:2], preserve_range=True).astype(bool)
            if out_mask is not None:
                mask = in_mask & np.bitwise_not(out_mask)
        else:
            mask = np.bitwise_not(out_mask)
        warped[mask] = img_out[mask]
    img_out[:,:] = warped


#Deprecated!!!
@deprecated
def project_numpy(img_in: np.ndarray, img_out: np.ndarray, inv_mat: np.ndarray,
                   in_no_data_value: Optional[float] = None, in_mask: Optional[np.ndarray] = None,
                   out_no_data_value: Optional[float] = None, out_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        assert img_in.shape[2] == img_out.shape[2]
        assert img_in.dtype == img_out.dtype
        a_raster = img_out
        b_raster = img_in
        a_mask, a_no_data_value = out_mask, out_no_data_value
        b_mask, b_no_data_value = in_mask, in_no_data_value
        flipped = False
        if img_in.shape[0] * img_in.shape[1] < img_out.shape[0] * img_out.shape[1]:
            a_raster, b_raster = b_raster, a_raster
            a_no_data_value, b_no_data_value = b_no_data_value, a_no_data_value
            b_mask, a_mask = a_mask, b_mask
            flipped = True
        coords = np.stack(np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing='ij'))
        if a_mask is not None and b_no_data_value is not None:
            coords = coords[:, a_mask & a_raster.data == b_no_data_value]
        elif a_mask is not None:
            coords = coords[:, a_mask]
        elif b_no_data_value is not None:
            coords = coords[:, a_raster == b_no_data_value]
        else:
            coords = coords.reshape((2, -1))
        eps = np.finfo(np.float32).eps # we will add a small epsilon to prevent off by one errors caused by rounding
        if not flipped:
            inv_mat = np.linalg.inv(inv_mat)
        projected_coords = inv_mat @ np.concatenate([coords.astype(float) + eps, np.ones([1, coords.shape[1]])])
        projected_coords = (projected_coords[:2] / projected_coords[2]).astype(int)
        projected_coords_mask = (projected_coords[0] >= 0) & (projected_coords[0] < b_raster.shape[0]) & \
                                 (projected_coords[1] >= 0) & (projected_coords[1] < b_raster.shape[1])
        if b_mask is not None and b_no_data_value is not None:
            projected_coords_mask = b_mask[projected_coords[0], projected_coords[1]] \
                & (b_raster.data == b_no_data_value)[projected_coords[0], projected_coords[1]]
        elif b_mask is not None:
            projected_coords_mask = b_mask[projected_coords[0], projected_coords[1]]
        elif b_no_data_value is not None:
            projected_coords_mask = (b_raster.data == b_no_data_value)[projected_coords[0], projected_coords[1]]
        coords = coords[:, projected_coords_mask]
        projected_coords = projected_coords[:, projected_coords_mask]
        if not flipped:
            projected_coords, coords = coords, projected_coords
        img_in[projected_coords[0], projected_coords[1]] = img_out[coords[0], coords[1]]
        return projected_coords, coords    