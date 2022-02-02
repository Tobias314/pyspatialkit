from typing import Tuple, Optional

import numpy as np

def interpolate_holes(img: np.ndarray, mask) -> np.ndarray:
    normalizer_thresh = 1
    while(not mask.all()):
        img, mask = interpolate_along_axis(img, mask, normalizer_thresh=normalizer_thresh)
        normalizer_thresh /= 2
    return img

def interpolate_along_axis(img: np.ndarray, mask: np.ndarray, normalizer_thresh: Optional[float] = None)-> Tuple[np.ndarray, np.ndarray]:
    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    mask_i = np.logical_not(mask)
    y_m = y.copy()
    x_m = x.copy()
    y_m[mask_i] = -1
    x_m[mask_i] = -1
    left = np.maximum.accumulate(x_m, axis=1)
    top = np.maximum.accumulate(y_m, axis=0)
    y_m[mask_i] = mask_i.shape[0]
    x_m[mask_i] = mask_i.shape[1]
    x_m = np.flip(x_m, axis=1)
    y_m = np.flip(y_m, axis=0)
    right = np.flip(np.minimum.accumulate(x_m, axis=1), axis=1)
    down = np.flip(np.minimum.accumulate(y_m, axis=0), axis=0)
    sums = np.zeros(img.shape)
    sums[mask] = img[mask]
    normalizers = np.zeros(img.shape)

    m = (left!=-1) & mask_i
    left = left[m]
    div = (x[m] - left + 1)
    div = 1 / div[:, np.newaxis]
    sums[m] += img[y[m], left] * div
    normalizers[m] += div

    m = (right!=img.shape[1]) & mask_i
    right = right[m]
    div = (right - x[m] + 1)
    div = 1 / div[:, np.newaxis]
    sums[m] += img[y[m],right] * div
    normalizers[m] += div

    m = (top!=-1) & mask_i
    top = top[m]
    div = (y[m] - top + 1)
    div = 1 / div[:, np.newaxis]
    sums[m] += img[top, x[m]] * div
    normalizers[m] += div

    m = (down!=img.shape[0]) & mask_i
    down = down[m]
    div = (down - y[m] + 1)
    div = 1 / div[:, np.newaxis]
    sums[m] += img[down, x[m]] * div
    normalizers[m] += div

    if normalizer_thresh is None:
        m = normalizers != 0
    else:
        m = normalizers >= normalizer_thresh
    m = m.squeeze() 
    sums[m] /= normalizers[m]
    return sums, m | mask


##div = 1 / np.repeat(div[:, np.newaxis], 1, axis=1)