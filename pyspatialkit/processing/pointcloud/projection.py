from typing import Optional, Union, Tuple

import numpy as np


def project_points_to_plane(points: np.ndarray, plane_equation: np.ndarray):
    pts_w = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    dists = pts_w @ plane_equation
    normal = plane_equation[:-1]
    return points - dists[:, np.newaxis] * normal[np.newaxis, :]

def points3d_to_image(xyz: np.ndarray, pixel_size: float,
                      values: Optional[Union[int,float,np.ndarray]] = None, empty_value=0, up_axis: int = 1,
                      ufunc: Optional[np.ufunc] = None,
                      return_mask: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    plane_axis = [0, 1, 2]
    del plane_axis[up_axis]
    s_indices = xyz[:, (plane_axis[0], plane_axis[1])]
    min_corner = s_indices.min(axis=0)
    min_corner = min_corner // pixel_size
    image_origin = min_corner * pixel_size
    s_indices = ((s_indices - np.array(image_origin)) / pixel_size).astype(int)
    max_corner = s_indices.max(axis=0)
    img_size = np.flip((max_corner + 1).astype(int),0)
    if values is None:
        image = np.full(img_size, empty_value, dtype=xyz.dtype)
    elif isinstance(values, np.ndarray):
        image = np.full(img_size, empty_value, dtype=values.dtype)
    else:
        image = np.full(img_size, empty_value, dtype=type(values))
    i1, i2 = img_size.shape[0] - 1 - s_indices[:, 1], s_indices[:, 0]
    if ufunc is None:
        image[i1, i2] = values
    else:
        if values is None:
            ufunc.at(image, (i1, i2), xyz[:, 2])
        else:
            ufunc.at(image, (i1, i2), values)
    if not return_mask:
        return image[:,:,np.newaxis], image_origin
    else:
        mask = np.full(img_size, False)
        mask[i1, i2] = True
        return image[:,:,np.newaxis], image_origin, mask