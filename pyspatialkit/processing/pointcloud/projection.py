from typing import Optional, Union, Tuple

import numpy as np


def project_points_to_plane(points: np.ndarray, plane_equation: np.ndarray):
    pts_w = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    dists = pts_w @ plane_equation
    normal = plane_equation[:-1]
    return points - dists[:, np.newaxis] * normal[np.newaxis, :]

def project_down(s_indices: np.ndarray, values: Union[int,float,np.ndarray],
                 img: np.ndarray, check_bounds: bool = False, ufunc: Optional[np.ufunc] = None, 
                 return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    i1, i2 = img.shape[0] - 1 - s_indices[:, 1], s_indices[:, 0]
    if check_bounds:
        bounds_mask = (i1 >= 0) & (i1< img.shape[0]) & (i2 >= 0) & (i2< img.shape[1])
        i1 = i1[bounds_mask]
        i2 = i2[bounds_mask]
        values = values[bounds_mask]
    if ufunc is None:
        img[i1, i2] = values
    else:
        ufunc.at(img, (i1, i2), values)
    if not return_mask:
        return img, None
    else:
        mask = np.full(img.shape[:2], False)
        mask[i1, i2] = True
        return img, mask

def project_to_image(xyz: np.ndarray, image: np.ndarray, transform: Optional[np.ndarray] = None,
                     values: Optional[Union[int,float,np.ndarray]] = None, up_axis: int = 1,
                     ufunc: Optional[np.ufunc] = None,
                     return_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    plane_axis = [0, 1, 2]
    del plane_axis[up_axis]
    s_indices = xyz[:, (plane_axis[0], plane_axis[1])]
    s_indices = np.concatenate([s_indices, np.ones((s_indices.shape[0], 1))], axis=1)
    s_indices = (s_indices @ transform.T).astype(int)
    if values is None:
        values = xyz[:, 2]
    if len(values.shape) == 1:
        values = values[:, np.newaxis]
    return project_down(s_indices=s_indices, values=values, img=image, check_bounds=True, ufunc=ufunc, return_mask=return_mask)
    

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
    if values is None:
        values = xyz[:, 2]
    if len(values.shape) == 1:
        values = values[:, np.newaxis]
    img, mask = project_down(s_indices=s_indices, values=values, img=image, check_bounds=False, ufunc=ufunc, return_mask=return_mask)
    if return_mask:
        return img, image_origin, mask
    else:
        return img, image_origin