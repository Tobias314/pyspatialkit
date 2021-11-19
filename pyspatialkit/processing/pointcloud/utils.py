from typing import Optional, Union, Tuple

import numpy as np


def project_points_to_plane(points: np.ndarray, plane_equation: np.ndarray):
    pts_w = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    dists = pts_w @ plane_equation
    normal = plane_equation[:-1]
    return points - dists[:, np.newaxis] * normal[np.newaxis, :]

def points3d_to_image(xyz: np.ndarray, pixel_size: float,
                      values: Union[int,float,np.ndarray] = 1, empty_value=0, up_axis: int = 1,
                      ufunc: Optional[np.ufunc] = None) -> Tuple[np.ndarray, np.ndarray]:
    plane_axis = [0, 1, 2]
    del plane_axis[up_axis]
    s_indices = xyz[:, (plane_axis[0], plane_axis[1])]
    min_corner = s_indices.min(axis=0)
    min_corner = min_corner // pixel_size
    image_origin = min_corner * pixel_size
    s_indices = ((s_indices - np.array(image_origin)) / pixel_size).astype(int)
    max_corner = s_indices.max(axis=0)
    image = np.full((max_corner + 1).astype(int), empty_value)
    if ufunc is None:
        image[s_indices[:, 0], s_indices[:, 1]] = values
    else:
        ufunc.at(image, (s_indices[:, 0], s_indices[:, 1]), values)
    return image, image_origin