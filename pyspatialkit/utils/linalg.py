import numpy as np


def projective_transform_from_pts(source_pts: np.ndarray, destination_pts: np.ndarray) -> np.ndarray:
    """
        taken from: https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript
    """
    def get_basis_transform(pts: np.ndarray):
        transp = pts.T
        a = np.concatenate([transp[:,:3], np.ones((1,3))])
        b = np.concatenate([transp[:, 3], [1]])
        return a * np.linalg.solve(a, b)
    return get_basis_transform(destination_pts) @ np.linalg.inv(get_basis_transform(source_pts))

def affine_transform_from_pts(source_pts: np.ndarray, destination_pts: np.ndarray) -> np.ndarray:
    a = np.zeros((6,6))
    for i in range(3):
        a[2*i,:2] = source_pts[i]
        a[2*i, 2] = 1
        a[2*i+1, 3:] = source_pts[i]
        a[2*i+1, 5] = 1
    b = destination_pts.reshape(6)
    last_row = np.zeros(3)
    last_row[2] = 1
    return np.concatenate([np.linalg.solve(a,b).reshape((2,3)), last_row[:, np.newaxis]], axis=1)