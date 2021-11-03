import numpy as np


def projective_transform_from_pts(source_pts: np.ndarray, destination_pts: np.ndarray):
    """
        Math taken from: https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript
    """
    def get_basis_transform(pts: np.ndarray):
        transp = pts.T
        a = np.concatenate([transp[:,:3], np.ones((1,3))])
        b = np.concatenate([transp[:, 3], [1]])
        return a * np.linalg.solve(a, b)
    return np.linalg.inv(get_basis_transform(source_pts)) @ get_basis_transform(destination_pts)