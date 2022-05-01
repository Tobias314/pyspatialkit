import numpy as np
import numpy.typing as npt
import open3d as o3d

def compute_normals_from_xyz(xyz_array: npt.NDArray[float], knn_k: int = 100, fast_normal_computation:bool = True) -> npt.NDArray[float]:
    xyz_o3d = o3d.utility.Vector3dVector(xyz_array)
    pc3d = o3d.geometry.PointCloud()
    pc3d.points = xyz_o3d
    pc3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn_k), fast_normal_computation)
    return np.array(pc3d.normals)