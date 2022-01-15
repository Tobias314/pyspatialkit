import random
from typing import Optional

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from scipy.linalg import lstsq
from sklearn.decomposition import PCA

from ...dataobjects.geopointcloud import GeoPointCloud


def normal_seeded_ransac(cloud: GeoPointCloud, dist_thresh=0.2, inner_iterations=100,
                         normal_criterion=lambda normal: True, return_clouds=False, return_equations=False,
                         return_point_ids=False,
                         min_inliers=1000, connected_components_dist: Optional[float] = None,
                         with_removal=True, lstsq_postprocessing=True):
    assert inner_iterations > 0
    plane_ids = np.ones(cloud.shape[0], dtype=int) * -1
    min_inliers = int(min_inliers)
    if not cloud.has_normals:
        cloud.estimate_normals()
    planes = []
    xyz_full = cloud.xyz.to_numpy()
    xyzd_full = np.concatenate([xyz_full, np.ones([xyz_full.shape[0], 1])], axis=1)
    normals_full = cloud.normals_xyz.to_numpy()
    point_ids_full = np.arange(cloud.shape[0])
    remaining_ids = point_ids_full.copy()
    iteration = 0
    current_id = 0
    while remaining_ids.shape[0] > 3 and remaining_ids.shape[0] > min_inliers:
        # print("iteration {}".format(iteration))
        iteration += 1
        xyz = xyz_full[remaining_ids]
        xyzd = xyzd_full[remaining_ids]
        normals = normals_full[remaining_ids]
        max_inliers = 0
        plane_mask = None
        plane_mask_full = None
        best_equation = None
        for i in range(inner_iterations):
            idx = random.randint(0, xyz.shape[0] - 1)
            p0 = xyz[idx]
            normal = normals[idx]
            d = -1 * np.dot(normal, p0)
            equation = np.array([normal[0], normal[1], normal[2], d])
            if not normal_criterion(normal):
                continue
            #w = xyz - p0
            dist = np.abs(xyzd @ equation)
            inliers = dist < dist_thresh
            num_inliers = inliers.sum()
            if not with_removal:
                num_inliers = (np.abs(xyzd_full @ equation) < dist_thresh).sum()
            if lstsq_postprocessing and num_inliers > 3: #and num_inliers > min_inliers/4:
                inlier_pts = xyz[inliers, :]
                pca = PCA(n_components=3)
                pca.fit(inlier_pts)
                normal = pca.components_[2]
                p0 = inlier_pts.mean(axis=0)
                d = -1 * np.dot(normal, p0)
                equation = np.array([normal[0], normal[1], normal[2], d])
                dist = np.abs(xyzd @ equation)
                inliers = dist < dist_thresh
                num_inliers = inliers.sum()
            if not with_removal:
                inliers_full = np.abs(xyzd_full @ equation) < dist_thresh
                num_inliers = (inliers_full).sum()
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                plane_mask = inliers
                if not with_removal:
                    plane_mask_full = inliers_full
                best_equation = equation
        if connected_components_dist is not None:
            # clustering = OPTICS(cluster_method='dbscan', eps=connected_components_dist,
            #                    min_cluster_size=min_inliers).fit(xyz[plane_mask])
            # labels = clustering.labels_
            if with_removal:
                xyz_o3d = o3d.utility.Vector3dVector(xyz[plane_mask])
            else:
                xyz_o3d = o3d.utility.Vector3dVector(xyz[plane_mask_full])
            pc3d = o3d.geometry.PointCloud()
            pc3d.points = xyz_o3d
            labels = np.array(pc3d.cluster_dbscan(eps=connected_components_dist, min_points=min_inliers))
            max_label = labels.max()
            # print(max_label)
            if max_label < 0:
                break
            for i in range(max_label + 1):
                mask = plane_mask.copy()
                mask[plane_mask] = labels == i
                if with_removal:
                    current_plane_ids = remaining_ids[mask]
                else:
                    current_plane_ids = point_ids_full[plane_mask_full]
                plane_ids[current_plane_ids] = current_id
                current_id += 1
                res = []
                if return_clouds:
                    res.append(cloud[current_plane_ids])
                if return_equations:
                    res.append(best_equation)
                if return_point_ids:
                    res.append(current_plane_ids)
                if res:
                    if len(res) > 1:
                        planes.append(tuple(res))
                    else:
                        planes.append(res[0])

            mask = plane_mask.copy()
            mask[plane_mask] = labels != -1
            remaining_ids = remaining_ids[np.logical_not(mask)]
        else:
            if max_inliers < min_inliers:
                break
            if with_removal:
                current_plane_ids = remaining_ids[plane_mask]
            else:
                current_plane_ids = point_ids_full[plane_mask_full]
            plane_ids[current_plane_ids] = current_id
            current_id += 1
            res = []
            if return_clouds:
                res.append(cloud[current_plane_ids])
            if return_equations:
                res.append(best_equation)
            if return_point_ids:
                res.append(current_plane_ids)
            if res:
                if len(res) > 1:
                    planes.append(tuple(res))
                else:
                    planes.append(res[0])
            remaining_ids = remaining_ids[np.logical_not(plane_mask)]
    return plane_ids, planes



def normal_seeded_ransac_old(cloud: GeoPointCloud, dist_thresh=0.2, inner_iterations=100,
                        normal_criterion=lambda normal: True, return_clouds=False, return_equations=False,
                        return_point_ids=False,
                        min_inliers=1000, connected_components_dist: Optional[float] = None,
                        with_removal=True, lstsq_postprocessing=True):
    assert inner_iterations > 0
    plane_ids = np.ones(cloud.shape[0], dtype=int) * -1
    min_inliers = int(min_inliers)
    if not cloud.has_normals:
        cloud.estimate_normals()
    planes = []
    xyz_full = cloud.xyz.to_numpy()
    xyzd_full = np.concatenate([xyz_full, np.ones([xyz_full.shape[0], 1])], axis=1)
    normals_full = cloud.normals_xyz.to_numpy()
    point_ids_full = np.arange(cloud.shape[0])
    remaining_ids = point_ids_full.copy()
    # remaining = cloud.copy()
    iteration = 0
    current_id = 0
    # xyz = cloud.xyz.to_numpy()
    while (remaining_ids.shape[0] > 3):
        # print("iteration {}".format(iteration))
        iteration += 1
        xyz = xyz_full[remaining_ids]
        xyzd = xyzd_full[remaining_ids]
        normals = normals_full[remaining_ids]
        max_inliers = 0
        plane_mask = None
        plane_mask_full = None
        best_equation = None
        for i in range(inner_iterations):
            idx = random.randint(0, xyz.shape[0] - 1)
            p0 = xyz[idx]
            normal = normals[idx]
            d = -1 * np.dot(normal, p0)
            equation = np.array([normal[0], normal[1], normal[2], d])
            if not normal_criterion(normal):
                continue
            #w = xyz - p0
            dist = np.abs(xyzd @ equation)
            inliers = dist < dist_thresh
            num_inliers = inliers.sum()
            if not with_removal:
                num_inliers = (np.abs(xyzd_full @ equation) < dist_thresh).sum()
            if lstsq_postprocessing and num_inliers > 3 and num_inliers > min_inliers/4:
                inlier_pts = xyz[inliers, :]
                pca = PCA(n_components=3)
                pca.fit(inlier_pts)
                normal = pca.components_[2]
                p0 = inlier_pts.mean(axis=0)
                d = -1 * np.dot(normal, p0)
                equation = np.array([normal[0], normal[1], normal[2], d])
                dist = np.abs(xyzd @ equation)
                inliers = dist < dist_thresh
            num_inliers = inliers.sum()
            if not with_removal:
                inliers_full = np.abs(xyzd_full @ equation) < dist_thresh
                num_inliers = (inliers_full).sum()
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                plane_mask = inliers
                if not with_removal:
                    plane_mask_full = inliers_full
                best_equation = equation
        if connected_components_dist is not None:
            # clustering = OPTICS(cluster_method='dbscan', eps=connected_components_dist,
            #                    min_cluster_size=min_inliers).fit(xyz[plane_mask])
            # labels = clustering.labels_
            if with_removal:
                xyz_o3d = o3d.utility.Vector3dVector(xyz[plane_mask])
            else:
                xyz_o3d = o3d.utility.Vector3dVector(xyz[plane_mask_full])
            pc3d = o3d.geometry.PointCloud()
            pc3d.points = xyz_o3d
            labels = np.array(pc3d.cluster_dbscan(eps=connected_components_dist, min_points=min_inliers))
            max_label = labels.max()
            # print(max_label)
            if max_label < 0:
                break
            for i in range(max_label + 1):
                mask = plane_mask.copy()
                mask[plane_mask] = labels == i
                if with_removal:
                    current_plane_ids = remaining_ids[mask]
                else:
                    current_plane_ids = point_ids_full[plane_mask_full]
                plane_ids[current_plane_ids] = current_id
                current_id += 1
                res = []
                if return_clouds:
                    res.append(cloud[current_plane_ids])
                if return_equations:
                    res.append(best_equation)
                if return_point_ids:
                    res.append(current_plane_ids)
                if res:
                    if len(res) > 1:
                        planes.append(tuple(res))
                    else:
                        planes.append(res[0])

            mask = plane_mask.copy()
            mask[plane_mask] = labels != -1
            remaining_ids = remaining_ids[np.logical_not(mask)]
        else:
            if max_inliers < min_inliers:
                break
            if with_removal:
                current_plane_ids = remaining_ids[plane_mask]
            else:
                current_plane_ids = point_ids_full[plane_mask_full]
            plane_ids[current_plane_ids] = current_id
            current_id += 1
            res = []
            if return_clouds:
                res.append(cloud[current_plane_ids])
            if return_equations:
                res.append(best_equation)
            if return_point_ids:
                res.append(current_plane_ids)
            if res:
                if len(res) > 1:
                    planes.append(tuple(res))
                else:
                    planes.append(res[0])
            remaining_ids = remaining_ids[np.logical_not(plane_mask)]
    return plane_ids, planes