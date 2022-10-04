from typing import List, Tuple
import math

import numpy as np

def merge_plane_equations(equations: List[np.ndarray], angle_thresh_deg = 8, distance_thresh=0.5)-> Tuple[List[np.ndarray], np.ndarray]:
    equations_copy = [eq for eq in equations]
    changes = True
    merged = [[i] for i, e in enumerate(equations_copy)]
    t = 0
    while changes:
        changes = False
        t += 1
        for i in range(len(equations_copy)):
            eq = equations_copy[i]
            if eq is None:
                continue
            for j in range(i, len(equations_copy)):
                if i == j or equations_copy[j] is None:
                    continue
                eq_b = equations_copy[j]
                angle = math.acos(np.dot(eq[:3], eq_b[:3]))
                negated_angle = math.acos(np.dot(-1 * eq[:3], eq_b[:3]))
                if negated_angle < angle:
                    angle = negated_angle
                    eq = -eq
                angle = math.degrees(angle)
                if angle < angle_thresh_deg and abs(eq[3] - eq_b[3]) < distance_thresh:
                    merged[i].extend(merged[j])
                    equations_copy[j] = None
                    changes = True
    new_class_ids = [None for i in equations]
    class_id = 0
    for i in range(len(new_class_ids)):
        if new_class_ids[i] is None:
            for c in merged[i]:
                new_class_ids[c] = class_id
            class_id += 1
    return [eq for eq in equations_copy if eq is not None], np.array(new_class_ids)


def plane_intersect(a, b):
    """
    taken from: https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  
    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
    aXb_vec = np.cross(a_vec, b_vec)
    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)
    p_inter = np.linalg.solve(A, d).T
    return p_inter[0], (p_inter + aXb_vec)[0]