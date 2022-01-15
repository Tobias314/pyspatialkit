from typing import List
import math

import numpy as np

def merge_plane_equations(equations: List[np.ndarray], angle_thresh_deg = 8, distance_thresh=0.5):
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