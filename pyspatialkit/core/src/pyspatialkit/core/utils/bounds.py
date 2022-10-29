from typing import Tuple

def bounds3d_edge_lengths(bounds: Tuple[float, float, float, float, float, float]) -> float:
    return (bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])

def bounds3d_volume(bounds: Tuple[float, float, float, float, float, float]) -> float:
    edge_lengths = bounds3d_edge_lengths(bounds=bounds)
    return edge_lengths[0] * edge_lengths[1] * edge_lengths[2]

def bounds3d_half_surface_area(bounds: Tuple[float, float, float, float, float, float]) -> float:
    edge_length = bounds3d_edge_lengths(bounds=bounds)
    return edge_length[0] * edge_length[1] + edge_length[1] * edge_length[2] + edge_length[0] * edge_length[2]