from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from pyproj import CRS, transformer

from ..crs.geocrs import GeoCRS, NoneCRS


class GeoRect:

    def __init__(self, top_left: Tuple[float, float],  bottom_right: Tuple[float, float],
                 top_right: Optional[Tuple[float, float]] = None, bottom_left: Optional[Tuple[float, float]] = None,
                 crs: GeoCRS = NoneCRS):
        self.crs = crs
        self.top_left = top_left
        self.bottom_right = bottom_right
        if top_right is None or bottom_left is None:
            self.top_right = (bottom_right[0], top_left[1])
            self.bottom_left = (top_left[0], bottom_right[1])
        else:
            self.top_right = top_right
            self.bottom_left = bottom_left


class GeoRectOld:

    #TODO: think about other constructor and class methods for constructing different GeoRects
    def __init__(self, center_pos: Tuple[float, float], width: float, height: float, crs, rotation_angle=0):
        self.center_pos = [center_pos[0], center_pos[1]]
        self.crs = crs
        bottom_left = [self.center_pos[0] - width / 2, self.center_pos[1] - height / 2]
        top_right = [self.center_pos[0] + width / 2, self.center_pos[1] + height / 2]
        bottom_right = [top_right[0], bottom_left[1]]
        top_left = [bottom_left[0], top_right[1]]
        self.polygon = Polygon([bottom_left, bottom_right, top_right, top_left])
        self.rotation_angle = rotation_angle
        self.polygon = affinity.rotate(self.polygon, self.rotation_angle)
        self._width_meters = None
        self._height_meters = None
        self.crs = crs
        self._world_rect_transform = None

    @classmethod
    def from_bounds(cls, bounds: List[float], crs: CRS):
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        center_pos = (bounds[0] + width / 2, bounds[1] + height / 2)
        return cls(center_pos, width, height, crs)


    @classmethod
    def from_center_width_height_up(cls, center_pos: Tuple[float, float], width: float, height: float, up: Tuple[float, float],
                                    crs: CRS):
        angle = math.atan2(up[1], up[0]) - math.atan2(1, 0)
        angle = math.degrees(angle)
        return cls(center_pos, width, height, crs, angle)

    def to_crs(self, new_crs: CRS = None, crs_transformer: CrsTransformer = None, inplace=True):
        if crs_transformer is not None:
            assert crs_transformer.crs_from.equals(self.crs)
            transformer = crs_transformer.transformer
            new_crs = crs_transformer.crs_to
        else:
            if new_crs is None or self.crs.equals(new_crs):
                if inplace:
                    return
                else:
                    return self.copy()
            transformer = Transformer.from_crs(self.crs, new_crs, always_xy=True)
        if inplace:
            rect = self
            self._world_rect_transform = None
        else:
            rect = self.copy()
        rect.polygon = transform(transformer.transform, self.polygon)
        rect.center_pos = transformer.transform(rect.center_pos[0], rect.center_pos[1])
        rect.crs = new_crs
        if not inplace:
            return rect

    def get_bounds(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.bounds

    #TODO: get_bounds and get_extent do the same thing, merge them
    def get_extent(self, target_crs=None):
        self.to_crs(target_crs)
        bounds = self.get_bounds()
        return [(bounds[0], bounds[1]), (bounds[2], bounds[3])]

    def get_bottom_left(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.exterior.coords[0]

    def get_top_right(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.exterior.coords[2]

    def get_bottom_right(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.exterior.coords[1]

    def get_top_left(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.exterior.coords[3]

    def get_width(self, target_crs=None):
        self.to_crs(target_crs)
        return np.linalg.norm(np.array(self.get_bottom_left()) - np.array(self.get_bottom_right()))

    def get_width_meters(self):
        if self._width_meters is None:
            self._width_meters = point_distance_meters(self.get_bottom_left(), self.get_bottom_right(), self.crs)
        return self._width_meters

    def get_height_meters(self):
        if self._height_meters is None:
            self._height_meters = point_distance_meters(self.get_bottom_left(), self.get_top_left(), self.crs)
        return self._height_meters

    def get_height(self, target_crs=None):
        self.to_crs(target_crs)
        return np.linalg.norm(np.array(self.get_bottom_left()) - np.array(self.get_top_left()))

    def get_corners(self, target_crs=None):
        self.to_crs(target_crs)
        return self.polygon.exterior.coords[:-1]

    def get_index_corners_on_raster(self, raster: rio.io.DatasetReader):
        corners = self.get_corners(target_crs=raster.crs)
        indexed_corners = []
        for corner in corners:
            indexed_corners.append(raster.index(*corner))
        return indexed_corners

    def get_dimensions_on_raster(self, raster: rio.io.DatasetReader):
        pixel_corners = self.get_index_corners_on_raster(raster)
        pixel_width = math.sqrt(
            (pixel_corners[1][0] - pixel_corners[0][0]) ** 2 + (pixel_corners[1][1] - pixel_corners[0][1]) ** 2)
        pixel_height = math.sqrt(
            (pixel_corners[0][0] - pixel_corners[3][0]) ** 2 + (pixel_corners[0][1] - pixel_corners[3][1]) ** 2)
        return pixel_width, pixel_height

    def get_polygon(self, target_crs: CRS = None):
        copy = self.to_crs(target_crs, inplace=False)
        return copy.polygon

    def get_matplotlib_path(self):
        return Path(np.array(self.get_corners()))

    def get_rotation_angle(self):
        return self.rotation_angle

    def intersects(self, other: 'GeoRect') -> bool:
        if not self.crs == other.crs:
           other = other.to_crs(self.crs, inplace=False)
        return self.polygon.intersects(other.polygon)

    def subdivide(self, num_subdivisions) -> List['GeoRect']:
        min_x = self.get_bounds()[0]
        min_y = self.get_bounds()[1]
        num_subdivisions += 1
        lx = self.get_width() / num_subdivisions
        ly = self.get_height() / num_subdivisions
        result = []
        for i in range(num_subdivisions):
            for j in range(num_subdivisions):
                result.append(GeoRect.from_bounds([min_x, min_y, min_x + lx, min_y + ly], crs=self.crs))
                min_x += lx
            min_y += ly
        return result


    def to_file(self, file_path: Union[str, Path], target_crs: CRS = None):
        poly = self.get_polygon(target_crs=target_crs)
        crs = self.crs if target_crs is None else target_crs
        write_shape_to_file(poly, crs, file_path)

    def get_corner_coordinates_relative_to_extent(self):
        return self.world_to_rect_relative(self.get_corners())

    def get_world_to_rect_transform(self):
        if self._world_rect_transform is not None:
            return self._world_rect_transform
        src_pts = np.array(self.get_corners())
        dst_pts = np.array([(0,0), (1,0), (1,1), (0,1)])
        self._world_rect_transform = estimate_transform('projective', src_pts, dst_pts)
        return self._world_rect_transform

    def world_to_rect_relative(self, point: Union[Tuple[float, float], Sequence[Tuple[float, float]]]):
        transform = self.get_world_to_rect_transform()
        if isinstance(point, tuple) and len(point) == 2:
            return transform(point)
        else:
            return transform(np.array(point))

    #TODO: there might be a better way without recalculating width, height, but probably it's not that much of an issue
    def copy(self):
        width = self.get_width()
        height = self.get_height()
        return GeoRect(center_pos=self.center_pos, width=width, height=height, crs=self.crs, rotation_angle=self.rotation_angle)

    def __str__(self):
        res = ""
        for c in self.get_corners():
            res = res + str(c) + ','
        return res