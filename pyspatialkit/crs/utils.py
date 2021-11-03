from pyproj import CRS, Transformer

from .geocrs import GeoCRS


def crs_bounds(crs: GeoCRS):
    bounds = crs.proj_crs.area_of_use.bounds
    minimum = bounds[:2]
    maximum = bounds[2:]
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs.proj_crs, always_xy=True)
    minimum = transformer.transform(*minimum)
    maximum = transformer.transform(*maximum)
    return [*minimum, *maximum]