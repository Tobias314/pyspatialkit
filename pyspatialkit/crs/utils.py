from pyproj import CRS, Transformer

from .geocrs import GeoCrs


def crs_bounds(crs: GeoCrs):
    if crs.proj_crs.area_of_use is None:
        raise AttributeError("Cannot compute bounds of given CRS")
    bounds = crs.proj_crs.area_of_use.bounds
    minimum = bounds[:2]
    maximum = bounds[2:]
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs.proj_crs, always_xy=True)
    minimum = transformer.transform(*minimum)
    maximum = transformer.transform(*maximum)
    return [*minimum, *maximum]