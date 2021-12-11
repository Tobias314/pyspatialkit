from pathlib import Path
import os
import json
from typing import Optional, Tuple
import shutil

import numpy as np

from .geolayer import GeoLayer

from ..utils.logging import logger
from ..utils.fileio import force_delete_directory
from .raster.georasterlayer import GeoRasterLayer
from ..crs.geocrs import GeoCrs
from ..globals import DEFAULT_CRS

class GeoStorage:

    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)
        self.directory_path.mkdir(parents=True, exist_ok=True)
        self.layers = {}
        if os.path.exists(self.directory_path / ".config"):
            with open(self.directory_path / ".config") as config_file:
                data = config_file.read()
                data = json.loads(data)
                for x in data["layers"]:
                    layer_dir = self.directory_path / x["name"] / ""
                    self.layers[os.path.basename(layer_dir)] = globals()[x["type"]](layer_dir)

    @property
    def name(self):
        return self.directory_path.name

    def persist_configuration(self):
        layers = []
        for name, layer in self.layers.items():
            layers.append({"name": name,
                           "type": str(type(layer).__name__)})
        configuration = {"layers": layers}
        with open(self.directory_path / ".config", 'w') as outfile:
            json.dump(configuration, outfile)

    def get_layer(self, layer_name: str):
        if layer_name not in self.layers:
            logger().warning("A layer with this name does not exist, returning None")
            return None
        return self.layers[layer_name]

    def delete(self):
        for layer in self.layers.values():
            layer.delete()
        shutil.rmtree(self.directory_path)

    def delete_layer(self, layer_name: str):
        if layer_name not in self.layers:
            logger().warning("A layer with this name does not exist. No layer will be deleted")
            return False
        dir_path = self.layers[layer_name].folder_path
        del self.layers[layer_name]
        force_delete_directory(dir_path)
        return True

    def _add_layer(self, layer_name: str, layer_type: type, *args, **kwargs) -> GeoLayer:
        if layer_name in self.layers:
            print("A layer with name {} already exists, returning existing layer.".format(layer_name))
            return self.get_layer(layer_name)
        layer = layer_type(directory_path = os.path.join(self.directory_path, layer_name), *args, **kwargs)
        self.layers[layer_name] = layer
        self.persist_configuration()
        return layer

    def add_raster_layer(self, layer_name: str, num_bands:int, dtype: np.dtype, crs: GeoCrs = DEFAULT_CRS,
                         bounds: Optional[Tuple[float, float, float, float]] = None, fill_value = 0,
                         pixel_size_xy: Tuple[float, float] = (1,1,), build_pyramid:bool=True) -> GeoRasterLayer:
        return self._add_layer(layer_name=layer_name, layer_type=GeoRasterLayer, num_bands=num_bands, dtype=dtype, crs=crs,
                                bounds=bounds, fill_value=fill_value,pixel_size_xy=pixel_size_xy, build_pyramid=build_pyramid)

    def plot_cesium(self):
        from ..visualization.cesium.backend.server import start_server #We do it here to avoid cyclic dependencies
        start_server(self)

