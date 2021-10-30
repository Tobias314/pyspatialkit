from pathlib import Path
import os
import json

from ..utils.logging import logger

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

    def delete_layer(self, layer_name: str):
        if layer_name not in self.layers:
            logger().warning("A layer with this name does not exist. No layer will be deleted")
            return False
        dir_path = self.layers[layer_name].folder_path
        del self.layers[layer_name]
        force_delete_dir(dir_path)
        return True

    def _add_layer(self, layer_name: str, layer_type: type, *args, **kwargs):
        if layer_name in self.layers:
            print("A layer with name {} already exists, returning existing layer.".format(layer_name))
            return self.get_layer(layer_name)
        layer = layer_type(os.path.join(self.dir_path, layer_name), *args, **kwargs)
        self.layers[layer_name] = layer
        self.persist_configuration()
        return layer

    def add_raster_layer(self, raster_layer_name: str) -> GeoRasterLayer:
        #TODO
        raise NotImplementedError