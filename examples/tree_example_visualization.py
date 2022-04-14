from pathlib import Path

from pyspatialkit.storage.geostorage import GeoStorage
from pyspatialkit.visualization.cesium.backend.server import start_server


def main():
    dir_path = Path('tree_detection_data/geostore/')
    if not dir_path.is_dir():
        print("No GeoStorage directory found!")
        return
    storage = GeoStorage(dir_path)
    start_server(storage, port=8181)        

if __name__=='__main__':
    main()