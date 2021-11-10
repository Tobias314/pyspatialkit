import argparse

from pyspatialkit.visualization.cesium.backend.server import start_server
from pyspatialkit.storage.geostorage import GeoStorage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts the visualization server for a give geostorage')
    parser.add_argument('geostorage_path', type=str, help='geostorage path')
    args = parser.parse_args()
    geostorage = GeoStorage(args.geostorage_path)
    start_server(geostorage)