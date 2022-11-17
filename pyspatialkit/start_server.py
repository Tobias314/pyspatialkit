import argparse

from pyspatialkit.server import start_server
from pyspatialkit.storage import GeoStorage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts the visualization server for a give geostorage')
    parser.add_argument('geostorage_path', type=str, help='geostorage path')
    args = parser.parse_args()
    print('loading GeoStorage from: {}'.format(args.geostorage_path))
    geostorage = GeoStorage(args.geostorage_path)
    print('Serving GeoStorage with layers: {}'.format(list(geostorage.layers.keys())))
    start_server(geostorage)