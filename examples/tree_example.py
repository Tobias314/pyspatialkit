import sys
sys.path.append('../tools/')

from geodownloader.geodownloader_saxony_anhalt import download_geodata

def main():
    aoi = gpd.GeoDataFrame.from_file('example_aoi.shp')
    directory_path = Path('tree_detection_data/')
    download_geodata(data_type='dom', num_threads=5, output_dir=directory_path /'dom', aoi=aoi)
    download_geodata(data_type='dgm', num_threads=5, output_dir=directory_path /'dgm', aoi=aoi)
    download_geodata(data_type=args.data_type, num_threads=5, output_dir=directory_path /'rgbi',aoi=aoi)