import argparse
import json
import requests
from pathlib import Path
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from typing import Optional
import geopandas as gpd

API_URL_RGBI100 = "https://www.lvermgeo.sachsen-anhalt.de/de/mod/3,1816,501/ajax/1/prepare/?items={}&format=zip"
API_URL_DOM2 = "https://www.lvermgeo.sachsen-anhalt.de/de/mod/2,1824,501/ajax/1/prepare/?items={}&format=zip"
API_URL_DGM2 = "https://www.lvermgeo.sachsen-anhalt.de/de/mod/2,1817,501/ajax/1/prepare/?items={}&format=zip"
API_URL_MAP = {
    'rgbi100': API_URL_RGBI100,
    'dom2': API_URL_DOM2,
    'dgm2': API_URL_DGM2
}

def retrieve_files(url: str, output_dir: Path):
    response = requests.get(url)
    download_url = response.text
    zip_response = requests.get(download_url)
    zf = zipfile.ZipFile(io.BytesIO(zip_response.content))
    for zip_item in zf.filelist:
        if zip_item.filename[-4:] != '.pdf':
            zf.extract(zip_item.filename, path=output_dir)


def download_geodata(data_type:str, chunk_size:int = 5, output_dir: Optional[str] = None, aoi: Optional[gpd.GeoDataFrame] = None,
                     num_threads: int = 1):
    tiles_geojson_path = 'tiles_{}.json'.format(data_type)
    geojson =  gpd.GeoDataFrame.from_file(tiles_geojson_path)
    if output_dir is None:
        output_dir = './{}/'.format(data_type)
    api_url = API_URL_MAP[data_type]
    if aoi is not None:
        aoi = aoi.to_crs(geojson.crs)
        geojson = geojson.overlay(aoi, how='intersection')
    else:
        geojson.rename(columns={'id':'id_1'}, inplace=True)
    boxes = set()
    ids = []
    for index, tile in geojson.iterrows():
        bbox = tile.geometry.bounds
        if bbox not in boxes:
            ids.append(tile.id_1)
            boxes.add(bbox)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(num_threads) as executor:
        futures = []
        for i in range(0,len(ids), chunk_size):
            ids_string = ','.join(ids[i:i+chunk_size])
            url = api_url.format(ids_string)
            print(url)
            futures.append(executor.submit(retrieve_files, url, output_dir))
        for future in tqdm(futures):
            future.result()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Download data from geodata portal of saxony anhalt')
    parser.add_argument('data_type', type=str, default='rgbi100', choices=['rgbi100', 'dgm2', 'dom2'],
                         help='Data type to choose e.g rgbi100 for rgbi images with resolution 100cm')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to write the results to')
    parser.add_argument('--aoi', type=str, default=None, help='Shapefile path with area of interest polygon')
    parser.add_argument('--chunk_size', type=int, default=5, help='How many tiles to fetch per request')
    parser.add_argument('--num_threads', type=int, default=1, help='Num threads to use for parallel requests')
    args = parser.parse_args()
    aoi = args.aoi
    if aoi is not None:
        aoi = gpd.GeoDataFrame.from_file(aoi)
    download_geodata(data_type=args.data_type, chunk_size=args.chunk_size, output_dir=args.output_dir,
                     aoi=aoi, num_threads=args.num_threads)