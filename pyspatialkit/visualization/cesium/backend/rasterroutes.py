from logging import exception
from pathlib import Path
import os
from io import BytesIO
import time

import numpy as np
from fastapi import APIRouter, Response, Request
from fastapi.responses import StreamingResponse, FileResponse
from genshi.template import MarkupTemplate
from pyproj import CRS as PyprojCRS
import imageio
import cv2 as cv

from ....crs.geocrs import GeoCrs
from ....spacedescriptors.georect import GeoRect
from ....storage.geostorage import GeoStorage

router = APIRouter(
    prefix='/backend',
    tags=["wms"],
    responses={404: {"description": "Not found"}},
)

path = Path(os.path.realpath(__file__)).parents[0]


@router.get("/{layer}/wms6")
async def get_wms_capabilities(request: Request, layer: str,
                               REQUEST: str, SERVICE: str = 'WMS',
                               LAYERS=None, CRS: str = None, SRS: str = None, BBOX: str = None, WIDTH=None, HEIGHT=None,
                               FORMAT='Image/png', VERSION='1.1.1', STYLES=''):  # TODO: type hints missing
    print("WMS-REQUEST")
    assert (SERVICE.lower() == 'wms')
    if REQUEST.lower() == "getcapabilities":
        with open(path / 'resources' / 'wmscapabilitiestemplate.xml') as template:
            tmpl = MarkupTemplate(template)
            absolute_uri = str(request.url).split('?')[0]
            # layer = get_geodatastorage(geostorage).get_layer_with_name(layer)
            ll_bbx_minx = -180
            ll_bbx_maxx = 180
            ll_bbx_miny = -90
            ll_bbx_maxy = 90
            stream = tmpl.generate(absolute_uri=absolute_uri, ll_bbx_minx=ll_bbx_minx, ll_bbx_miny=ll_bbx_miny,
                                   ll_bbx_maxx=ll_bbx_maxx, ll_bbx_maxy=ll_bbx_maxy, geodatastorage_name='GeoStorage',
                                   layer_name=layer)
            return Response(content=stream.render('xml'), media_type="application/xml")
    elif REQUEST.lower() == "getmap":
        start = time.time()
        bbx = [float(i) for i in BBOX.split(',')]
        if CRS is not None:
            crs = GeoCrs(CRS)
        elif SRS is not None:
            crs = GeoCrs(SRS)
        else:
            raise ValueError("No CRS/SRS provided")
        georect = GeoRect(bbx[:2], bbx[2:], crs=crs)
        geostorage: GeoStorage = request.app.geostorage
        georect = GeoRect.from_bounds(bbx, crs=GeoCrs(CRS))
        layer = geostorage.get_layer(layer)
        get_raster_start = time.time()
        #try:
        raster = layer.get_raster_for_rect(georect, no_data_value=int(0), resolution_rc=(int(HEIGHT), int(WIDTH)))
        print(raster.data.sum())
        raster_data = raster.data.astype(np.uint8)
        print("BOUNDS: " + str(bbx))
        print("BOUNDS TRANSFORMED: " + str(georect.get_bounds()))
        print("RESOLUTION: " + str((int(HEIGHT), int(WIDTH))))
        print(layer.directory_path)
        print("get_raster_for_rect(..) took {} seconds".format(time.time() - get_raster_start))
        #except Exception as e:
            #print(e)
            #raster_data = np.ones((int(HEIGHT), int(WIDTH)), dtype=np.uint8) * 160
        #raster_data = np.moveaxis(raster_data, 1, 0)
        #raster_data = np.flip(raster_data, 0)
        #raster_data = np.flip(raster_data, 1)
        #img_png = BytesIO()
        #imageio.imwrite('test.png', raster_data, format='png')
        #imageio.imwrite(img_png, raster_data, format='png')
        #img_png.seek(0)
        print("Whole wms request took {} seconds.".format(time.time() - start))
        raster_data = np.flip(raster_data, axis=2)
        cv.imwrite('img.png', raster_data)
        img_png = cv.imencode('.png', raster_data, params=[cv.IMWRITE_PNG_COMPRESSION, 0])[1]
        with open('test.png', 'wb') as f:
            f.write(img_png.tobytes())
        return StreamingResponse(BytesIO(img_png), media_type="image/png")
    return "Request not known"


@router.get("/download/test")
def download_file():
    return FileResponse('./test_patch.las', media_type='application/octet-stream', filename='test.las')