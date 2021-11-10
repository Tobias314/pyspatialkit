from pathlib import Path
import os
from io import BytesIO
import time

import numpy as np
from fastapi import APIRouter, Response, Request
from fastapi.responses import StreamingResponse
from genshi.template import MarkupTemplate
from pyproj import CRS as PyprojCRS
import imageio

from ....crs.geocrs import GeoCrs
from ....spacedescriptors.georect import GeoRect
from ....storage.geostorage import GeoStorage

router = APIRouter(
    prefix='/backend',
    tags=["wms"],
    responses={404: {"description": "Not found"}},
)

path = Path(os.path.realpath(__file__)).parents[0]


@router.get("/{layer}/wms")
async def get_wms_capabilities(request: Request, layer: str,
                               REQUEST: str, SERVICE: str = 'WMS',
                               LAYERS=None, CRS: str = None, SRS: str = None, BBOX: str = None, WIDTH=None, HEIGHT=None,
                               FORMAT='Image/png', VERSION='1.1.1', STYLES=''):  # TODO: type hints missing
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
        layer = geostorage.get_layer(layer)
        georect.to_crs(layer.crs)
        get_raster_start = time.time()
        try:
            raster = layer.get_raster_for_rect(georect, resolution=(int(WIDTH), int(HEIGHT)))
            print(raster.data.sum())
            raster_data = raster.data.astype(np.uint8)
            print("BOUNDS: " + str(bbx))
            print("BOUNDS TRANSFORMED: " + str(georect.get_bounds()))
            print("RESOLUTION: " + str((int(WIDTH), int(HEIGHT))))
            print(layer.directory_path)
            print("get_raster_for_rect(..) took {} seconds".format(time.time() - get_raster_start))
        except:
            print('except')
            raster_data = np.ones((int(WIDTH), int(HEIGHT)), dtype=np.uint8) * 160
        img_png = BytesIO()
        imageio.imwrite(img_png, raster_data, format='png')
        img_png.seek(0)
        print("Whole wms request took {} seconds.".format(time.time() - start))
        return StreamingResponse(img_png, media_type="image/png")
    return "Request not known"