from fastapi import APIRouter, Response, Request

from ....storage.geostorage import GeoStorage

from ....storage.geolayer import GeoLayer
from ....storage.raster.georasterlayer import GeoRasterLayer
from ....storage.pointcloud.geopointcloudlayer import GeoPointCloudLayer

router = APIRouter(
    prefix='/backend',
    tags=["geostorage"],
    responses={404: {"description": "Not found (projects)"}},
)

@router.get("/layers")
async def get_layers(request: Request):
        geostorage: GeoStorage = request.app.geostorage
        response = []
        for name, layer in geostorage.layers.items():
            response.append(generate_descriptor_for_layer(name, layer))
        return {"layers": response}

def generate_descriptor_for_layer(layer_name: str, layer: GeoLayer):
    descriptor = {}
    descriptor["name"] = layer_name
    descriptor["type"] = str(type(layer).__name__)
    if isinstance(layer, GeoRasterLayer):
        descriptor["dataType"] = "raster"
    elif isinstance(layer, GeoPointCloudLayer):
        descriptor["dataType"] = "pointcloud"
    else:
        descriptor["dataType"] = "custom"
    return descriptor