export enum LayerTypes {
    GEO_RASTER_LAYER = "GeoRasterLayer",
    GEO_POINT_CLOUD_LAYER = "GeoPointCloudLayer"
}

export interface LayersDescriptor{
    layers: [LayerDescriptor]
}

export interface LayerDescriptor{
    name: string;
    type: string;
    dataType: string;
}