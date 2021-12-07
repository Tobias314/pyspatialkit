export enum LayerTypes {
    GEO_RASTER_LAYER = "GeoRasterLayer",
}

export interface LayersDescriptor{
    layers: [LayerDescriptor]
}

export interface LayerDescriptor{
    name: string;
    type: string;
    dataType: string;
}