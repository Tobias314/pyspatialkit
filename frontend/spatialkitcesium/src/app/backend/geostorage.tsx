import $ from "jquery"
import {BACKEND_URL} from "../constants"
import {LayerInterface} from "../layers/layerinterface"
import {RasterLayer} from "../layers/rasterlayer"
import {LayersDescriptor, LayerDescriptor, LayerTypes} from "./descriptors/layerdescriptor"

function parseLayerDescriptor(layerDescriptor: LayerDescriptor) : LayerInterface | undefined{
    if (layerDescriptor.type == LayerTypes.GEO_RASTER_LAYER){
        return new RasterLayer(layerDescriptor.name);
    }else{
        throw new TypeError("Backend responded with layer of unknown type!")
        return undefined;
    }
}

export function getLayers(onSuccess: (layerList: Array<LayerInterface>) => void){
    $.ajax({
        url: BACKEND_URL + "/layers",
        dataType: 'json',
        success: function(layersDescriptor: LayersDescriptor){
            let result: Array<LayerInterface> = new Array();
            for(var layerDescriptor of layersDescriptor.layers){
                let layer = parseLayerDescriptor(layerDescriptor);
                if(layer !== undefined){
                    result.push(layer);
                }
            }
            onSuccess(result);
        }
    })
}