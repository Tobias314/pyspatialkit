import React from "react";
import { WebMapServiceImageryProvider, ImageryProvider} from "cesium"
import { ImageryLayer } from "resium";
import { LayerInterface } from "./layerinterface";
import { BACKEND_URL } from "../constants";


export class RasterLayer implements LayerInterface{

    constructor(name: string){
        this.name = name;
    }

    toResiumComponent(): any{
        return (<ImageryLayer imageryProvider={this.imageryProvider}/>)
    }

    name: string;
    _imageryProvider: ImageryProvider = undefined

    get imageryProvider(): ImageryProvider{
        if(this._imageryProvider==undefined){
            const url = `${BACKEND_URL}/${this.name}/wms`
            this._imageryProvider = new WebMapServiceImageryProvider({
                url : url,
                layers : '0',
            });
        }
        return this._imageryProvider
    }
}