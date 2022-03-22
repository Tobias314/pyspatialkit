import React from "react";
import { Cesium3DTileset } from "resium";
import { Cesium3DTileStyle, PointCloudShading} from "cesium"
import { LayerInterface } from "./layerinterface";
import { BACKEND_URL } from "../constants";


export class PointCloudLayer implements LayerInterface{
    name: string;
    tilesetUrl: string;
    style: Cesium3DTileStyle;
    shading: PointCloudShading

    constructor(name: string){
        this.name = name;
        this.tilesetUrl = `${BACKEND_URL}/${this.name}/tiles/root.json`
        this.style = new Cesium3DTileStyle({
                pointSize: 10
            });
        this.shading = new PointCloudShading({eyeDomeLighting:true})
    }

    toResiumComponent(): any{
        return (<Cesium3DTileset url={this.tilesetUrl} style={this.style} pointCloudShading={this.shading}/>)
    }
}