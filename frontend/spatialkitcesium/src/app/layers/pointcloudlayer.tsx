import React from "react";
import { Cesium3DTileset } from "resium";
import { LayerInterface } from "./layerinterface";
import { BACKEND_URL } from "../constants";


export class PointCloudLayer implements LayerInterface{

    constructor(name: string){
        this.name = name;
        this.tilesetUrl = `${BACKEND_URL}/${this.name}/tiles/root.json`
    }

    name: string;
    tilesetUrl: string;

    toResiumComponent(): any{
        return (<Cesium3DTileset url={this.tilesetUrl}/>)
    }
}