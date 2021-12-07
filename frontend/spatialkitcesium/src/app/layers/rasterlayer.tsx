import { LayerInterface } from "./layerinterface";


export class RasterLayer implements LayerInterface{
    name: string;

    constructor(name: string){
        this.name = name;
    }
}