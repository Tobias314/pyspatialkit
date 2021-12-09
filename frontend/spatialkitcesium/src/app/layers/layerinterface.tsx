import {CesiumComponentType} from 'resium'

export interface LayerInterface {
    name: string;

    toResiumComponent(): any;
}