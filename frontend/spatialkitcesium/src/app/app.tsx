import 'bootstrap/dist/css/bootstrap.min.css';
import '../styling/main.css'


import React from "react";
import ReactDOM from "react-dom";
import {sayHello} from "./test";
import { Viewer } from "resium";

import {CesiumViewer} from "./components/cesiumviewer"
import {LayerListView} from "./components/layerlistview"
import { LayerInterface } from './layers/layerinterface';
import {getLayers} from './backend/geostorage';

interface AppState{
    layers: Array<LayerInterface>;
    selectedLayers: Set<number>;
}

class App extends React.Component<{}, AppState>{

    constructor(props: {}){
        super(props);
        this.state = {
            layers: new Array(),
            selectedLayers: new Set(),
        };
    }

    componentDidMount(){
        getLayers(layers => {
            this.setState({
                layers: layers
            });
        });
    }

    toggleLayer(layerIndex: number){
        this.setState((state) => {
            let selectedLayers = state.selectedLayers;
            if(selectedLayers.has(layerIndex)){
                selectedLayers.delete(layerIndex);
            }else{
                selectedLayers.add(layerIndex)
            }
            return {selectedLayers: selectedLayers};
        });
    }

    render(){
        return (
            <div className="d-flex">
                <LayerListView layers={this.state.layers} selectedLayers={this.state.selectedLayers} onToggleLayer={this.toggleLayer.bind(this)}/>
                <CesiumViewer layers={[...this.state.selectedLayers].map(layerIndex => this.state.layers[layerIndex])}/>
            </div>
        )
  }
}

ReactDOM.render(<App />, document.getElementById("root"));