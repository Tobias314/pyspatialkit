import 'bootstrap/dist/css/bootstrap.min.css';
import '../styling/main.css'


import React from "react";
import ReactDOM from "react-dom";
import {sayHello} from "./test";
import { Viewer } from "resium";

import {CesiumViewer} from "./cesiumviewer"
import {LayerList} from "./layerlist"

function App() {
    return (
        <div className="d-flex">
            <LayerList layerNames={['layer1', 'layer2', 'layer3', 'layer4']}/>
            <CesiumViewer/>
        </div>
    )
  }

ReactDOM.render(<App />, document.getElementById("root"));