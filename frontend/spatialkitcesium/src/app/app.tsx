import React from "react";
import ReactDOM from "react-dom";
import {sayHello} from "./test";


ReactDOM.render(
    <div>
    <button onClick={sayHello}>Default</button>,
    <div>Hello reac3t23!</div>
    </div>,
    document.getElementById('root')
)