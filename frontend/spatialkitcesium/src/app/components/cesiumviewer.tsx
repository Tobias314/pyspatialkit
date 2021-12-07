import React from "react"
import { Viewer } from "resium";

export class CesiumViewer extends React.Component{

    constructor(props){
        super(props)
    }

    render(){
        return (
            <Viewer className="full-screen-height container-fluid"/>
        )
    }
}