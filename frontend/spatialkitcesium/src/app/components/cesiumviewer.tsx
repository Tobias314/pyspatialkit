import React from "react"
import { Cartesian3, WebMapServiceImageryProvider} from "cesium";
import { Viewer, Entity, PointGraphics, ImageryLayer, CameraLookAt, CameraFlyTo} from "resium";
import {viewerCesium3DTilesInspectorMixin} from "cesium"
import { LayerInterface } from "../layers/layerinterface";

declare interface CesiumViewerProps{
    layers: LayerInterface[]
}

const targetPosition = Cartesian3.fromDegrees(12.435281, 51.846743, 400)
const cameraFlyTo = <CameraFlyTo destination={targetPosition} duration={15}/>

export class CesiumViewer extends React.Component<CesiumViewerProps>{

    constructor(props: CesiumViewerProps){
        super(props)
    }

    render(){
        let renderComponents = new Array();
        let key = 0
        for(let layer of this.props.layers){
            renderComponents.push(<React.Fragment key={key}>{layer.toResiumComponent()}</React.Fragment>)
        }
        
        //<ImageryLayer imageryProvider={imageryProvider} />
        return (
            <Viewer className="full-screen-height container-fluid" extend={viewerCesium3DTilesInspectorMixin}>
                {cameraFlyTo}
                {renderComponents}
            </Viewer>
        );
    }
}