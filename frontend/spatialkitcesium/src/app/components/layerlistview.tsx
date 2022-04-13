import React from "react"

import ListGroup from "react-bootstrap/ListGroup"
import { LayerInterface } from "../layers/layerinterface";

export declare interface LayerListProps{
    layers: LayerInterface[],
    selectedLayers: Set<number>,
    onToggleLayer: (layerIndex: number) => void
}

export class LayerListView extends React.Component<LayerListProps>{

    constructor(props: LayerListProps){
        super(props);
    }
    

    render(){
        const listItems = this.props.layers.map((layer, index) => 
                <ListGroup.Item action key={index} onClick={() => this.props.onToggleLayer(index)} >
                    <div className="nobr">
                        {layer.name}  <input type="checkbox" checked={this.props.selectedLayers.has(index)} onChange={() => {}}/>
                    </div>
                </ListGroup.Item>
        );
        return (
            <ListGroup>
                {listItems}
            </ListGroup>
        )
    }
}