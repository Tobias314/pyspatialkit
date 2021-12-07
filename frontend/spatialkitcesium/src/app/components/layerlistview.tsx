import React from "react"

import ListGroup from "react-bootstrap/ListGroup"
import { LayerInterface } from "../layers/layerinterface";

export declare interface LayerListProps{
    layers: LayerInterface[]
}

export class LayerListView extends React.Component<LayerListProps>{

    

    render(){
        const listItems = this.props.layers.map((layer) => 
            <ListGroup.Item action key={layer.name}>{layer.name}
            </ListGroup.Item>
        );
        return (
            <ListGroup>
                {listItems}
            </ListGroup>
        )
    }
}