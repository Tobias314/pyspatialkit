import React from "react"

import ListGroup from "react-bootstrap/ListGroup"

export declare interface LayerListProps{
    layerNames: string[]
}

export class LayerList extends React.Component<LayerListProps>{

    constructor(props: LayerListProps){
        super(props)
    }

    render(){
        const listItems = this.props.layerNames.map((layer: string) => 
            <ListGroup.Item action key={layer}>{layer}
            </ListGroup.Item>
        );
        return (
            <ListGroup>
                {listItems}
            </ListGroup>
        )
    }
}