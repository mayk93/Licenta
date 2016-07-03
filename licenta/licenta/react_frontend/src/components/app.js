/* React */
import React, {Component} from 'react';

/* Components and Containers */
import AppNav from '../containers/nav';
import AppBody from '../containers/body';

export default class App extends Component {
    constructor(props) {
        
        super(props);
    }

    render() {
        return (
            <div>
                <AppNav />
                <AppBody />
            </div>
        );
    }
}
