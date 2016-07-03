/* React */
import React, {Component} from 'react';

/* Components */
import { AppNav } from '../containers/nav';

export default class App extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <AppNav />
        );
    }
}
