/**
 * Created by Michael on 02/07/16.
 */

import React, {Component} from 'react';
import {Navbar, Nav, NavItem } from 'react-bootstrap';

export class AppNav extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <Navbar inverse>
                <Navbar.Header>
                    <Navbar.Brand>
                        <a href="#">Licenta</a>
                    </Navbar.Brand>
                    <Navbar.Toggle />
                </Navbar.Header>
                <Navbar.Collapse>
                    <Nav>
                        <NavItem href="#">Aplicatia principala</NavItem>
                        <NavItem href="#">Cautare imagini</NavItem>
                    </Nav>
                    <Nav pullRight>
                        <NavItem href="#">Lucrarea scrisa</NavItem>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        );
    }
}