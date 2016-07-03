/**
 * Created by Michael on 02/07/16.
 */

/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

/* Actions */
import { change_current_view } from '../actions/index';

/* Other */
import {Navbar, Nav, NavItem } from 'react-bootstrap';

class AppNav extends Component {
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
                        <NavItem href="#" onClick={ () => {this.props.change_current_view("main_app")} }>
                            Aplicatia principala
                        </NavItem>
                        <NavItem href="#" onClick={ () => {this.props.change_current_view("search")} }>
                            Cautare imagini
                        </NavItem>
                    </Nav>
                    <Nav pullRight>
                        <NavItem href="#" onClick={ () => {this.props.change_current_view("project")} }>
                            Lucrarea scrisa
                        </NavItem>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        );
    }
}

function mapStateToProps(state) {
    return {
      current_view: state.current_view
    };
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({ change_current_view }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(AppNav);