/**
 * Created by Michael on 03/07/16.
 */

/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

/* Components and Containers */
import MainApp from './mainapp';
import TestApp from './testapp';

import SearchApp from '../components/search';
import Project from '../components/project';

class AppBody extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        if (this.props.current_view == "main_app") {
            return (
                <MainApp />
            );
        } else if (this.props.current_view == "test_app") {
          return (
              <TestApp />
          );  
        } else if(this.props.current_view == "search") {
            return (
                <SearchApp />
            );
        } else if(this.props.current_view == "project") {
            return (
                <Project />
            );
        } else {
            return <MainApp />
        }

    }
}

function mapStateToProps(state) {
    return {
      current_view: state.current_view
    };
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({  }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(AppBody);