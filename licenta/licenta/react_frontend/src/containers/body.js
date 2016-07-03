/**
 * Created by Michael on 03/07/16.
 */

/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

class AppBody extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        if (this.props.current_view == "main_app") {
            return (
                <div>
                    Main App hardcoded
                </div>
            );
        } else {
            return (
                <div>
                    { this.props.current_view }
                </div>
            );
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