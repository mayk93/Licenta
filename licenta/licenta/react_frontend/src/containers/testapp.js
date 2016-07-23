/**
 * Created by Michael on 23/07/16.
 */


/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

/* Actions */
import { get_chart_data } from '../actions/index';

/* Other */
import Plotly from 'react-plotlyjs';


class TestApp extends Component {
    constructor(props) {
        super(props);
        this.state = {
            chart_data: []
        };

        this.props.get_chart_data();

        console.log("Constructor props: ", this.props);
    }

    componentWillUpdate(new_props){
        this.setState({
           chart_data: this.props.chart_data
        });

        console.log("componentWillUpdate props: ", this.props);
        console.log("componentWillUpdate new props: ", new_props);
        console.log("componentWillUpdate state: ", this.state.chart_data);
    }

    render() {
        console.log("Render props: ", this.props);
        console.log("Render state: ", this.state);

        return (
          <Plotly className="whatever" data={this.props.chart_data} />
        );
    }
}


function mapStateToProps(state) {
    return {
        chart_data: state.chart_data
    };
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({ get_chart_data }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(TestApp);