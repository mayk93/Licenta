/**
 * Created by Michael on 23/07/16.
 */


/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

/* Actions */
import { get_chart_data, approximate_chart_function, clear_chart } from '../actions/index';

/* Other */
import Plotly from 'react-plotlyjs';


class TestApp extends Component {
    constructor(props) {
        super(props);
        this.props.get_chart_data();
    }


    render() {
        console.log("Render props: ", this.props);

        return (
          <div>
            <Plotly data={this.props.chart_data.concat(this.props.approximation_data)} />
            <button type="button"
                    className="btn btn-default"
                    onClick={() => {this.props.clear_chart(); this.props.get_chart_data()}}>New Chart</button>
            <button type="button"
                    className="btn btn-default"
                    onClick={() => {this.props.approximate_chart_function(this.props.chart_data)}}>Solve</button>
          </div>
        );
    }
}


function mapStateToProps(state) {
    return {
        chart_data: state.chart_data,
        approximation_data: state.approximation_data
    };
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({ get_chart_data, approximate_chart_function, clear_chart }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(TestApp);