/**
 * Created by Michael on 03/07/16.
 */

/* React */
import React, {Component} from 'react';

/* Redux */
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';

/* Actions */
import { process_image } from '../actions/index';

/* Other */
import Dropzone from 'react-dropzone';

class MainApp extends Component {
    constructor(props) {
        super(props);

        this.state = {
            dropzone_available: true,
            dropped_item: null,
            preview: "",
            endpoint: "/process_image_open_theano/"
        };

        this.onDrop = this.onDrop.bind(this);
    }

    dropzone_box() {
        return (
            <Dropzone className="dropbox_style"
                      activeClassName="dropbox_style_active"
                      ref="dropzone"
                      onDrop={(files) => {this.onDrop(files)}}>
                <div style={{textAlign: "center", marginTop: "150px"}}>
                    Analizeaza o imagine.
                </div>
            </Dropzone>
        );
    }

    dropped_display(source) {
        return (
            <div className="dropbox_style">
                <img src={ source }
                     style={{'width': '290px', 'height': '290px'}}/>
            </div>
        );
    }

    componentWillUpdate(nextProps, nextState) {
        console.log("Will update np: ", nextProps);
        console.log("Will update np: ", nextState);
    }

    onDrop(dropped_item) {
        console.log("onDrop called. Current state: ", this.state);
        this.setState({
            dropzone_available: false,
            dropped_item: dropped_item[0],
            preview: dropped_item[0].preview
      }, () => {
            console.log("onDrop callback. Current state: ", this.state);
            console.log("This is dropped display: ", this.dropped_display);
            this.props.process_image(dropped_item[0], this.state.endpoint);
        });

    }

    render() {
        console.log("Rendering with state: ", this.state);
        return (
            <div>
                <div className="jumbotron">

                    {this.state.dropzone_available ? this.dropzone_box() : this.dropped_display(this.state.preview)}

                    {!this.state.dropzone_available ?
                    <div className="center_button">
                        <button className="buttonStyle"
                                onClick={() => {
                                    this.setState({
                                        dropzone_available: true,
                                        dropped_item: null,
                                        preview: ""
                                    });
                                }}>
                            Incearca o alta imagine
                        </button>
                    </div> : <div></div>}
                </div>

                <div className="btn-group" role="group" aria-label="...">
                  <button type="button"
                          className="btn btn-default"
                          onClick={() => {
                            this.setState({
                                endpoint: '/process_image_open_cv/'
                            });
                          }}>Endpoint Open CV</button>
                  <button type="button"
                          className="btn btn-default"
                          onClick={() => {
                            this.setState({
                                endpoint: '/process_image_open_theano/'
                            });
                          }}>Endpoint Theano</button>
                </div>

                <div className="well">
                    className: {this.state.endpoint}
                </div>
            </div>
        );
    }
}


function mapStateToProps(state) {
    return {};
}

function mapDispatchToProps(dispatch) {
    return bindActionCreators({ process_image }, dispatch);
}

export default connect(mapStateToProps, mapDispatchToProps)(MainApp);