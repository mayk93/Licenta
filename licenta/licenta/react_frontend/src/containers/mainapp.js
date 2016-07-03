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
import { Button } from 'react-bootstrap';

class MainApp extends Component {
    constructor(props) {
        super(props);

        this.state = {
            files: []
        }
    }

    onDrop(files) {
        console.log("onDrop called. Current state: ", this.state);
        this.setState({
            files: files
      }, () => {
            console.log("onDrop callback. Current state: ", this.state);
        });
    }

    render() {
        return (
            <div className="jumbotron">
                <Dropzone className="dropbox_style"
                          activeClassName="dropbox_style_active"
                          ref="dropzone"
                          onDrop={this.onDrop}>
                    <div style={{"text-align": "center", "margin-top": "150px"}}>
                        Analizeaza o imagine.
                    </div>
                </Dropzone>
                {
                    this.state.files.length > 0 ?
                    <div>
                        <h2>Uploading {this.state.files.length} files...</h2>
                        <div>
                        {this.state.files.map((file) => <img src={file.preview} /> )}
                        </div>
                    </div>
                    :
                    null
                }
                <div className="center_button">
                    <button bsStyle="" className="buttonStyle">
                        Incearca o alta imagine
                    </button>
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