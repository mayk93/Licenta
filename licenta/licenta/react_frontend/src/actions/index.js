import request from 'superagent';

export function change_current_view(target) {
    return {
        type: "VIEW_CHANGE",
        payload: target
    };
}


export function process_image_laison(value) {
    return {
        type: "IMAGE_PROCESS",
        payload: value
    };
}


export function process_image(image, endpoint) {
    console.log("Now processing image. Sending to: ", endpoint);
    return function(dispatch) {
        var process_request = request.post(endpoint);
        process_request.attach(image.name, image);
        process_request.end((error, response) => {
            if (error == null) {
                console.log("Success processing image! Response: ", response);
                dispatch(process_image_laison(response.body.prediction));
            } else {
                console.log("Exception processing image!");
                dispatch(process_image_laison(null));
            }
        });
    }
}

export function clear_chart() {
    return function(dispatch) {
        dispatch(chart_reducer_laison([]));
        dispatch(approximation_reducer_laison([]));
    }
}

export function chart_reducer_laison(value) {
    return {
        type: "CHART_DATA",
        payload: value
    };
}

export function approximation_reducer_laison(value) {
    return {
        type: "APPROXIMATION_DATA",
        payload: value
    };
}


export function get_chart_data(value) {
    if ( typeof value == "undefined" ) {
        value = 5;
    }
    console.log("Now get_chart_data. Slope: ", value);
    return function(dispatch) {
        var chart_request = request.post('/chart_data/');
        chart_request.send({ slope: value });
        chart_request.end((error, response) => {
            if (error == null) {
                console.log("Success chart_request: ", response);
                dispatch(chart_reducer_laison(response.body.data));
            } else {
                console.log("Exception chart_request!");
                return {
                    type: "CHART_DATA",
                    payload: []
                };
            }
        });
    }
}

export function approximate_chart_function(chart_data) {
    console.log("Now sending to solve endpoint");
    return function(dispatch) {
        var approximate_request = request.post('/approximate/');
        approximate_request.send({ data: chart_data });
        approximate_request.end((error, response) => {
            if (error == null) {
                console.log("Success approximating! Response: ", response);
                dispatch(approximation_reducer_laison(response.body.approximation));
            } else {
                console.log("Exception approximating!");
                return {
                    type: "APPROXIMATION_DATA",
                    payload: []
                };
            }
        });
    }
}