import request from 'superagent';

export function change_current_view(target) {
    return {
        type: "VIEW_CHANGE",
        payload: target
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
                return {
                    type: "IMAGE_PROCESS",
                    payload: response.body
                };
            } else {
                console.log("Exception processing image!");
                return {
                    type: "IMAGE_PROCESS",
                    payload: null
                };
            }
        });
    }
}


export function chart_reducer_laison(value) {
    return {
        type: "CHART_DATA",
        payload: value
    };
}


export function get_chart_data() {
    console.log("Now get_chart_data");
    return function(dispatch) {
        var chart_request = request.get('/chart_data');
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