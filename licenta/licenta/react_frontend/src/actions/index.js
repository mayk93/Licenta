import request from 'superagent';

export function change_current_view(target) {
    return {
        type: "VIEW_CHANGE",
        payload: target
    };
}

export function process_image(image) {
    console.log("Now processing image.");
    var process_request = request.post('/process_image/');
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