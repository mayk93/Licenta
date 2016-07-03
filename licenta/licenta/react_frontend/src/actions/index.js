export function change_current_view(target) {
    return {
        type: "VIEW_CHANGE",
        payload: target
    };
}

export function process_image(image) {
    console.log("Now processing image.");
    return {
        type: "IMAGE_PROCESS",
        payload: true
    };
}