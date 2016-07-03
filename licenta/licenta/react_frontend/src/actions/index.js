export function change_current_view(target) {
    return {
        type: "VIEW_CHANGE",
        payload: target
    };
}