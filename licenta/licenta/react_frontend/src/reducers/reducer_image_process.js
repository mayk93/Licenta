/**
 * Created by Michael on 04/07/16.
 */

export default function (state = null, action) {
    switch (action.type) {
        case "IMAGE_PROCESS":
            return action.payload;
        default:
            return state;
    }
}