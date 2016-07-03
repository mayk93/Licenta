/**
 * Created by Michael on 03/07/16.
 */

export default function (state = '', action) {
    switch (action.type) {
        case "VIEW_CHANGE":
            return action.payload;
        default:

            return state;
    }
}