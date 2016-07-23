/**
 * Created by Michael on 23/07/16.
 */

/**
 * Created by Michael on 23/07/16.
 */

export default function (state = [], action) {
    console.log("Approximation reducer. Action type: ", action.type);
    switch (action.type) {
        case "APPROXIMATION_DATA":
            console.log("Reducer. Setting approximation to: ", action.payload);
            return action.payload;
        default:
            return state;
    }
}