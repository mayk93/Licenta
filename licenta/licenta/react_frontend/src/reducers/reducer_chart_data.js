/**
 * Created by Michael on 23/07/16.
 */

export default function (state = [], action) {
    console.log("Chart reducer. Action type: ", action.type);
    switch (action.type) {
        case "CHART_DATA":
            console.log("Reducer. Setting chart_data to: ", action.payload);
            return action.payload;
        default:
            return state;
    }
}