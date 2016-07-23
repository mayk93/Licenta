import {combineReducers} from 'redux';
import CurrentViewReducer from './reducer_current_view';
import ImageProcessReducer from './reducer_image_process';
import ChartDataReducer from './reducer_chart_data';
import ApproximationReducer from './reducer_approximation';

const rootReducer = combineReducers({
    current_view: CurrentViewReducer,
    image_process: ImageProcessReducer,
    chart_data: ChartDataReducer,
    approximation_data: ApproximationReducer
});

export default rootReducer;
