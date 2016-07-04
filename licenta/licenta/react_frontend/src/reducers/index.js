import {combineReducers} from 'redux';
import CurrentViewReducer from './reducer_current_view';
import ImageProcessReducer from './reducer_image_process';

const rootReducer = combineReducers({
    current_view: CurrentViewReducer,
    image_process: ImageProcessReducer
});

export default rootReducer;
