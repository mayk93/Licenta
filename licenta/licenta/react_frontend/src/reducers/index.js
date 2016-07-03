import { combineReducers } from 'redux';
import CurrentViewReducer from './reducer_current_view';

const rootReducer = combineReducers({
  current_view: CurrentViewReducer
});

export default rootReducer;
