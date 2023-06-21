import argparse
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import mlflow
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import GridSearchCV
import mango
from mango import Tuner

# Start Logging


os.makedirs("./results", exist_ok=True)

mlflow.start_run()

def main():
    

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model")

    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.train_data, 'train.csv'))
    train_df['t'] = pd.to_datetime(train_df['t'])
    train_df['t'] = train_df['t'].dt.tz_localize(None)
    train_df.set_index("t", inplace=True)

    test_df = pd.read_csv(os.path.join(args.test_data, 'test.csv'))
    test_df['t'] = pd.to_datetime(test_df['t'])
    test_df['t'] = test_df['t'].dt.tz_localize(None)
    test_df.set_index("t", inplace=True)

    print("train info")
    print(train_df.info())

    print("test info")
    print(test_df.info())

    # def objective_function(args_list):
    #     errors = []
    #     for params in args_list:
    #         try:
    #             model = ExponentialSmoothing(train_df['c'],trend='MUL',seasonal='MUL', freq='T', seasonal_periods=1440).fit(optimized=False, **params)
    #             forecast = model.forecast(len(test_df)).values
    #             error = mse(test_df['c'], forecast, squared=False)
    #         # print(params, ": error = ", error) 
    #             errors.append(error)
    #         except:
    #             errors.append(1000.0)
    #     return errors

    # param_grid = dict(
    # smoothing_level = np.linspace(0.01, 0.9, 10),
    # smoothing_trend =  np.linspace(0.01, 0.9, 10),
    # smoothing_seasonal = np.linspace(0.01, 0.9, 10),
    # )
    # grid = ParameterGrid(param_grid)

    # conf_Dict = dict()
    # conf_Dict['initial_random'] = 100
    # conf_Dict['num_iteration'] = 50

    # tuner = Tuner(param_grid, objective_function, conf_Dict)
    # results = tuner.minimize()
    # print("HyperParameter Tuning completed!")


    # print('smoothing_trend', results['best_params']['smoothing_trend'])
    # print('smoothing_seasonal', results['best_params']['smoothing_seasonal'])
    # print('smoothing_level', results['best_params']['smoothing_level'])

    # mlflow.log_param('smoothing_trend', results['best_params']['smoothing_trend'])
    # mlflow.log_param('smoothing_seasonal', results['best_params']['smoothing_seasonal'])
    # mlflow.log_param('smoothing_level', results['best_params']['smoothing_level'])

    # df_train_temp = train_df.copy()
    # forecasts = np.array([])
    # for i in range(len(test_df)):
    #     fitted_model = ExponentialSmoothing(df_train_temp['c'],trend='MUL',seasonal='MUL', freq='T',
    #                                         seasonal_periods=1440).fit(
    #                                         smoothing_trend = results['best_params']['smoothing_trend'], 
    #                                         smoothing_seasonal = results['best_params']['smoothing_seasonal'],
    #                                         smoothing_level =  results['best_params']['smoothing_level'],
    #                                         optimized=False)
    #     forecast = fitted_model.forecast(1).values[0]
    #     forecasts = np.append(forecasts, forecast)
    #     df_train_temp.reset_index(inplace=True)
    #     df_train_temp.loc[len(df_train_temp)] = test_df.index[i], forecast
    #     df_train_temp.set_index("t", inplace=True)
    # df_predictions = df_train_temp.iloc[-len(test_df):]
    # rmse = mse(test_df['c'], df_predictions, squared=False)

    # mlflow.log_metric('RMSE', rmse)
    
    # train = pd.concat([train_df, test_df], ignore_index=False)
    # print(train.info())
    # print()
    # print(train)
    # fitted_model = ExponentialSmoothing(train['c'],trend='MUL',seasonal='MUL', freq='T',
    #                                         seasonal_periods=1440).fit(
    #                                         smoothing_trend = results['best_params']['smoothing_trend'], 
    #                                         smoothing_seasonal = results['best_params']['smoothing_seasonal'],
    #                                         smoothing_level =  results['best_params']['smoothing_level'],
    #                                         optimized=False)

    # print("saving model with mlflow")
    # mlflow.statsmodels.save_model(
    #     statsmodels_model=fitted_model,
    #     path = args.model
    # )

    # mlflow.end_run()
   
if __name__ == "__main__":
    main()
