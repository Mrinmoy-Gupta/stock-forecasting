
import numpy as np
import pandas as pd
import os
import argparse
import logging
import mlflow
from datetime import datetime, timedelta
from dateutil import parser
import dateutil.relativedelta
import pytz
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath

def data_process(start, end, df):
    duplicate_df = pd.date_range(start,end, freq='T')
    df_temp = pd.DataFrame({ 't': duplicate_df, 'd': None }) 
    common = df_temp.merge(df, on=["t"])
    row_not_in_originaldf = df_temp[~df_temp.t.isin(common.t)]
    row_not_in_originaldf.reset_index(drop=True, inplace=True)
    row_not_in_originaldf['c'] = row_not_in_originaldf['d']
    new_row_not_in_originaldf=row_not_in_originaldf.drop('d',axis=1)
    final_df=pd.concat([df,new_row_not_in_originaldf],ignore_index=True)
    final_df = final_df.sort_values(by="t",ignore_index=True)
    final_df=final_df.drop_duplicates('t',keep='first')
    new_final_df=final_df.fillna(method='ffill')
    new_final_df.drop(new_final_df.index[-1], inplace=True)
    return new_final_df



def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--n_test_points", type=int, required=False, default=300)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--refined_data", type=str, help="path to refined data")
    
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    df = pd.read_csv(os.path.join(args.data, 'data.csv'))
    df=df.sort_values(by='t')

    df['t'] = pd.to_datetime(df['t'])
    df['t'] = df['t'] + pd.Timedelta(hours=5.5)
    
    start_date=df.iloc[0].tolist()
    end_date=df.iloc[-1].tolist()
    new_final_df = data_process(start_date[0], end_date[0], df)
    print()
    print(new_final_df)
    print()
    train_df = new_final_df.iloc[:-args.n_test_points]
    test_df = new_final_df.iloc[-args.n_test_points:]

    mlflow.log_metric("num_train_samples", train_df.shape[0])
    mlflow.log_metric("num_test_samples", test_df.shape[0])

    print(train_df.shape)
    print(test_df.shape)

    train_path = os.path.join(args.train_data, 'train.csv')
    test_path = os.path.join(args.test_data, 'test.csv')
    
    print(train_path)
    print(test_path)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    now = datetime.today() + pd.Timedelta(hours = 5.5)
    end = pd.to_datetime(now.strftime("%Y-%m-%d %H:%M") + ":00")
#     end = now
    print(end)
    new_df = data_process(start_date[0], end, df)
    refined_data_path = os.path.join(args.refined_data, 'refined_data.csv')
    print(new_df.tail())
    new_df.to_csv(refined_data_path, index=False)

    ws = Workspace.get(name="AML-AWH-AZPOC",
    subscription_id='ab8d90b6-c993-497e-aa6b-033721a4ccb3',
    resource_group='HANU-ROHIT-RG')

    datastore = Datastore.get(ws, 'workspaceblobstore')

    ds = Dataset.File.upload_directory(src_dir=args.train_data,
            target=DataPath(datastore,  'dataset/mrinmoy/train'),
            show_progress=True, overwrite=True)
    
    ds = Dataset.File.upload_directory(src_dir=args.test_data,
            target=DataPath(datastore,  'dataset/mrinmoy/test'),
            show_progress=True, overwrite=True)

    ds = Dataset.File.upload_directory(src_dir=args.refined_data,
            target=DataPath(datastore,  'dataset/mrinmoy/refined_data'),
            show_progress=True, overwrite=True)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
