import time
import http.client, json
from datetime import datetime, timedelta
from dateutil import parser
import dateutil.relativedelta
import pandas as pd
import numpy as np
import pytz
import argparse
import logging
import mlflow
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import requests
import json

def datetotimestamp(date):
    time_tuple = date.timetuple()
    timestamp = round(time.mktime(time_tuple))
    return timestamp

def timestamptodate(timestamp):
    return datetime.fromtimestamp(timestamp)


def realTimeData(datastore):

    # try:
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'dataset/mrinmoy/refined_data/refined_data.csv'))
    print("++++DATA IS ALREADY PRESENT ++++")
    pandas_df = dataset.to_pandas_dataframe() 
    result = pandas_df.sort_values(by="t")
    temp = result.iloc[-1].tolist()
    start = pd.to_datetime(temp[0])
    end = datetime.today() + pd.Timedelta(hours=5.5)
    c = end - start
    minutes = c.total_seconds() // 60
    count_back = int(minutes)
    end_date = datetotimestamp(end)
    start_date = datetotimestamp(start)
    url1 = '/techCharts/indianMarket/stock/history?symbol=TATAELXSI&resolution=1&from='+str(start_date)+'&to='+str(end_date)+'&countback='+str(count_back)+'&currencyCode=INR'
    conn = http.client.HTTPSConnection("priceapi.moneycontrol.com")
    payload = ""
    headers = {}
    conn.request("GET", url1, payload, headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data.decode("utf-8"))
    actual_df = pd.DataFrame(response)
    actual_df.drop(['s', 'o','h','l','v'], axis=1,inplace=True)
    actual_df["t"] = actual_df["t"].apply(timestamptodate)
    actual_df["t"] = actual_df["t"] + pd.Timedelta(hours=5.5)
    actual_df=actual_df.drop_duplicates('t',keep='first')
    df = pd.concat([result, actual_df], ignore_index=False)
    print(actual_df.tail())
    return df
    # except:
    #     print("****NO PREVIOUS DATA OF STOCK FOUND, COLLECTING DATA FOR ENTIRE YEAR****")
    #     date = datetime.today() + pd.Timedelta(hours=5.5)
    #     print(date)
    #     prev_yr_date = date + dateutil.relativedelta.relativedelta(months=-12)
    #     print(prev_yr_date)
    #     start = datetotimestamp(prev_yr_date)
    #     end = datetotimestamp(date)
    #     c = date - prev_yr_date
    #     minutes = c.total_seconds() // 60
    #     count_back = int(minutes)
    #     url1 = '/techCharts/indianMarket/stock/history?symbol=TATAELXSI&resolution=1&from='+str(start)+'&to='+str(end)+'&countback='+str(count_back)+'&currencyCode=INR'
    #     conn = http.client.HTTPSConnection("priceapi.moneycontrol.com")
    #     payload = ""
    #     headers = {}
    #     conn.request("GET", url1, payload, headers)
    #     res = conn.getresponse()
    #     data = res.read()
    #     response = json.loads(data.decode("utf-8"))
    #     actual_df = pd.DataFrame(response)
    #     actual_df["t"] = actual_df["t"].apply(timestamptodate)
    #     actual_df["t"] = actual_df["t"] + pd.Timedelta(hours=5.5)
    #     actual_df.drop(['s', 'o','h','l','v'], axis=1,inplace=True)
    #     return actual_df
    #     print(actual_df.tail())
####
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, help="path to forecasts")
    parser.add_argument("--refined_data", type=str, help="path to refined_data")
    args = parser.parse_args()

    ws = Workspace.get(name="AML-AWH-AZPOC",
               subscription_id='ab8d90b6-c993-497e-aa6b-033721a4ccb3',
               resource_group='HANU-ROHIT-RG')
    datastore = Datastore.get(ws, 'workspaceblobstore')

    # df = realTimeData(datastore)
    # now = datetime.today() + pd.Timedelta(hours = 5.5)
    # now = pd.to_datetime(now.strftime("%Y-%m-%d %H:%M") + ":00")
    # print(now)
    # while(True):
    #     if(now.time()>pd.to_datetime("15:30:00").time()):
    #         break
    #     last = df.iloc[-1].to_list()[1]
    #     if(last == now):
            
    #         break
    #     else:
    #         df = realTimeData(datastore)
    #         continue

    # df.to_csv(os.path.join(args.refined_data, 'refined_data.csv'), index=False)
    # ds = Dataset.File.upload_directory(src_dir=args.refined_data,
    #         target=DataPath(datastore,  'dataset/mrinmoy/refined_data'),
    #         show_progress=True, overwrite=True)

    url = 'http://127.0.0.1:8000/predict'

    data = {
    'smoothing_trend' : 0.06999999999999999, 
    'smoothing_seasonal' : 0.05,
    'smoothing_level' :  0.39,
    }

    load = json.dumps(data)
    response = requests.post(url, data=load)
    response_json = json.loads(response.json())
    predictions = pd.DataFrame(response_json['data'], columns=response_json['columns'])
    predictions['t'] = pd.to_datetime(predictions['t'])
    t = pd.to_datetime(predictions['t'].values[0])
    c = predictions['predictions'].values[0]

    try:
        dataset = Dataset.Tabular.from_delimited_files(path = (datastore, 'dataset/mrinmoy/preds/predictions.csv'))
        print(1)
        pred_df = dataset.to_pandas_dataframe() 
        pred_df.drop(columns=pred_df.columns[0], axis=1, inplace=True)
        pred_df.loc[len(pred_df)] = t, c
    except:
        print(2)
        pred_df = pd.DataFrame({'t':t, 'c': c}, index = pd.RangeIndex(start=0, step=1, stop=1))
    print(pred_df)
    
    pred_df.to_csv(os.path.join(args.predictions, 'predictions.csv'))
    

    ds = Dataset.File.upload_directory(src_dir=args.predictions,
            target=DataPath(datastore,  'dataset/mrinmoy/preds'),
            show_progress=True, overwrite=True)

    

if __name__ == "__main__":
    main()


