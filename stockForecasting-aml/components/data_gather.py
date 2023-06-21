import time
import http.client, json
import pandas as pd
from datetime import datetime
from dateutil import parser
import dateutil.relativedelta
import numpy as np
import os
from pytz import timezone 
import argparse
import logging
import mlflow
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath

def datetotimestamp(date):
    time_tuple = date.timetuple()
    timestamp = round(time.mktime(time_tuple))
    return timestamp

def timestamptodate(timestamp):
    return datetime.fromtimestamp(timestamp)

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_data", type=str, help="path to output data data")
    
    args = parser.parse_args()
    print('output path......', args.output_data)
    ws = Workspace.get(name="AML-AWH-AZPOC",
               subscription_id='ab8d90b6-c993-497e-aa6b-033721a4ccb3',
               resource_group='HANU-ROHIT-RG')

    datastore = Datastore.get(ws, 'workspaceblobstore')

    # try:
    # dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'dataset/mrinmoy/data.csv'))
    # print("++++DATA IS ALREADY PRESENT ++++")
    # pandas_df = dataset.to_pandas_dataframe() 
    # result = pandas_df.sort_values(by="t")
    # temp = result.iloc[-1].tolist()
    # start = temp[1]
    # end = datetime.today() + pd.Timedelta(hours=5.5)
    # c = end - start
    # minutes = c.total_seconds() // 60
    # count_back = int(minutes)
    # end_date = datetotimestamp(end)
    # start_date = datetotimestamp(start)
    # url1 = '/techCharts/indianMarket/stock/history?symbol=TATAELXSI&resolution=1&from='+str(start_date)+'&to='+str(end_date)+'&countback='+str(count_back)+'&currencyCode=INR'
    # conn = http.client.HTTPSConnection("priceapi.moneycontrol.com")
    # payload = ""
    # headers = {}
    # conn.request("GET", url1, payload, headers)
    # res = conn.getresponse()
    # data = res.read()
    # response = json.loads(data.decode("utf-8"))
    # actual_df = pd.DataFrame(response)
    # actual_df.drop(['s', 'o','h','l','v'], axis=1,inplace=True)
    # actual_df["t"] = actual_df["t"].apply(timestamptodate)
    # actual_df["t"] = actual_df["t"] + pd.Timedelta(hours=5.5)
    # actual_df=actual_df.drop_duplicates('t',keep='first')
    # result = result.drop(columns=['Column1'])
    # df = pd.concat([result, actual_df], ignore_index=False)
    # df.to_csv(os.path.join(args.output_data, 'data.csv'), index=False)
    # print(df)

    # except:
    print("****NO PREVIOUS DATA OF STOCK FOUND, COLLECTING DATA FOR ENTIRE YEAR****")
    date = datetime.today() + pd.Timedelta(hours=5.5)
    print(date)
    prev_yr_date = date + dateutil.relativedelta.relativedelta(months=-12)
    print(prev_yr_date)
    start = datetotimestamp(prev_yr_date)
    end = datetotimestamp(date)
    c = date - prev_yr_date
    minutes = c.total_seconds() // 60
    count_back = int(minutes)
    url1 = '/techCharts/indianMarket/stock/history?symbol=TATAELXSI&resolution=1&from='+str(start)+'&to='+str(end)+'&countback='+str(count_back)+'&currencyCode=INR'
    conn = http.client.HTTPSConnection("priceapi.moneycontrol.com")
    payload = ""
    headers = {}
    conn.request("GET", url1, payload, headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data.decode("utf-8"))
    actual_df = pd.DataFrame(response)
    actual_df["t"] = actual_df["t"].apply(timestamptodate)
    actual_df["t"] = actual_df["t"] + pd.Timedelta(hours=5.5)
    actual_df.drop(['s', 'o','h','l','v'], axis=1,inplace=True)
    actual_df.to_csv(os.path.join(args.output_data,'data.csv'), index=False)
    print(actual_df)

        
    ds = Dataset.File.upload_directory(src_dir= args.output_data,
            target=DataPath(datastore,  'dataset/mrinmoy'),
            show_progress=True, overwrite=True)

if __name__ == "__main__":
    main()
