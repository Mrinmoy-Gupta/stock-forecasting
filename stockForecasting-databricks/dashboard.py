# Databricks notebook source
import time
import http.client, json
from datetime import datetime, timedelta
from dateutil import parser
import dateutil.relativedelta
from databricks import feature_store
from datetime import datetime
import pandas as pd
import pytz

# COMMAND ----------

def datetotimestamp(date):
    time_tuple = date.timetuple()
    timestamp = round(time.mktime(time_tuple))
    return timestamp

def timestamptodate(timestamp):
    return datetime.fromtimestamp(timestamp)

# COMMAND ----------

def realTimeData(verbose=True):
    ##LOGIC DATASCRAPPING
    tables = spark.sql("""Table name""")
    df = tables.toPandas()
    table_list = df["tableName"].to_list()

    if "real_time_data_dashboards" not in table_list:
        if verbose:
            print("*******************NO PREVIOUS DATA OF STOCK FOUND, COLLECTING DATA FOR ENTIRE YEAR*******************")
        date = datetime.today() + pd.Timedelta(hours=5.5)
        if verbose:
            print(date)
        prev_yr_date = date + dateutil.relativedelta.relativedelta(months=-12)
        if verbose:
            print(prev_yr_date)
        start = datetotimestamp(prev_yr_date)
        end = datetotimestamp(date)
        c = date - prev_yr_date
        minutes = c.total_seconds() // 60
        count_back = int(minutes)

        url1 = ''
        conn = http.client.HTTPSConnection("")
        payload = ""
        headers = {}
        conn.request("GET", url1, payload, headers)
        res = conn.getresponse()
        data = res.read()
        response = json.loads(data.decode("utf-8"))
        data_final = pd.DataFrame(response)
        data_final["t"] = data_final["t"].apply(timestamptodate)
        data_final["t"] = data_final["t"] + pd.Timedelta(hours=5.5)  ##converting timestamps acc. to ist zone.
        ##THE RESPONSE RETURNED BY THE URL HAS DATA IN US TIME. AND WE WILL CONVERT TO IST TIME IN DATAPREPROCESSING
        sparkDF = spark.createDataFrame(data_final)
        sparkDF.write.format("delta").saveAsTable("""Tble name""")
    else:
        if verbose:
            print("++++++++++++++++++++++DATA IS ALREADY PRESENT +++++++++++++++++++++++++++++")
        temp_df = spark.read.table("""Table name""")
        pandas_df = temp_df.toPandas()
        result = pandas_df.sort_values(by="t")
        temp = result.iloc[-1].tolist()
        start = temp[1]
        end = datetime.today() + pd.Timedelta(hours=5.5)
        c = end - start
        minutes = c.total_seconds() // 60
        count_back = int(minutes)
        end_date = datetotimestamp(end)
        start_date = datetotimestamp(start)
        url1 = ''
        conn = http.client.HTTPSConnection("")
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
        if verbose:
            print(actual_df.head())
        return actual_df

# COMMAND ----------

actual_df = realTimeData()

# COMMAND ----------

import mlflow
experiment_id_nb = ''
runs_nb = mlflow.search_runs(experiment_ids=experiment_id_nb)
run_id_nb=runs_nb.iloc[0]['run_id']
data_nb,_ = mlflow.get_run(run_id_nb)
rmse_nb = data_nb[1].metrics['rmse']

experiment_id_tft = ''
runs_tft= mlflow.search_runs(experiment_ids=experiment_id_tft)
run_id_tft=runs_tft.iloc[0]['run_id']
data_tft, info = mlflow.get_run(run_id_tft)
rmse_tft = data_tft[1].metrics['rmse']

rmse_nb, rmse_tft

# COMMAND ----------

if rmse_tft<rmse_nb:
    prediction_table_name = """Table name"""
else:
    prediction_table_name = """Table name"""

# COMMAND ----------

temp=spark.read.table("""Table name""")
dash=temp.toPandas()
dash.sort_values(by="t").tail()

# COMMAND ----------

df_spark = spark.read.table(prediction_table_name)
predicted_df = df_spark.toPandas()
predicted_df.sort_values(by='t').tail()

# COMMAND ----------

indiatmz = pytz.timezone('Asia/Kolkata')
now_for_actual = (datetime.now(indiatmz)).strftime("%Y-%m-%d %H:%M") + ":00"
now_for_predicted = (datetime.now(indiatmz) + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")+":00" 
now_for_actual, now_for_predicted

# COMMAND ----------

while(True):
    try:
        dash.loc[dash['t']==now_for_actual,'c_actual'] = actual_df[actual_df['t']==now_for_actual]['c'].values[0]
        dash.loc[dash['t']==now_for_predicted, 'c_preds'] = predicted_df[predicted_df['t']==now_for_predicted]['predicted'].values[0]
        break
    except:
        actual_df = realTimeData(False)

# COMMAND ----------

sparkDF_dash=spark.createDataFrame(dash)
sparkDF_dash.write.mode("overwrite").format("delta").saveAsTable("""Table name""")