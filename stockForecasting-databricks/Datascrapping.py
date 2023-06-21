# Databricks notebook source
# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS """Table name""";

# COMMAND ----------

import time
import http.client, json
import pandas as pd
from datetime import datetime
from dateutil import parser
import dateutil.relativedelta
import csv
import os
from databricks import feature_store
from pytz import timezone 
from datetime import datetime

# COMMAND ----------

def datetotimestamp(date):
    time_tuple = date.timetuple()
    timestamp = round(time.mktime(time_tuple))
    return timestamp

def timestamptodate(timestamp):
    return datetime.fromtimestamp(timestamp)

# COMMAND ----------

##LOGIC DATASCRAPPING
tables=spark.sql("""Table name""")
df=tables.toPandas()
table_list=df['tableName'].to_list()

if 'raw_data' not in table_list:
    print("*******************NO PREVIOUS DATA OF STOCK FOUND, COLLECTING DATA FOR ENTIRE YEAR*******************")
    date = datetime.today()+pd.Timedelta(hours=5.5)
    print(date)
    prev_yr_date = (date + dateutil.relativedelta.relativedelta(months=-12))
    print(prev_yr_date)
    start = datetotimestamp(prev_yr_date)
    end = datetotimestamp(date)
    c=date-prev_yr_date
    minutes = c.total_seconds() // 60
    count_back=int(minutes)
    
    url1 = ''
    conn = http.client.HTTPSConnection("")
    payload = ''
    headers = {}
    conn.request("GET", url1, payload, headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data.decode("utf-8"))
    data_final=pd.DataFrame(response)
    data_final['t'] = data_final['t'].apply(timestamptodate)
    ##THE RESPONSE RETURNED BY THE URL HAS DATA IN US TIME. AND WE WILL CONVERT TO IST TIME IN DATAPREPROCESSING
    sparkDF=spark.createDataFrame(data_final)
    sparkDF.write.format("delta").saveAsTable(""" Table Name """)

else:
    print('++++++++++++++++++++++DATA IS ALREADY PRESENT +++++++++++++++++++++++++++++')
    temp_df = spark.read.table("""Table name""")
    pandas_df=temp_df.toPandas()
    result=pandas_df.sort_values(by='t')
    temp=result.iloc[-1].tolist()
    start = temp[1]+pd.Timedelta(hours=5.5)
    end = datetime.today()+pd.Timedelta(hours=5.5)
    c=end-start
    minutes = c.total_seconds() // 60
    count_back=int(minutes)
    end_date=datetotimestamp(end)
    start_date=datetotimestamp(start)
    url1 = ''
    conn = http.client.HTTPSConnection("")
    payload = ''
    headers = {}
    conn.request("GET", url1, payload, headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data.decode("utf-8"))
    data_final_1=pd.DataFrame(response)
    data_final_1['t'] = data_final_1['t'].apply(timestamptodate)
    sparkDF=spark.createDataFrame(data_final_1)
    sparkDF.write.mode("append").format("delta").saveAsTable("""Table name""")
## NEED TO REMOVE DUPLICATE ROWS PRESENT IN feature_store_stocks.Raw_Data
    
    

# COMMAND ----------

spark_df = spark.read.format("delta").load("""Table name""")

# COMMAND ----------

pandas_df = spark_df.toPandas()
display(pandas_df)