# Databricks notebook source
import pandas as pd
from databricks import feature_store

# COMMAND ----------

fs= feature_store.FeatureStoreClient()

# COMMAND ----------

def datapreprocessing():
    spark_df=spark.read.table("""Table name""")
    df = spark_df.toPandas()
    df=df.sort_values(by='t')
    df['t'] = df['t'] + pd.Timedelta(hours=5.5)
    df=df.drop_duplicates('t',keep='first') ###dropping duplicates timestamp if any on basis of timestamps
    df.drop(['s', 'o','h','l','v'], axis=1,inplace=True)
    start_date=df.iloc[0].tolist()
    end_date=df.iloc[-1].tolist()
    duplicate_df = pd.date_range(start_date[0],end_date[0], freq='T')
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
    preprocessed_spark_df=spark.createDataFrame(new_final_df) 
    fs.write_table(
  name="""Table name""",
  df = preprocessed_spark_df,
  mode = 'overwrite'
)
    print('DataPreprocessing Successfull, Feature Store Created -"""Table name"""')

# COMMAND ----------

datapreprocessing()
