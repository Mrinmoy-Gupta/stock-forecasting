# Databricks notebook source
dbutils.widgets.text("Epochs_tft", "", "epochs_tft")
dbutils.widgets.text("Attheads_tft", "", "attheads_tft")
dbutils.widgets.text("Dropout_tft", "", "dropout_tft")
dbutils.widgets.text("Batchsize_tft", "", "batchsize_tft")

# COMMAND ----------

num_epochs_tft = str(dbutils.widgets.get("Epochs_tft"))
num_attheads_tft = str(dbutils.widgets.get("Attheads_tft"))
dropout_ratio_tft = str(dbutils.widgets.get("Dropout_tft"))
batch_size_tft = str(dbutils.widgets.get("Batchsize_tft"))

# COMMAND ----------

##IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler ### (look after scaler)
from darts.models import TFTModel 
from darts.metrics import mape, mae, rmse,r2_score
from darts.utils.likelihood_models import QuantileRegression
from databricks import feature_store

from mlflow.tracking import MlflowClient
import mlflow

# COMMAND ----------

# # default quantiles for QuantileRegression
QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

qL1, qL2, qL3 = 0.01, 0.05, 0.10        # percentiles of predictions: lower bounds
qU1, qU2, qU3 = 1-qL1, 1-qL2, 1-qL3     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
label_q3 = f'{int(qU3 * 100)} / {int(qL3 * 100)} percentile band'

# COMMAND ----------

N_FC = 36           # default forecast horizon
RAND = 42           # set random state
N_SAMPLES = 1     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1          # parallel processors to use;  -1 = all processors

# COMMAND ----------

print(int(num_epochs_tft))
print(int(num_attheads_tft))
print(float(dropout_ratio_tft))
print(int(batch_size_tft))

# COMMAND ----------

EPOCHS = int(num_epochs_tft)
ATTHEADS = int(num_attheads_tft)
DROPOUT = float(dropout_ratio_tft)              ## dropout ratio to avoid overfitting
BATCH = int(batch_size_tft)               ##number of attention heads
INLEN = 32                 ##size (node count) of the input layer
HIDDEN = 64                ## Number of hidden layers  
LSTMLAYERS = 1

# COMMAND ----------

# ## MODEL PARAMETERS
# EPOCHS = 60
# INLEN = 32      ##size (node count) of the input layer
# HIDDEN = 64      ## Number of hidden layers  
# LSTMLAYERS = 1
# ATTHEADS = 4  ##number of attention heads
# DROPOUT = 0.25  ## dropout ratio to avoid overfitting
# BATCH = 64

# COMMAND ----------

fs= feature_store.FeatureStoreClient()

# COMMAND ----------

spark_df = fs.read_table("""Table name""")

# COMMAND ----------

df = spark_df.toPandas()

# COMMAND ----------

# if the source is a dataframe: create a time series object from a dataframe's column
ts = TimeSeries.from_dataframe(df,'t','c',fill_missing_dates=False, freq=None,fillna_value=None) ##freq check for explicitly mention

# COMMAND ----------

# train/test
TRAIN = 0.8
if isinstance(TRAIN, str):
    split = pd.Timestamp(TRAIN)
else:
    split = TRAIN
ts_train, ts_test = ts.split_after(split)

# scale the time series on the training settransformer = Scaler()
## Neural networks are sensitive to variations in magnitude.  therefore we scale our training data.

transformer = Scaler()
transformer = transformer.fit(ts_train)
#save the scaler locally using the package pickel 

# import pickle

# pickle.dump(transformer, open('transformer.pkl', 'wb')) 
ts_ttrain = transformer.fit_transform(ts_train)
ts_ttest = transformer.transform(ts_test)
ts_t = transformer.transform(ts)

# COMMAND ----------

ts_ttrain.end_time()

# COMMAND ----------

ts_ttest.start_time()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS feature_store_stocks.predictions1;

# COMMAND ----------

mlflow.end_run()
mlflow.set_experiment("""URI""")
# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run(run_name="tft_model_training") 
tft_params = {
    'EPOCHS' : EPOCHS,
    'INLEN' :INLEN ,     
    'HIDDEN' : HIDDEN ,    
    'LSTMLAYERS' : LSTMLAYERS,
    'ATTHEADS' : ATTHEADS  ,
    'DROPOUT' :DROPOUT  ,
    'BATCH' :BATCH,
    'N_FC' :N_FC  ,      
    'RAND' : RAND   ,       
    'N_SAMPLES' : N_SAMPLES,     
    'N_JOBS' : N_JOBS
}
mlflow.log_param("hyper-parameters", tft_params)

model = TFTModel(   input_chunk_length=INLEN,
                    output_chunk_length=N_FC,
                    hidden_size=HIDDEN,
                    lstm_layers=LSTMLAYERS,
                    num_attention_heads=ATTHEADS,
                    dropout=DROPOUT,
                    batch_size=BATCH,
                    n_epochs=EPOCHS,
                    add_relative_index=True,
                    likelihood=QuantileRegression(quantiles=QUANTILES), 
                    # loss_fn=MSELoss(),
                    random_state=RAND,
                    pl_trainer_kwargs = {"accelerator": "gpu"},
                    force_reset=True)

model.fit(ts_ttrain,verbose=True)
# testing: generate predictions
ts_tpred = model.predict(   n=len(ts_ttest), 
                            num_samples=1,   
                            n_jobs=N_JOBS)
ts_pred = transformer.inverse_transform(ts_tpred)
duplicate_df = pd.date_range(ts_pred.start_time(),ts_pred.end_time(), freq='T')
pred=TimeSeries.pd_series(ts_pred)
df_final= pd.DataFrame({ 't': duplicate_df, 'pred': pred }) 
sparkDF=spark.createDataFrame(df_final)
sparkDF.write.format("delta").saveAsTable("""Table name""")


mlflow.sklearn.log_model(model,artifact_path='tft_model',
                             registered_model_name = 'tft_model')


result_mape=mape(ts_test, ts_pred)
print(result_mape)
mlflow.log_metric('mape',result_mape)

result_mae=mae(ts_test, ts_pred)
print(result_mae)
result_rmse = rmse(ts_test, ts_pred)
print(result_rmse)
result_r2=r2_score(ts_test, ts_pred)
mlflow.log_metric('mae',result_mae)
mlflow.log_metric('rmse',result_rmse)
mlflow.log_metric('r2_score',result_r2)           
            

# COMMAND ----------

ts_tpred_future = model.predict(   n=(len(ts_test) + 10000), 
                            num_samples=N_SAMPLES,   
                            n_jobs=N_JOBS)
# inverse transform
ts_pred_future = transformer.inverse_transform(ts_tpred_future)

df_predictions = pd.DataFrame({ "predicted":TimeSeries.pd_series(ts_pred_future)}, index = TimeSeries.pd_series(ts_pred_future).index)
df_predictions.reset_index(inplace=True)

# COMMAND ----------

dash = pd.DataFrame()
dash['t'] = df_predictions['t']
dash['c_preds'] = np.NAN
dash['c_actual'] = np.NAN
dash.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS feature_store_stocks.final_preds_tft;

# COMMAND ----------

sparkDF_final=spark.createDataFrame(df_predictions)
sparkDF_final.write.format("delta").saveAsTable("""Table name""")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS feature_store_stocks.dashboard_predictions;

# COMMAND ----------

sparkDF_dash=spark.createDataFrame(dash)
sparkDF_dash.write.format("delta").saveAsTable("""Table name""")