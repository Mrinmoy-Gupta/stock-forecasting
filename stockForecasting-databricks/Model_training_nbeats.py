# Databricks notebook source
dbutils.widgets.text("Epochs_nb", "", "epochs_nb")
dbutils.widgets.text("Lwidth_nb", "", "lwidth_nb")
dbutils.widgets.text("Batchsize_nb", "", "batchsize_nb")

# COMMAND ----------

num_epochs_nb = str(dbutils.widgets.get("Epochs_nb"))
lwidth_nb = str(dbutils.widgets.get("Lwidth_nb"))
batch_size_nb = str(dbutils.widgets.get("Batchsize_nb"))

# COMMAND ----------

# Importing the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.metrics import mape, rmse, mae, r2_score
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

# COMMAND ----------

int(num_epochs_nb)

# COMMAND ----------

int(lwidth_nb)

# COMMAND ----------

int(batch_size_nb)

# COMMAND ----------

## MODEL PARAMETERS

EPOCHS = int(num_epochs_nb)
INLEN = 32                        # input size
BLOCKS = 32                       # the number of blocks in a stack 
LWIDTH = int(lwidth_nb)             # width of the layers in each block
BATCH =  int(batch_size_nb)          # batch size
LEARN = 1e-3                      # learning rate
VALWAIT = 1                       # epochs to wait before evaluating the loss on the test/validation set
N_FC = 1                          # output size

RAND = 42                         # random seed
N_SAMPLES = 1                     # number of times a prediction is sampled from a probabilistic model
N_JOBS = -1                       # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.90        # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'

label_q1, label_q2


# COMMAND ----------

#Importing the data which was scrapped and preprocessed
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()
df_spark = fs.read_table("""Table name""")
df = df_spark.toPandas()
df.tail()

# COMMAND ----------

df.plot(x='t', y='c')

# COMMAND ----------

# create a time series object from a dataframe's column
ts = TimeSeries.from_dataframe(df,'t','c',fill_missing_dates=True, freq=None,fillna_value=0) 

# COMMAND ----------

# Splitting the data into training and testing

ts_train, ts_test = ts.split_after(SPLIT)

# scale the time series on the training settransformer = Scaler()
# Neural networks are sensitive to variations in magnitude.  therefore we scale our training data.
transformer = Scaler()
transformer = transformer.fit(ts_train)

import pickle

pickle.dump(transformer, open('transformer.pkl', 'wb')) 
ts_ttrain = transformer.fit_transform(ts_train)
ts_ttest = transformer.transform(ts_test)
ts_t = transformer.transform(ts)


# COMMAND ----------

# printing the start and end time of the train and test dataset
print("train start: ", ts_train.start_time())
print("train end: ", ts_train.end_time())
print("test start: ", ts_test.start_time())
print("test end: ", ts_test.end_time())

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.end_run()
mlflow.set_experiment("""URI""")
# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run(run_name="darts_nbeats") 
    # recording the training params
NBEATS_params = {
    'EPOCHS' : EPOCHS,
    'INLEN' : INLEN ,
    'BLOCKS' : BLOCKS,
    'LWIDTH' : LWIDTH,
    'BATCH' : BATCH,
    'LEARN' : LEARN,
    'N_FC' : N_FC,
    'RAND' : RAND,       
    'N_SAMPLES' : N_SAMPLES,     
    'N_JOBS' : N_JOBS
}
# Logging the training params
mlflow.log_param("hyper-parameters", NBEATS_params)

# initializing the model
model = NBEATSModel( input_chunk_length=INLEN,
                    output_chunk_length=N_FC, 
                    num_stacks=BLOCKS,
                    layer_widths=LWIDTH,
                    batch_size=BATCH,
                    n_epochs=EPOCHS,
                    nr_epochs_val_period=VALWAIT, 
                    likelihood=QuantileRegression(QUANTILES), 
                    optimizer_kwargs={"lr": LEARN}, 
                    log_tensorboard=True,
                    generic_architecture=True, 
                    random_state=RAND,
                )
# training start on the scaled train data
model.fit(series=ts_ttrain, verbose=True)
# prediction of the scaled data
ts_tpred = model.predict(n=len(ts_ttest), 
                    num_samples=N_SAMPLES,
                    n_jobs=N_JOBS)

# inverse transforming the predicted data
ts_pred = transformer.inverse_transform(ts_tpred)

# Saving our predictions

# calculating metrices
result_mape = mape(ts_test, ts_pred)
result_mae = mae(ts_test, ts_pred)
result_rmse = rmse(ts_test, ts_pred)
result_r2 = r2_score(ts_test, ts_pred)
print("result_mape", result_mape)
print("result_mae", result_mae)
print("result_rmse", result_rmse)
print("result_r2", result_r2)
# logging our metrices in mlflow for tracking
mlflow.log_metrics({'rmse':result_rmse, 'mape':result_mape, 'r2_score':result_r2, 'mae':result_mae})

# saving our model
mlflow.sklearn.log_model(model,artifact_path='darts_nbeats', registered_model_name = 'darts_nbeats')
mlflow.log_artifact('transformer.pkl') 


# COMMAND ----------

ts_tpred_future = model.predict(   n=(len(ts_test) + 10000), 
                            num_samples=N_SAMPLES,   
                            n_jobs=N_JOBS)
# inverse transform
ts_pred_future = transformer.inverse_transform(ts_tpred_future)


# COMMAND ----------

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
# MAGIC DROP TABLE IF EXISTS feature_store_stocks.final_preds_nb;

# COMMAND ----------

sparkDF_final=spark.createDataFrame(df_predictions)
sparkDF_final.write.format("delta").saveAsTable("feature_store_stocks.final_preds_nb")

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS feature_store_stocks.dashboard_predictions_nb;

# COMMAND ----------

sparkDF_dash=spark.createDataFrame(dash)
sparkDF_dash.write.format("delta").saveAsTable("feature_store_stocks.dashboard_predictions")