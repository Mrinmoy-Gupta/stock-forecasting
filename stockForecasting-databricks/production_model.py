# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

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

rmse_nb,rmse_tft

# COMMAND ----------

if rmse_tft<rmse_nb:
    experiment_id = ''
    name = "name='tft_model'"
else:
    experiment_id = ''
    name = "name='darts_nbeats'"
    

# COMMAND ----------


runs = mlflow.search_runs(experiment_ids=experiment_id)
run_id=runs.iloc[0]['run_id']
for mv in client.search_model_versions(name):
    if (dict(mv)['run_id']) == run_id:
        version_number = dict(mv)['version']
        model_name = dict(mv)['name']
print(version_number)
print(model_name)
client.transition_model_version_stage(
    name=model_name,
    version=version_number,
    stage="production",
)

# COMMAND ----------

