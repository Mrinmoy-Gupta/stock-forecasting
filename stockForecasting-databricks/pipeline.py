# Databricks notebook source
dbutils.widgets.text("Epochs_nb", "", "epochs_nb")
dbutils.widgets.text("Lwidth_nb", "", "lwidth_nb")
dbutils.widgets.text("Batchsize_nb", "", "batchsize_nb")

# COMMAND ----------

num_epochs_nb = str(dbutils.widgets.get("Epochs_nb"))
lwidth_nb = str(dbutils.widgets.get("Lwidth_nb"))
batch_size_nb = str(dbutils.widgets.get("Batchsize_nb"))

# COMMAND ----------

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

# MAGIC %run ./Datascrapping

# COMMAND ----------

# MAGIC %run ./DataPreprocessing_Bronzelayer

# COMMAND ----------

# MAGIC %run ./Webhooks_outlook

# COMMAND ----------

dbutils.notebook.run('<Notebook uri>',timeout_seconds=0,arguments={'Epochs_tft':num_epochs_tft,'Attheads_tft':num_attheads_tft,'Dropout_tft':dropout_ratio_tft,'Batchsize_tft':batch_size_tft})

# COMMAND ----------

dbutils.notebook.run('<Notebook uri>',timeout_seconds=0,arguments={'Epochs_nb':num_epochs_nb,'Lwidth_nb':lwidth_nb,'Batchsize_nb':batch_size_nb})

# COMMAND ----------

# MAGIC %run ./staging_model

# COMMAND ----------

# MAGIC %run ./production_model