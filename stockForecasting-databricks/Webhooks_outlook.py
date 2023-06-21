# Databricks notebook source
from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec, HttpUrlSpec

# COMMAND ----------

access_token='' ##store in key vault or store it in dbfs
model_name = ''
# job_id = 0 # INSERT ID OF PRE-DEFINED JOB
mail_url = ''
# COMMAND ----------

# Create a HTTP webhook that will create alerts about registered models created
http_url_spec = HttpUrlSpec(url=mail_url, secret="secret_string")
http_webhook = RegistryWebhooksClient().create_webhook(
  events=["TRANSITION_REQUEST_CREATED", "MODEL_VERSION_CREATED", "MODEL_VERSION_TRANSITIONED_STAGE"],
  http_url_spec=http_url_spec,
  model_name=model_name
)
http_webhook

# COMMAND ----------

# Test the HTTP webhook
RegistryWebhooksClient().test_webhook(id=http_webhook.id)