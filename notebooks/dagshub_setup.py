import dagshub
import mlflow

dagshub.set_tracking_uri = "https://dagshub.com/prasadshinde-2005/Mini-project.mlflow"
dagshub.init(repo_owner='prasadshinde-2005', repo_name='Mini-project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
