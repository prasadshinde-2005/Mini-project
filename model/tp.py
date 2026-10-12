import dagshub, mlflow
from mlflow.tracking import MlflowClient

dagshub.init(repo_owner="prasadshinde-2005", repo_name="Mini-project", mlflow=True)
run_id = "3cff04e61d5141d4895e4847f78ef812"
client = MlflowClient()

print("Run artifacts:", [a.path for a in client.list_artifacts(run_id)])

try:
    exp_id = client.get_run(run_id).info.experiment_id
    print(mlflow.search_logged_models(experiment_ids=[exp_id]))
except Exception as e:
    print("Logged models lookup failed:", e)