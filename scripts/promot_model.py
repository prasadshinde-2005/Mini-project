import os 
import mlflow 


def promote_model():
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
                "https://dagshub.com/prasadshinde-2005/Mini-project.mlflow"
            )

    client = mlflow.MlflowClient()
    model_name = "spmodel"
    latest_version_aliased = client.get_latest_versions(model_name , alias =["challenger"])[0].version 

    prod_version = client.get_latest_versions(model_name , alias = ["champion"])
    for version in prod_version:
        client.transition_model_version_stage(
            name = model_name , 
            version = version.version ,
            alias = "candidate"

        )

        client.transition_model_version_stage(
                    name = model_name , 
                    version = latest_version_aliased ,
                    alias = "champion"
        
                )

        print(f'model version{latest_version_aliased}promoted to production')

if __name__ =='__main__':
    promote_model()
