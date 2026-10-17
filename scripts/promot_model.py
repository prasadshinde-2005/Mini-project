import os
import mlflow
from mlflow.exceptions import MlflowException

MODEL_NAME = "spmodel"
NEW_ALIAS = "challenger"   # naya version jis alias pe hai (agar tumhara alias "candidate" hai to yahan badlo)


def promote_model():
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")

    client = mlflow.MlflowClient()

    # 1. naya version nikalo
    try:
        new_version = client.get_model_version_by_alias(MODEL_NAME, NEW_ALIAS).version
    except MlflowException as e:
        raise RuntimeError(f"No model version has alias '{NEW_ALIAS}': {e}")

    # 2. purana champion nikalo (pehli baar run pe nahi hoga)
    try:
        old_champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    except MlflowException:
        old_champion = None

    if old_champion == new_version:
        print(f"Version {new_version} is already 'champion'. Nothing to do.")
        return

    # 3. naya version -> champion
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    print(f"Version {new_version} is now 'champion'")

    # 4. purana champion -> candidate
    if old_champion is not None:
        client.set_registered_model_alias(MODEL_NAME, "candidate", old_champion)
        print(f"Old champion (version {old_champion}) is now 'candidate'")
    else:
        print("No previous champion found, so no 'candidate' alias was set.")


if __name__ == "__main__":
    promote_model()