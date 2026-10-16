import json
import logging
import os 
import dagshub
import mlflow
from mlflow.tracking import MlflowClient

# dagshub.init() login/authentication bhi handle karta hai (403 error se bachata hai)
dagshub.set_tracking_uri = "https://dagshub.com/prasadshinde-2005/Mini-project.mlflow"

dagshub.init(repo_owner='prasadshinde-2005', repo_name='Mini-project', mlflow=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_info(file_path: str) -> dict:
    with open(file_path, "r") as file:
        model_info = json.load(file)
    return model_info


def register_model(model_name: str, model_info: dict):
    # Evaluation script ne model ka seedha URI save kiya hai (best option).
    # Purani exp_info.json mein ye nahi hoga, tab runs:/ path par fallback.
    model_uri = model_info.get("model_uri") or (
        f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    )
    logger.info("Registering model from %s", model_uri)

    model_version = mlflow.register_model(model_uri, model_name)
    logger.info("Registered %s version %s", model_name, model_version.version)

    # "stage" deprecated hai, uski jagah alias use hota hai
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=model_version.version,
    )
    logger.info("Alias 'candidate' set on version %s", model_version.version)


def main():
    model_info_path = "reports/exp_info.json"
    model_info = load_model_info(model_info_path)

    model_name = "spmodel"
    register_model(model_name, model_info)


if __name__ == "__main__":
    main()