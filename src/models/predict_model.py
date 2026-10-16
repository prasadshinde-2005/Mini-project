import os
import sys
import json
import pickle
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


token = os.getenv("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")

# ---------------- DagsHub / MLflow setup ----------------
# dagshub.init(mlflow=True) login bhi handle karta hai aur MLflow tracking URI bhi set karta hai.
dagshub.init(repo_owner="prasadshinde-2005", repo_name="Mini-project", mlflow=True)

EXPERIMENT_NAME = "new_sp-dvc-pipeline"


# ---------------- Logging Configuration ----------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------- Load Test Data ----------------

def read_data(data_path):
    try:
        logger.info("Loading test data from %s", data_path)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data not found: {data_path}")

        data = pd.read_csv(data_path)

        if data.empty:
            raise ValueError("Test dataset is empty.")

        logger.info("Test data loaded successfully.")
        logger.debug("Test data shape: %s", data.shape)

        return data

    except Exception:
        logger.exception("Failed to load test data.")
        raise


# ---------------- Separate Features and Labels ----------------

def separate_data(test_data):
    try:
        logger.info("Separating features and target.")

        if "label" not in test_data.columns:
            raise ValueError("Target column 'label' is missing.")

        x_test = test_data.drop(columns=["label"]).values
        y_test = test_data["label"].values

        if len(x_test) == 0:
            raise ValueError("Test features are empty.")

        if pd.isna(y_test).any():
            raise ValueError("Test labels contain missing values.")

        logger.info("Features and labels separated successfully.")
        logger.debug("X_test shape: %s", x_test.shape)
        logger.debug("Y_test shape: %s", y_test.shape)
        logger.debug("Unique test labels: %s", set(y_test))

        return x_test, y_test

    except Exception:
        logger.exception("Failed to separate test data.")
        raise


# ---------------- Load Trained Model ----------------

def load_model(model_path):
    try:
        logger.info("Loading trained model from %s", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Only load pickle files from trusted sources.
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully.")
        logger.debug("Model classes: %s", model.classes_)

        return model

    except Exception:
        logger.exception("Failed to load model.")
        raise


# ---------------- Make Predictions ----------------

def predict(model, x_test):
    try:
        logger.info("Making predictions.")

        y_pred = model.predict(x_test)

        # In data_ingestion.py:
        # happiness = 1
        # sadness = 0
        positive_class = 1

        if positive_class not in model.classes_:
            raise ValueError(
                "Positive class 1 (happiness) is missing from the trained model."
            )

        y_pred_proba_all = model.predict_proba(x_test)

        # Probability column corresponding to class 1
        positive_index = list(model.classes_).index(positive_class)
        y_pred_proba = y_pred_proba_all[:, positive_index]

        logger.info("Predictions completed successfully.")
        logger.debug("Prediction count: %d", len(y_pred))

        return y_pred, y_pred_proba

    except Exception:
        logger.exception("Prediction failed.")
        raise


# ---------------- Calculate Evaluation Metrics ----------------

def calculate_metrics(y_test, y_pred, y_pred_proba):
    try:
        logger.info("Calculating evaluation metrics.")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

        # AUC requires both classes to be present in y_test.
        if len(set(y_test)) < 2:
            raise ValueError(
                "ROC AUC cannot be calculated because the test data "
                "contains only one class."
            )

        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
        }

        logger.info("Metrics calculated successfully.")
        logger.debug("Metrics: %s", metrics_dict)

        return metrics_dict

    except Exception:
        logger.exception("Failed to calculate metrics.")
        raise


# ---------------- Save Metrics ----------------

def save_metrics(metrics, output_path):
    try:
        logger.info("Saving metrics to %s", output_path)

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved successfully.")

    except Exception:
        logger.exception("Failed to save metrics.")
        raise


# ---------------- Log everything to MLflow ----------------

def log_to_mlflow(model, metrics, metrics_path, model_path):
    """Metrics, params, artifacts aur model MLflow mein log karta hai.
    Model ka URI return karta hai (registry ke liye)."""

    mlflow.log_metrics(metrics)
    mlflow.log_params(model.get_params())

    # Plain files: ye run ke Artifacts tab mein dikhengi
    mlflow.log_artifact(metrics_path)   # metrics.json
    mlflow.log_artifact(model_path)     # model.pkl

    # MLflow-format model. MLflow 3 mein ye Artifacts tab mein nahi,
    # Logged models / Models tab mein dikhta hai.
    # try/except jaan-bujhkar nahi lagaya: fail ho toh asli error dikhe
    # aur galat exp_info.json na bane.
    model_info = mlflow.sklearn.log_model(model, name="model")
    logger.info("Model logged to MLflow: %s", model_info.model_uri)

    return model_info.model_uri


# ---------------- Save Model Info (for model registry) ----------------

def save_model_info(run_id: str, model_path: str, model_uri: str, file_path: str) -> None:
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)   # reports/ folder na ho toh bana do

        model_info = {
            "run_id": run_id,
            "model_path": model_path,
            "model_uri": model_uri,
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(model_info, file, indent=4)

        logger.info("Model info saved to %s", file_path)

    except Exception:
        logger.exception("Failed to save model info.")
        raise


# ---------------- Main Function ----------------

def main():
    # Paths relative to the project root (script project root se chalao)
    test_data_path = "data/feature/test_bow.csv"
    model_path = "model.pkl"
    metrics_path = "metrics.json"
    model_info_path = "reports/exp_info.json"

    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        # `with` use kiya taaki error aane par bhi run sahi se band ho
        with mlflow.start_run() as run:
            logger.info("Starting model evaluation.")

            # Load data and trained model
            test_data = read_data(test_data_path)
            x_test, y_test = separate_data(test_data)
            model = load_model(model_path)

            # Predict and evaluate
            y_pred, y_pred_proba = predict(model, x_test)
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

            # Save metrics locally (DVC isi file ko track karta hai)
            save_metrics(metrics, metrics_path)

            # Log everything to MLflow / DagsHub
            model_uri = log_to_mlflow(model, metrics, metrics_path, model_path)

            # Sirf tab save hoga jab upar sab kuch successfully ho gaya
            save_model_info(run.info.run_id, "model", model_uri, model_info_path)

            logger.info("Model evaluation completed successfully.")

    except Exception:
        logger.exception("Model evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()