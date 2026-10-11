import pandas as pd
import yaml
import pickle
import logging
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load parameters
def parameter(params_path):
    try:
        logger.debug("Loading model parameters from %s", params_path)

        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        C = params["train_model"]["C"]
        penalty = params["train_model"]["penalty"]
        solver = params["train_model"]["solver"]

        if isinstance(C, bool) or not isinstance(C, (int, float)) or C <= 0:
            raise ValueError("C must be a positive number.")

        if penalty not in ("l1", "l2", "elasticnet"):
            raise ValueError("penalty must be one of 'l1', 'l2', 'elasticnet'.")

        valid_solvers = {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
        if solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}.")
                
        logger.info("Model parameters loaded successfully.")
        logger.debug("C: %s", C)
        logger.debug("penalty: %s", penalty)
        logger.debug("solver: %s", solver)

        return C, penalty, solver

    except FileNotFoundError:
        logger.error("params file not found: %s", params_path)
        raise
    except KeyError as e:
        logger.error("Missing key in params file: %s", e)
        raise

# Load data
def read_data(url):
    try:
        logger.debug("Loading training data from %s", url)

        train_data = pd.read_csv(url)

        if train_data.empty:
            raise ValueError("Training dataset is empty.")

        logger.info("Training data loaded successfully.")
        logger.debug("Training data shape: %s", train_data.shape)

        return train_data

    except Exception:
        logger.exception("Failed to load training data.")
        raise


# Separate features and target
def separate_data(train_data):
    try:
        logger.debug("Separating features and target")

        if train_data.shape[1] < 2:
            raise ValueError(
                "Training data must contain features and a target column."
            )

        if "label" not in train_data.columns:
            raise ValueError("Target column 'label' is missing.")

        x_train = train_data.drop(columns=["label"]).values
        y_train = train_data["label"].values

        if len(x_train) != len(y_train):
            raise ValueError("Features and labels have different lengths.")

        if pd.isna(y_train).any():
            raise ValueError("Target column contains missing values.")

        logger.info("Features and target separated successfully.")
        logger.debug("X_train shape: %s", x_train.shape)
        logger.debug("y_train shape: %s", y_train.shape)
        logger.debug("Target classes: %s", pd.unique(y_train))

        return x_train, y_train

    except Exception:
        logger.exception("Failed to separate features and target.")
        raise


# Train model
def make_model(x_train, y_train, C , penalty , solver):
    try:
        logger.info("Starting model training...")
        logger.debug("Training samples: %s", len(x_train))
        logger.debug("Number of features: %s", x_train.shape[1])

        if len(x_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty.")

        if len(pd.unique(y_train)) < 2:
            raise ValueError(
                "Training requires at least two target classes."
            )

        clf = LogisticRegression(
            C = 1,
            penalty = "l2",
            solver = "liblinear" ,
            random_state=42
        )

        clf.fit(x_train, y_train)

        logger.info("Model training completed successfully.")
        logger.debug("Model classes: %s", clf.classes_)

        return clf

    except Exception:
        logger.exception("Model training failed.")
        raise


# Save model
def save_model(LR):
    try:
        model_path = "model.pkl"

        logger.debug("Saving model to %s", model_path)

        with open(model_path, "wb") as file:
            pickle.dump(LR, file)

        logger.info("Model saved successfully at %s", model_path)

    except Exception:
        logger.exception("Failed to save model.")
        raise


# Main function
def main():
    try:
        logger.info("Starting model building...")

        C , penalty ,solver = parameter("params.yaml")

        train_data = read_data("./data/feature/train_tfidf.csv")

        x_train, y_train = separate_data(train_data)

        LR = make_model(
            x_train, y_train, C , penalty , solver
        )

        save_model(LR)

        logger.info("Model building completed successfully.")

    except Exception:
        logger.exception("Model building pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
