
import pandas as pd
import numpy as np
import os
import yaml
import logging
import sys

from sklearn.feature_extraction.text import CountVectorizer


# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load parameters
def load_params():
    try:
        logger.debug("Loading parameters from params.yaml")

        with open("params.yaml", "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        max_features = params["build_features"]["max_features"]

        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError("max_features must be a positive integer.")

        logger.info("Parameters loaded successfully.")
        logger.debug("Max features: %s", max_features)

        return max_features

    except Exception:
        logger.exception("Failed to load parameters.")
        raise


# Load processed data
def load_data():
    try:
        logger.debug("Loading processed train and test data")

        train_data = pd.read_csv("./data/processed/train_csv")
        test_data = pd.read_csv("./data/processed/test_csv")

        if train_data.empty or test_data.empty:
            raise ValueError("Train or test dataset is empty.")

        logger.info("Data loaded successfully.")
        logger.debug("Train shape: %s", train_data.shape)
        logger.debug("Test shape: %s", test_data.shape)

        return train_data, test_data

    except Exception:
        logger.exception("Failed to load processed data.")
        raise


# Handle missing values
def handle_missing_values(train_data, test_data):
    try:
        logger.debug("Handling missing values")

        required_columns = {"content", "sentiment"}

        for name, df in [("train", train_data), ("test", test_data)]:
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"{name} data is missing columns: {missing_columns}"
                )

        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data["content"] = train_data["content"].fillna("")
        test_data["content"] = test_data["content"].fillna("")

        train_data["sentiment"] = train_data["sentiment"].fillna("")
        test_data["sentiment"] = test_data["sentiment"].fillna("")

        logger.info("Missing values handled successfully.")

        return train_data, test_data

    except Exception:
        logger.exception("Failed to handle missing values.")
        raise


# Separate features and labels
def split_data(train_data, test_data):
    try:
        logger.debug("Separating features and labels")

        x_train = train_data["content"].astype(str).values
        y_train = train_data["sentiment"].values

        x_test = test_data["content"].astype(str).values
        y_test = test_data["sentiment"].values

        if len(x_train) == 0 or len(x_test) == 0:
            raise ValueError("Train or test features are empty.")

        logger.info("Features and labels separated.")
        logger.debug("X_train shape: %s", x_train.shape)
        logger.debug("X_test shape: %s", x_test.shape)

        return x_train, y_train, x_test, y_test

    except Exception:
        logger.exception("Failed to split data.")
        raise


# Apply Bag of Words
def apply_bow(x_train, x_test, max_features):
    try:
        logger.debug("Applying Bag of Words")

        vectorizer = CountVectorizer(max_features=max_features)

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        logger.info(" tfidf applied successfully.")
        logger.debug("Train tfidf shape: %s", x_train_bow.shape)
        logger.debug("Test tfidf shape: %s", x_test_bow.shape)
        logger.debug("Vocabulary size: %s", len(vectorizer.vocabulary_))

        return x_train_bow , x_test_bow

    except ValueError:
        logger.exception(
            "BoW failed. Check whether the text is empty or contains "
            "valid tokens."
        )
        raise

    except Exception:
        logger.exception("Unexpected error while applying BoW.")
        raise


# Create feature DataFrames
def create_feature_df(x_train_bow , y_train, x_test_bow , y_test):
    try:
        logger.debug("Creating feature DataFrames")

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        logger.info("Feature DataFrames created.")
        logger.debug("Train feature shape: %s", train_df.shape)
        logger.debug("Test feature shape: %s", test_df.shape)

        return train_df, test_df

    except Exception:
        logger.exception("Failed to create feature DataFrames.")
        raise


# Save feature data
def save_feature_data(train_df, test_df):
    try:
        data_path = os.path.join("data", "feature")

        logger.debug("Creating output directory: %s", data_path)
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train_tfidf.csv")
        test_path = os.path.join(data_path, "test_tfidf.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Train features saved to: %s", train_path)
        logger.info("Test features saved to: %s", test_path)

    except Exception:
        logger.exception("Failed to save feature data.")
        raise


# Main function
def main():
    try:
        logger.info("Starting feature engineering...")

        max_features = load_params()

        train_data, test_data = load_data()

        train_data, test_data = handle_missing_values(
            train_data, test_data
        )

        x_train, y_train, x_test, y_test = split_data(
            train_data, test_data
        )

        x_train_bow, x_test_bow = apply_bow(
            x_train, x_test, max_features
        )

        train_df, test_df = create_feature_df(
            x_train_bow, y_train, x_test_bow, y_test
        )

        save_feature_data(train_df, test_df)

        logger.info("Feature engineering completed successfully.")

    except Exception:
        logger.exception("Feature engineering failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()


