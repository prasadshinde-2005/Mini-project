import dagshub
import pandas as pd
import os
import yaml
import logging
import sys


from urllib.error import URLError, HTTPError
from sklearn.model_selection import train_test_split


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load parameters
def load_params(params_path: str) -> float:
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        if not isinstance(params, dict):
            raise ValueError("Invalid YAML structure in params file.")

        test_size = params["make_dataset"]["test_size"]

        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError("test_size must be a number between 0 and 1.")

        logger.info("Parameters loaded successfully.")
        return float(test_size)

    except FileNotFoundError as e:
        logger.error("Parameter file not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("Invalid YAML format in %s", params_path)
        raise

    except (KeyError, TypeError) as e:
        logger.error("Missing or invalid data_ingestion.test_size in params.yaml")
        raise

    except ValueError as e:
        logger.error("Invalid parameter: %s", e)
        raise


# Import data
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)

        if df.empty:
            raise ValueError("The input CSV file is empty.")

        logger.info("Data loaded successfully. Shape: %s", df.shape)
        return df

    except (HTTPError, URLError) as e:
        logger.error("Unable to access the data URL: %s", e)
        raise

    except pd.errors.EmptyDataError:
        logger.error("The CSV file contains no data.")
        raise

    except pd.errors.ParserError:
        logger.error("Unable to parse the CSV file.")
        raise

    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.error("Error reading data: %s", e)
        raise


# Process data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        required_columns = {"tweet_id", "sentiment"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Drop tweet_id and keep only the required sentiments
        df = df.drop(columns=["tweet_id"])

        final_df = df[
            df["sentiment"].isin(["happiness", "sadness"])
        ].copy()

        if final_df.empty:
            raise ValueError(
                "No rows found for happiness or sadness."
            )

        # Convert labels to numeric values
        final_df["sentiment"] = final_df["sentiment"].map({
            "happiness": 1,
            "sadness": 0
        })

        if final_df["sentiment"].isnull().any():
            raise ValueError("Unexpected missing sentiment values.")

        logger.info("Data processed successfully. Shape: %s", final_df.shape)
        return final_df

    except (KeyError, ValueError) as e:
        logger.error("Data processing failed: %s", e)
        raise


# Save processed data
def save_data(
    data_path: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train_csv")
        test_path = os.path.join(data_path, "test_csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Train and test data saved in %s", data_path)

    except (OSError, PermissionError) as e:
        logger.error("Failed to save data: %s", e)
        raise


# Main function
def main():
    try:
        logger.info("Starting make_datsset...")

        test_size = load_params("params.yaml")

        df = read_data(
            "https://raw.githubusercontent.com/campusx-official/"
            "jupyter-masterclass/refs/heads/main/tweet_emotions.csv"
        )

        final_df = process_data(df)

        # Check that both classes have enough rows
        class_counts = final_df["sentiment"].value_counts()

        if len(class_counts) < 2 or class_counts.min() < 2:
            raise ValueError(
                "Each sentiment class must have at least 2 rows."
            )

        train_df, test_df = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42,
            stratify=final_df["sentiment"]
        )

        data_path = os.path.join("data", "raw")
        save_data(data_path, train_df, test_df)

        logger.info("Data ingestion completed successfully.")

    except Exception:
        logger.exception("Data ingestion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

