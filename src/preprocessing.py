
import pandas as pd
import os
import re
import string
import nltk
import logging
import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

stop_words = set()
lemmatizer = WordNetLemmatizer()


# Load NLTK resources
def load_nltk_resources():
    global stop_words

    try:
        logger.debug("Checking NLTK resources...")

        for resource in ["wordnet", "stopwords"]:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(
                    f"Failed to download NLTK resource: {resource}"
                )

        stop_words = set(stopwords.words("english"))

        logger.info("NLTK resources loaded successfully.")

    except Exception:
        logger.exception("Failed to load NLTK resources.")
        raise


# Fetch the data
def read_data(url: str) -> pd.DataFrame:
    try:
        logger.debug("Reading data from: %s", url)

        df = pd.read_csv(url)

        if df.empty:
            raise ValueError(f"No data found in {url}")

        logger.info("Data loaded successfully. Shape: %s", df.shape)
        return df

    except FileNotFoundError:
        logger.exception("File not found: %s", url)
        raise

    except pd.errors.EmptyDataError:
        logger.exception("The CSV file is empty: %s", url)
        raise

    except pd.errors.ParserError:
        logger.exception("Unable to parse CSV: %s", url)
        raise

    except Exception:
        logger.exception("Error while reading data: %s", url)
        raise


# Lowercase
def lower_case(text):
    return str(text).lower()


# Remove URLs
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


# Remove punctuation
def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Remove numbers
def remove_number(text):
    return re.sub(r"\d+", "", text)


# Remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)


# Remove stop words
def remove_stop_word(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# Lemmatization
def lemmatization(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# Normalize a single sentence
def normalization_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_urls(sentence)
    sentence = remove_punctuation(sentence)
    sentence = remove_number(sentence)
    sentence = remove_non_ascii(sentence)
    sentence = remove_stop_word(sentence)
    sentence = lemmatization(sentence)

    return sentence


# Normalize a DataFrame
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Starting text normalization...")

        if "content" not in df.columns:
            raise ValueError("Required column 'content' is missing.")

        df = df.copy()
        df["content"] = df["content"].fillna("").apply(
            normalization_sentence
        )

        logger.info("Text normalization completed. Shape: %s", df.shape)
        return df

    except Exception:
        logger.exception("Error during text normalization.")
        raise


# Normalize train and test data
def normalized_data(train_data, test_data):
    try:
        logger.info("Normalizing training data...")
        train_processed_data = normalize_text(train_data)

        logger.info("Normalizing testing data...")
        test_processed_data = normalize_text(test_data)

        return train_processed_data, test_processed_data

    except Exception:
        logger.exception("Failed to normalize train/test data.")
        raise


# Save processed data
def save_process_data(
    data_path,
    train_processed_data,
    test_processed_data
):
    try:
        logger.debug("Creating output directory: %s", data_path)
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train_csv")
        test_path = os.path.join(data_path, "test_csv")

        train_processed_data.to_csv(train_path, index=False)
        test_processed_data.to_csv(test_path, index=False)

        logger.info("Train data saved to: %s", train_path)
        logger.info("Test data saved to: %s", test_path)

    except Exception:
        logger.exception("Failed to save processed data.")
        raise


# Main function
def main():
    try:
        logger.info("Starting data preprocessing...")

        load_nltk_resources()

        train_data = read_data("./data/raw/train_csv")
        test_data = read_data("./data/raw/test_csv")

        logger.debug("Train data shape: %s", train_data.shape)
        logger.debug("Test data shape: %s", test_data.shape)

        train_processed_data, test_processed_data = normalized_data(
            train_data, test_data
        )

        data_path = os.path.join("data", "processed")

        save_process_data(
            data_path,
            train_processed_data,
            test_processed_data
        )

        logger.info("Data preprocessing completed successfully.")

    except Exception:
        logger.exception("Data preprocessing failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
