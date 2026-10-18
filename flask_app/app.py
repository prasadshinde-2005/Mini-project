
from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd

from preprocessing_utility import normalization_sentence

# Initialize Flask app
app = Flask(__name__)

# DagsHub authentication
token = os.getenv("DAGSHUB_TOKEN")

if not token:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/prasadshinde-2005/Mini-project.mlflow"
)

# Initialize MLflow client
client = mlflow.MlflowClient()


# Get latest model version
def get_latest_version(model_name):
    latest_versions = client.get_latest_versions(model_name)

    if not latest_versions:
        raise ValueError(
            f"No model versions found for '{model_name}'"
        )

    # Return the version number
    return latest_versions[0].version


# Model name and version
model_name = "spmodel"
model_version = get_latest_version(model_name)

# Load registered model
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Load trained vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Route 1: Home page
@app.route("/")
def home():
    return render_template("index.html", result=None)


# Route 2: Prediction
@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("text", "").strip()

    if not text:
        return render_template(
            "index.html",
            result="Please enter some text."
        )

    # Preprocess input text
    text = normalization_sentence(text)

    # Transform text using the trained vectorizer
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    feature_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )

    # Make prediction
    result = model.predict(feature_df)

    # Display prediction
    return render_template(
        "index.html",
        result=result[0]
    )


# Run Flask application
if __name__ == "__main__":
    print(app.url_map)
    app.run(debug=True)
