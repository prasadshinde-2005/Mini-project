from flask import Flask ,render_template , request 
import mlflow 
import pickle 
import dagshub
import os 
import pandas as pd  
from preprocessing_utility import normalization_sentence

# import flask
app = Flask(__name__)

# for dagshub authentication in ci
token = os.getenv("DAGSHUB_TOKEN")
if not token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")


# model load from model registry
def get_latest_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_model_version_by_alias(model_name , "champion" )
    if not latest_version:
        latest_version = client.client.get_model_version_by_alias(model_name , "candidate")
    return latest_version[0].version if latest_version else None     

# model name and versiion 
model_name = "spmodel"
model_version = get_latest_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)


vectorizer = pickle.load(open("model/vectorizer.pkl" ,'rb'))

# route 1 
@app.route("/")
def home():
    return render_template("index.html" , result = None)



# route 2
@app.route('/predict' , methods = ["POST"])
def predict():

    text = request.form["text"]


    # clean 
    text = normalization_sentence(text) 

    # apply bow on prediction data
    features = vectorizer.transform([text])
    
    #convert sparse matrix to dataframe
    feature_df = pd.DataFrame.sparse.from_spmatric(features)
    feature_df = pd.DataFrame(features.toarray() , columns = [str(i) for i in range(features.shape[1])])
    # prdiction 
    result = model.predict(feature_df)

    
    return render_template("index.html" , result = result[0])
print(app.url_map)
# Start the Flask app only once, at the bottom
if __name__ == "__main__":
    app.run(debug=True)