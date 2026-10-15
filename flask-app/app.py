from flask import Flask ,render_template , request 
import mlflow 
import pickle 
import dagshub
from preprocessing_utility import normalization_sentence
app = Flask(__name__)

dagshub.init(repo_owner='prasadshinde-2005', repo_name='Mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")


# load a model from model registgry


model_name = "spmodel"
model_version = 3

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

    # prdiction 
    result = model.predict(features)

    
    return render_template("index.html" , result = result[0])
print(app.url_map)
# Start the Flask app only once, at the bottom
if __name__ == "__main__":
    app.run(debug=True)