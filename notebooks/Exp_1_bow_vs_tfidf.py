
import pandas as pd
import os
import yaml
import logging
import sys
import mlflow
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score 

import dagshub 
import mlflow 

dagshub.init(repo_owner='prasadshinde-2005', repo_name='Mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")


mlflow.set_experiment("Logistic_reg Baseline model_using_tfidf") 


df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/"
            "jupyter-masterclass/refs/heads/main/tweet_emotions.csv"
        )

# stay only 2 sentiments 
x = df["sentiment"].isin(["happiness" , "sadness"])
df = df[x]

# remove the unwanted columns 
df.drop(columns =["tweet_id"] , inplace = True)

#import re 
import nltk
import string 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



# Function for data cleaning

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize tools
stop_words = set(stopwords.words("english"))

# Keep negation words because they affect sentiment
stop_words = stop_words - {"no", "not", "nor"}

lemmatizer = WordNetLemmatizer()


# 1. Convert text to lowercase
def lower_case(text):
    return str(text).lower()


# 2. Remove URLs
def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


# 3. Remove punctuation
def remove_punctuation(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# 4. Remove numbers
def remove_number(text):
    return re.sub(r"\d+", "", text)


# 5. Remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)


# 6. Remove stopwords
def remove_stop_word(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# 7. Lemmatize words
def lemmatization(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# 8. Complete text normalization pipeline
def normalization_sentence(text):
    text = lower_case(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_number(text)
    text = remove_non_ascii(text)
    text = remove_stop_word(text)
    text = lemmatization(text)

    return text

# normalized the data
df["content"] = df["content"].apply(normalization_sentence)

# convert the text into vector
df["sentiment"] = df["sentiment"].replace({"sadness":0 , "happiness":1})

# set experiment
mlflow.set_experiment("BOW VS TF-IDF")

# Define fetaure extraction metode 
vectorizers = {"BOW" : CountVectorizer() ,
              "Tf-idf":TfidfVectorizer()}

# Import all algo 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
import mlflow.xgboost

# Define algorithm 
algorithms = {
    "LogisticRegression" : LogisticRegression() ,
    "MultinomialNB" :MultinomialNB() ,
    "XGboost" : XGBClassifier() ,
    "RandomForest": RandomForestClassifier(),
    "GradientBoostin": GradientBoostingClassifier()
}

with mlflow.start_run(run_name = "All_Experiment") as parent_run :
    # chils run for algo and vectorizer 
    for algo_name , model in algorithms.items():
        for vec_name , vectorizer in vectorizers.items():

            with mlflow.start_run(run_name =f"{algo_name} with {vec_name}" , nested = True) as child_run:
                x = vectorizer.fit_transform(df["content"])
                y = df["sentiment"]
                y = y.astype(int)
                x_train, x_test, y_train , y_test = train_test_split( x, y, test_size = 0.2 , random_state = 42)

                # log Preprocessing parameters 
                mlflow.log_param("vectorizer"  , vec_name)
                mlflow.log_param("algorithm" , algo_name)
                mlflow.log_param("test_size" , 0.2)

                # model training 
               
                model.fit(x_train , y_train)

                # log model parameters 

                if algo_name == "logisticRegression":
                    mlflow.log_param("C" ,model.C)
                elif algo_name =="MultinomialNB":
                    mlflow.log_param("alpha" , model.alpha)
                elif algo_name =="RandomForest": 
                    mlflow.log_param("n_estimators" ,model.n_estimators)
                    mlflow.log_param("max_depth" , model.max_depth)  

                elif algo_name =="GradientBossting":
                    mlflow.log_param("n_estimators" ,model.n_estimators)
                    mlflow.log_param("max_depth" , model.max_depth)  
                    mlflow.log_param("learning_rate" , model.learning_rate) 

                   
                    # model validation 
                y_pred = model.predict(x_test)

                # find accuracy 
                accuracy = accuracy_score(y_test , y_pred)

                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )

               #  Log The params 
                mlflow.log_metric("accuracy" , accuracy)
                mlflow.log_metric("precision" , precision)
                mlflow.log_metric("recall" , recall)
                mlflow.log_metric("f1" , f1)
                
                                              
                # Log the model using its appropriate MLflow flavor
                if isinstance(model, XGBClassifier):
                    mlflow.xgboost.log_model(
                        xgb_model=model,
                        name="model-1" 
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model , name ="model-1" , skops_trusted_types=["sklearn.tree._tree.Tree"])
               # log notebook
                mlflow.log_artifact(__file__) 

                
                # print all algo with metrics
                print(f"Algorithm:{algo_name} , Feature Engineering:{vec_name}") 
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 score:", f1)