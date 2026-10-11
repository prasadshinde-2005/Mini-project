
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
from sklearn.model_selection import GridSearchCV
import dagshub 
import mlflow 


dagshub.init(repo_owner='prasadshinde-2005', repo_name='Mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/prasadshinde-2005/Mini-project.mlflow")



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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df["content"])
y = df["sentiment"]

x_train, x_test, y_train ,y_test =train_test_split(x, y, test_size = 0.2 , random_state = 42)

mlflow.set_experiment("Logestic_reg_with_BOW") 

# Define parameters 
param_grid ={
    "C" :[0.1 ,1, 10],
    "penalty":["l1", "l2"] ,
    "solver":["liblinear"]
}

with mlflow.start_run():
    # perform grid search
    grid_search = GridSearchCV(LogisticRegression() , param_grid , cv = 5, scoring ="f1" , n_jobs = -1)
    grid_search.fit(x_train , y_train)

    for params , mean_score , std_score in zip(grid_search.cv_results_["params"] , grid_search.cv_results["mean_test_score"] , grid_search.cv_results["std_test_score"]): 
        with mlflow.start_run(run_name = f"LR with_params:{params}" ,netsed =True):
            model = LogisticRegression(**params)
            model.fit(x_train, y_train)
 
                    
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



            # log parametersand metrics 
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score" ,mean_score)
            mlflow.log_metric("std_cv_score" , std_score)
            mlflow.log_metric("accuarcy" ,accuracy)
            mlflow.log_metric("precision" , precision)
            mlflow.log_metric("recall" , recall)
            mlflow.log_metric("f1" , f1)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            mlflow.log_params(best_params) 
            mlflow.log_metric("Best_f1_score" , best_score) 

            print(f"best_param{best_params}")
            print(f"best F1 score{best_score}")                