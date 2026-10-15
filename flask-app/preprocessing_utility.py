
import pandas as pd
import os
import re
import string
import nltk
import logging
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# create the objects your functions use
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
def normalization_sentence(text):
    text = lower_case(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_number(text)
    text = remove_non_ascii(text)
    text = remove_stop_word(text)
    text = lemmatization(text)

    return text

