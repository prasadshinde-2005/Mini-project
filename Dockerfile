FROM python:3.11

WORKDIR /app

COPY flask_app/ /app/

COPY model/vectorizer.pkl /app/model/vectorizer.pkl

RUN pip install -r requirement.txt

RUN python -m nltk.downloader stopwords wordnet 

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

