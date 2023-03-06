import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle 

st.start
pipeline = None
df = pd.read_csv("IMDB_movie_reviews_labeled.csv")

st.

X = df.loc[:, ['review']]
y = df.sentiment

if st.button():


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,␣
,→stratify=y)

X_train_docs = [doc for doc in X_train.review]

pipeline = Pipeline([
('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english',␣
,→max_features=1000)),
('cls', LinearSVC())
])

pipeline.fit(X_train_docs, y_train)

Pipeline(steps=[('vect',
TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
stop_words='english')),
('cls', LinearSVC())])

cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()

predicted = pipeline.predict([doc for doc in X_test.review])

accuracy_score(y_test, predicted)
df_sample = df = pd.read_csv("IMDB_movie_reviews_test_sample.csv")
predicted_sample = pipeline.predict([doc for doc in df_sample.review])
accuracy_score(df_sample.sentiment, predicted_sample)
        with open('pipeline.pkl', 'wb')
        pickle.dump(pipelien,f)


st.subheader("Tesing the model")
review_text=st.text_area("movie Review")

if st.button("Predict"): 
    with open('pipelune.pkl','rb')as f:
        pipeline = pickle.load(f)
        sentiment = pipeline.predict([review_text])
        st.write("Predicted sentimentis:",sentiment)

pclass=st.selectbox("pclass", options=["1", "2", "3"])
age = st.text_input("age")
