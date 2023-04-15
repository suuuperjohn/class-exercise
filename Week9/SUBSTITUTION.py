import pandas as pd
import numpy as np
import spacy
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def preprocess_text(text, nlp):
    doc = nlp(text.lower())
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(words)

def word_averaging(wv, words):
    all_words, mean = set(), []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.key_to_index:
            mean.append(wv[word])
            all_words.add(wv.key_to_index[word])
    if not mean:
        return np.zeros(wv.vector_size,)
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])

def main():
    data = pd.read_csv("train.csv")
    data["combined_review"] = data["benefits_review"] + " " + data["side_effects_review"] + " " + data["comments_review"]
    data["label"] = np.where(data["rating"] >= 5, "positive", "negative")

    nlp = spacy.load("en_core_web_sm")
    data["combined_review"] = data["combined_review"].apply(lambda x: preprocess_text(x, nlp))
    data["tokenized_review"] = data["combined_review"].apply(lambda x: x.split())

    X = data["combined_review"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_methods = {
        "bow": CountVectorizer(stop_words='english', lowercase=True),
        "tfidf": TfidfVectorizer(stop_words='english', lowercase=True),
    }

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=100)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10)),
    ]

    for feature_name, feature_method in feature_methods.items():
        X_train_vec = feature_method.fit_transform(X_train)
        X_test_vec = feature_method.transform(X_test)

        for model_name, model in models:
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"{feature_name} + {model_name}:")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"F1 Score: {f1:.2f}\n")

    
