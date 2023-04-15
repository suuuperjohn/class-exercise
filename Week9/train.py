import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
import re
import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

# define helper functions for text preprocessing and feature extraction.
def sentence_embedding(sentence, word2vec):
    # preprocess the sentence
    sentence = re.sub(r'\W', ' ', sentence.lower())
    words = sentence.split()

    # get the word embeddings
    word_embeddings = [word2vec[word] for word in words if word in word2vec]
    
    if len(word_embeddings) > 0:
        sentence_embedding = np.mean(word_embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(word2vec.vector_size) 
    
    return sentence_embedding

def fit_vectorizer(X, method="bow"):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    elif method == "bow":
        vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    else:
        raise ValueError("Invalid method")
    
    vectorizer.fit(X)
    return vectorizer

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = text.split()
    return words

def create_sentence_embeddings(X, vectorizer, method="tfidf"):
    if "tfidf" in method or "bow" in method:
        X_vec = vectorizer.transform(X)
    elif method == "word2vec":
        # use Word2Vec embeddings
        train_word2vec_data = [preprocess_text(sentence) for sentence in X]
        model = Word2Vec(sentences=train_word2vec_data, vector_size=100, window=5, \
                         min_count=1, workers=4, sg=1, epochs=10)
        model.save("word2vec_model.model")
        X_vec = np.array([sentence_embedding(sentence, model.wv) for sentence in X])

    else:
        raise ValueError("Invalid method")
    
    return X_vec

def main():
    def train_and_print(feature_method, model_name):
        if "tfidf" in feature_method or "bow" in feature_method:
            vectorizer = fit_vectorizer(X_train, method=feature_method)
            X_train_vec = create_sentence_embeddings(X_train, vectorizer, method=feature_method)
            X_test_vec = create_sentence_embeddings(X_test, vectorizer, method=feature_method)
        else:
            X_train_vec = create_sentence_embeddings(X_train, None, method=feature_method)
            X_test_vec = create_sentence_embeddings(X_test, None, method=feature_method)
            scaler = preprocessing.StandardScaler().fit(X_train_vec)
            X_train_vec = scaler.transform(X_train_vec)
            X_test_vec = scaler.transform(X_test_vec)

        best_model = None

        for name, model in models:
            if name == model_name:
                best_model = model

        best_model.fit(X_train_vec, y_train)
        y_pred = best_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Model: {feature_method} + {model_name} with F1 Score = {best_f1_score:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
    #Read the training datasets from CSV files.    
    data = pd.read_csv("train.csv")
    #Combine the benefits, side effects, and comments columns into a single text column.
    data["combined_review"] = data["benefits_review"] + " " + data["side_effects_review"] + " " + data["comments_review"]
    #Assign a label (positive or negative) based on the rating.
    data["label"] = np.where(data["rating"] >= 5, "positive", "negative")
#Read the testing datasets from CSV files.
    test_data = pd.read_csv("test.csv")
    test_data["combined_review"] = test_data["benefits_review"] + " " + test_data["side_effects_review"] + " " + test_data["comments_review"]
    test_data["label"] = np.where(test_data["rating"] >= 5, "positive", "negative")

    X_train, y_train = data["combined_review"], data["label"]
    X_test, y_test = test_data["combined_review"], test_data["label"]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    feature_methods = ["bow", "tfidf", "word2vec"]
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10)),
        ("Logistic Regression", LogisticRegression(max_iter=100)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=10))
    ]


    print("Training...")
    #Use k-fold cross-validation to evaluate the performance of each classifier with each feature extraction method.
    kfold = KFold(n_splits=4, shuffle=True, random_state=0)
    #Retrain the best performing model on the entire training set and test it on the test set.
    best_model = (None, None, -1)  # (feature_method, model_name, f1_score)

    for feature_method in feature_methods:
        if "tfidf" in feature_method or "bow" in feature_method:
            vectorizer = fit_vectorizer(X_train, method=feature_method)
            X_train_vec = create_sentence_embeddings(X_train, vectorizer, method=feature_method)
        else:
            X_train_vec = create_sentence_embeddings(X_train, None, method=feature_method)
            scaler = preprocessing.StandardScaler().fit(X_train_vec)
            X_train_vec = scaler.transform(X_train_vec)

        for model_name, model in models:
            
            cv_results = cross_val_score(model, X_train_vec, y_train, cv=kfold, scoring="f1_weighted")
            mean_f1 = cv_results.mean()
            
            if mean_f1 > best_model[2]:
                best_model = (feature_method, model_name, mean_f1)

            print(f"{feature_method} + {model_name}: F1 Score (mean) = {mean_f1:.3f}, F1 Score (std) = {cv_results.std():.3f}")

    best_feature_method, best_model_name, best_f1_score = best_model

    print("Retrain and Test...")
    # best model
    train_and_print(best_feature_method, best_model_name)
    # baseline model
    most_frequent_class = np.bincount(y_train).argmax()
    y_pred = np.full_like(y_test, most_frequent_class)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
#Print the performance of the best model and compare it to a baseline model that predicts the most frequent class.
    print(f"Model: baseline with F1 Score = {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
if __name__ == "__main__":
    main()