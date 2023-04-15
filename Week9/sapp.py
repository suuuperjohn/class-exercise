import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from train_and_test import sentence_embedding, fit_vectorizer, create_sentence_embeddings  # import functions from your main script
from wordcloud import WordCloud

def load_data():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    return train_data, test_data

def preprocess_data(train_data, test_data):
    train_data["combined_review"] = train_data["benefits_review"] + " " + train_data["side_effects_review"] + " " + train_data["comments_review"]
    train_data["label"] = np.where(train_data["rating"] >= 5, "positive", "negative")

    test_data["combined_review"] = test_data["benefits_review"] + " " + test_data["side_effects_review"] + " " + test_data["comments_review"]
    test_data["label"] = np.where(test_data["rating"] >= 5, "positive", "negative")

    return train_data, test_data

def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

@st.cache
def train_model(train_data):
    X_train, y_train = train_data["combined_review"], train_data["label"]
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)

    vectorizer = fit_vectorizer(X_train, method="bow")
    X_train_vec = create_sentence_embeddings(X_train, vectorizer, method="bow")

    model = LogisticRegression(max_iter=100)
    model.fit(X_train_vec, y_train_encoded)
    return model, vectorizer, label_encoder

def display_classification_report(y_true, y_pred, label_encoder):
    y_pred = label_encoder.inverse_transform(y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    report_df = pd.DataFrame(report).transpose().round(2).drop("accuracy",axis=0)
    return report_df

st.title("Drug Review Sentiment Analysis")

st.markdown("""
This interactive analytics application showcases the performance of a sentiment analysis model for drug reviews.
The model uses Bag of Words (BoW) feature extraction and a Logistic Regression classifier.
""")

train_data, test_data = load_data()
train_data, test_data = preprocess_data(train_data, test_data)
model, vectorizer, label_encoder = train_model(train_data)

st.header("1. Model Performance on Test Set")
X_test, y_test = test_data["combined_review"], test_data["label"]
y_test_encoded = label_encoder.transform(y_test)
X_test_vec = create_sentence_embeddings(X_test, vectorizer, method="bow")
y_pred = model.predict(X_test_vec)
y_pred_labels = label_encoder.inverse_transform(y_pred)

plot_confusion_matrix(y_test, y_pred_labels)
#Provide a summary of the model performance evaluation process and the best performing model.
st.markdown("""
Classification Report:
""")
st.write(display_classification_report(y_test, y_pred, label_encoder))

st.write(f"Accuracy: {accuracy_score(y_test, y_pred_labels):.2f}")


st.header("2. Dataset Overview and Summary")
st.markdown("""
Here's a sample of the dataset used for training and validating the model.
""")
st.write(train_data.sample(10))

# dataset shape
st.markdown("The dataset contains **{} rows** and **{} columns**.".format(train_data.shape[0], train_data.shape[1]))

# distribution of ratings
st.subheader("Distribution of Ratings")
rating_counts = train_data["rating"].value_counts()
st.bar_chart(rating_counts)

# distribution of labels (positive/negative)
st.subheader("Distribution of Labels (Positive/Negative)")
label_counts = train_data["label"].value_counts()
st.bar_chart(label_counts)

# show the average length of reviews
train_data["review_length"] = train_data["combined_review"].apply(lambda x: len(x.split()))
avg_review_length = train_data["review_length"].mean()
st.markdown("The average length of the reviews in the dataset is **{:.0f} words**.".format(avg_review_length))

st.subheader("Word Cloud of Most Common Words")
text = ' '.join(review for review in train_data["combined_review"])
wordcloud = WordCloud(stopwords=None, background_color="white", max_words=100, contour_width=3, contour_color="steelblue").generate(text)

fig_c, ax = plt.subplots()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(fig_c)

st.header("3. Model Performance Summary")
st.markdown("""
We evaluated different feature extraction methods and machine learning models, including:

- Feature Extraction Methods: Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word2Vec.
- Machine Learning Models: Logistic Regression, XGBoost, and Decision Tree.

The best performing model was Logistic Regression combined with Bag of Words (BoW) feature extraction. This model achieved an F1 score of 0.81 and an accuracy of 0.81 on the test set.


As a comparison, we also implemented a baseline model that predicts the majority class from the training set. The baseline model achieved an F1 score of 0.67 and an accuracy of 0.77 on the test set. This shows that our best model (BoW + Logistic Regression) significantly outperforms the baseline model and can provide valuable insights for stakeholders.
""")

st.header("4. Test Model Performance in Real Time")
st.markdown("""
Provide a drug review below to test the sentiment analysis model in real time.
""")
input_text = st.text_area("Enter a drug review here...")


if st.button("Analyze Sentiment"):
    if input_text:
        input_vec = create_sentence_embeddings([input_text], vectorizer, method="bow")
        input_pred = model.predict(input_vec)
        input_pred_label = label_encoder.inverse_transform(input_pred)[0]
        st.markdown(f"**Predicted sentiment:** {input_pred_label.capitalize()}")
    else:
        st.warning("Please enter a drug review to analyze.")


