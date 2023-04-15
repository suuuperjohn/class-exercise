import pandas as pd

data = pd.read_csv("train.csv")

data.columns = [column.strip().lower().replace(" ", "_") for column in data.columns]
# Check for missing values
print(data.isnull().sum())

# If there are missing values, you can fill them with the mean or median (for numerical columns) or the most frequent value (for categorical columns)
data.fillna(data.mean(), inplace=True)

import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a string
    text = ' '.join(words)

    return text

# Apply preprocessing to the text columns
data['benefits_review'] = data['benefits_review'].apply(preprocess_text)
data['side_effects'] = data['side_effects'].apply(preprocess_text)
data['overall_comment'] = data['overall_comment'].apply(preprocess_text)

from sklearn.feature_extraction.text import CountVectorizer

# Combine all text columns into a single column
data['combined_text'] = data['benefits'] + ' ' + data['side_effects'] + ' ' + data['overall_comment']

# Create a Bag of Words model
vectorizer = CountVectorizer(max_features=5000)  # You can adjust the number of features as needed
X = vectorizer.fit_transform(data['combined_text']).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF model
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed
X = vectorizer.fit_transform(data['combined_text']).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF model
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed
X = vectorizer.fit_transform(data['combined_text']).toarray()

# Convert target variable to binary format
data['response'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0) # Adjust the threshold as needed
y = data['response'].values

from sklearn.model_selection import train_test_split

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # Adjust the test_size parameter as needed

from sklearn.linear_model import LogisticRegression

# Train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.svm import SVC

# Train the SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Build the Neural Network
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Neural Network
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network
nn_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

from sklearn.metrics import accuracy_score, f1_score

# Logistic Regression
lr_pred = lr_model.predict(X_val)

# Random Forest
rf_pred = rf_model.predict(X_val)

# Support Vector Machine (SVM)
svm_pred = svm_model.predict(X_val)

# Neural Networks
nn_pred = (nn_model.predict(X_val) > 0.5).astype("int32") # Use a threshold of 0.5 for binary classification

# Logistic Regression
lr_accuracy = accuracy_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred)

# Random Forest
rf_accuracy = accuracy_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred)

# Support Vector Machine (SVM)
svm_accuracy = accuracy_score(y_val, svm_pred)
svm_f1 = f1_score(y_val, svm_pred)

# Neural Networks
nn_accuracy = accuracy_score(y_val, nn_pred)
nn_f1 = f1_score(y_val, nn_pred)


print("Logistic Regression:")
print(f"Accuracy: {lr_accuracy:.2f}, F1 Score: {lr_f1:.2f}")

print("Random Forest:")
print(f"Accuracy: {rf_accuracy:.2f}, F1 Score: {rf_f1:.2f}")

print("Support Vector Machine (SVM):")
print(f"Accuracy: {svm_accuracy:.2f}, F1 Score: {svm_f1:.2f}")

print("Neural Networks:")
print(f"Accuracy: {nn_accuracy:.2f}, F1 Score: {nn_f1:.2f}")


import numpy as np

X_full = np.concatenate((X_train, X_val), axis=0)
y_full = np.concatenate((y_train, y_val), axis=0)

# Build the Neural Network
final_nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_full.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Neural Network
final_nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network
final_nn_model.fit(X_full, y_full, batch_size=32, epochs=10)


# Save the final Neural Network model
final_nn_model.save('final_nn_model.h5')

# Save the text vectorizer (e.g., TF-IDF vectorizer)
import pickle

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
