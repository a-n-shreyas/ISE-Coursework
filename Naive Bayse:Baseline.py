import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
file_paths = ["/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"]
dataframes = [pd.read_csv(file, encoding='utf-8') for file in file_paths]

# Merge datasets into one DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Select relevant columns and handle missing values
df = df[['Title', 'Body', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']  # Combine Title & Body
df = df[['text', 'class']]  # Keep only relevant columns

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on test set using Naïve Bayes
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate Naïve Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# Print Naïve Bayes results
print("Naïve Bayes Performance:")
print(f"Accuracy: {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f}")
print(f"F1 Score: {f1_nb:.4f}")
