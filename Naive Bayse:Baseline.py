import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
file_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Select relevant columns and handle missing values
df = df[['Title', 'Body', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']  # Combine Title & Body
df = df[['text', 'class']]  # Keep only relevant columns

# Store evaluation metrics across 30 runs
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Train and evaluate the model 30 times
for i in range(30):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=i, stratify=df['class'])

    # Convert text to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train Naïve Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Predict on test set using Naïve Bayes
    y_pred_nb = nb_model.predict(X_test_tfidf)

    # Evaluate model
    accuracy_list.append(accuracy_score(y_test, y_pred_nb))
    precision_list.append(precision_score(y_test, y_pred_nb, zero_division=1))
    recall_list.append(recall_score(y_test, y_pred_nb, zero_division=1))
    f1_list.append(f1_score(y_test, y_pred_nb, zero_division=1))

# Compute mean & median of evaluation metrics
print("\nFinal Naïve Bayes Performance Across 30 Runs:")
print(f"Mean Accuracy: {np.mean(accuracy_list):.4f}, Median Accuracy: {np.median(accuracy_list):.4f}")
print(f"Mean Precision: {np.mean(precision_list):.4f}, Median Precision: {np.median(precision_list):.4f}")
print(f"Mean Recall: {np.mean(recall_list):.4f}, Median Recall: {np.median(recall_list):.4f}")
print(f"Mean F1 Score: {np.mean(f1_list):.4f}, Median F1 Score: {np.median(f1_list):.4f}")
