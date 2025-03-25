import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
file_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/tensorflow.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Select relevant columns and handle missing values
df = df[['Title', 'Body', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']
df = df[['text', 'class']]

# Store evaluation metrics across 30 runs
results = []

# Train and evaluate the model 30 times
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=i, stratify=df['class'])

    # Convert text to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train Naïve Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = nb_model.predict(X_test_tfidf)
    results.append([
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=1),
        recall_score(y_test, y_pred, zero_division=1),
        f1_score(y_test, y_pred, zero_division=1)
    ])

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Accuracy", "Precision", "Recall", "F1-score"])
results_df.to_csv("tensorflow_naive_bayes_results.csv", index=False)

print("Naïve Bayes results saved to tensorflow_naive_bayes_results.csv")
