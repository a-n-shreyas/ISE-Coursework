import pandas as pd
import numpy as np
import nltk
import gensim
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download VADER lexicon
nltk.download("vader_lexicon")

# Load dataset
file_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Select relevant columns and drop missing values
df = df[['Title', 'Body', 'Comments', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']  # Combine Title & Body
df = df[['text', 'Comments', 'class']]  # Keep only relevant columns

# Check class distribution
print("Class Distribution:\n", df['class'].value_counts())

# Sentiment Analysis using VADER (Moved to Earlier Step)
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Comments'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Convert sentiment scores to categories
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))

# One-Hot Encoding for sentiment labels
encoder = OneHotEncoder(sparse_output=False)
sentiment_encoded = encoder.fit_transform(df[['sentiment_label']])

# Tokenization for Word2Vec
df['tokenized_text'] = df['text'].apply(lambda x: x.lower().split())

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)

# Train TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_weights = tfidf.fit_transform(df['text']).toarray()
vocab = tfidf.get_feature_names_out()
word_to_weight = {word: tfidf_weights[:, i].mean() for i, word in enumerate(vocab)}

# Convert text to TF-IDF-weighted Word2Vec embeddings
def get_weighted_word2vec(tokens, model, word_weights, vector_size):
    vectors = []
    for word in tokens:
        if word in model.wv and word in word_weights:
            vectors.append(model.wv[word] * word_weights[word])
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df['word2vec_features'] = df['tokenized_text'].apply(lambda x: get_weighted_word2vec(x, word2vec_model, word_to_weight, 100))

# Convert list of vectors to array
X = np.vstack(df['word2vec_features'])
y = df['class']

# Store evaluation metrics across 30 runs
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Train and evaluate the model 30 times
for i in range(30):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)

    # Apply SMOTE to fix class imbalance
    smote = SMOTE(random_state=i)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning using Grid Search (Run only on first loop for efficiency)
    if i == 0:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=i),
                                   param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Best model from Grid Search
        best_rf_model = grid_search.best_estimator_

    # Train Random Forest model
    best_rf_model.fit(X_train_resampled, y_train_resampled)

    # Predictions (Using Lower Threshold 0.3)
    y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]  # Get probability of class 1 (Bug)
    y_pred = (y_pred_proba > 0.3).astype(int)  # Lower threshold from 0.5 to 0.3

    # Evaluate model
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, zero_division=1))
    recall_list.append(recall_score(y_test, y_pred, zero_division=1))
    f1_list.append(f1_score(y_test, y_pred, zero_division=1))

# Compute mean & median of evaluation metrics
print("\nFinal Performance Across 30 Runs:")
print(f"Mean Accuracy: {np.mean(accuracy_list):.4f}, Median Accuracy: {np.median(accuracy_list):.4f}")
print(f"Mean Precision: {np.mean(precision_list):.4f}, Median Precision: {np.median(precision_list):.4f}")
print(f"Mean Recall: {np.mean(recall_list):.4f}, Median Recall: {np.median(recall_list):.4f}")
print(f"Mean F1 Score: {np.mean(f1_list):.4f}, Median F1 Score: {np.median(f1_list):.4f}")

# Print sentiment analysis summary (Now Happens Before Model Training)
sentiment_counts = df['sentiment_label'].value_counts()
print("\nSentiment Analysis Summary (VADER):")
print(sentiment_counts)

# Print One-Hot Encoded Labels
print("\nOne-Hot Encoded Sentiment Labels (Example):")
print(sentiment_encoded[:5])
