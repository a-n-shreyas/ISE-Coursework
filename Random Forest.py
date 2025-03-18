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

# Load datasets
file_paths = ["/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"]
dataframes = [pd.read_csv(file, encoding="utf-8") for file in file_paths]

# Merge datasets into one DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Select relevant columns and drop missing values
df = df[['Title', 'Body', 'Comments', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']  # Combine Title & Body
df = df[['text', 'Comments', 'class']]  # Keep only relevant columns

# Check class distribution
print("Class Distribution:\n", df['class'].value_counts())

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to fix class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Train best Random Forest model
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions (Using Lower Threshold 0.3)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]  # Get probability of class 1 (Bug)
y_pred = (y_pred_proba > 0.3).astype(int)  # Lower threshold from 0.5 to 0.3

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Print evaluation results
print("\nRandom Forest + TF-IDF Weighted Word2Vec Performance (After Tuning):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Comments'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Convert sentiment scores to categories
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))

# One-Hot Encoding for sentiment labels
encoder = OneHotEncoder(sparse=False)
sentiment_encoded = encoder.fit_transform(df[['sentiment_label']])

# Print sentiment analysis summary
sentiment_counts = df['sentiment_label'].value_counts()
print("\nSentiment Analysis Summary (VADER):")
print(sentiment_counts)

# Print One-Hot Encoded Labels
print("\nOne-Hot Encoded Sentiment Labels (Example):")
print(sentiment_encoded[:5])
