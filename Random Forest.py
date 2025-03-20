import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
file_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Select relevant columns and drop missing values
df = df[['Title', 'Body', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']
df = df[['text', 'class']]

# Tokenization for Word2Vec
df['tokenized_text'] = df['text'].apply(lambda x: x.lower().split())

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)

# Convert text to Word2Vec embeddings
def get_avg_word2vec(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df['word2vec_features'] = df['tokenized_text'].apply(lambda x: get_avg_word2vec(x, word2vec_model, 100))

# Convert list of vectors to array
X = np.vstack(df['word2vec_features'])
y = df['class']

# Store evaluation metrics across 30 runs
results = []

# Train and evaluate the model 30 times
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)

    # Apply SMOTE to fix class imbalance
    smote = SMOTE(random_state=i)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=i, class_weight='balanced')
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Predict and evaluate
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.3).astype(int)
    results.append([
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=1),
        recall_score(y_test, y_pred, zero_division=1),
        f1_score(y_test, y_pred, zero_division=1)
    ])

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Accuracy", "Precision", "Recall", "F1-score"])
results_df.to_csv("random_forest_results.csv", index=False)

print("Random Forest results saved to random_forest_results.csv")
