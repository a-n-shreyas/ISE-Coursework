import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

# Load dataset
file_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Preprocessing
df = df[['Title', 'Body', 'class']].dropna()
df['text'] = df['Title'] + " " + df['Body']
df = df[['text', 'class']]

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

results = []

for i in range(30):
    # Split first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['class'],
        test_size=0.2,
        random_state=i,
        stratify=df['class']
    )

    # Feature Engineering
    # -------------------
    # 1. Word2Vec Features (trained only on training data)
    train_tokens = X_train.apply(lambda x: x.lower().split())
    w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)


    def get_w2v_features(text):
        tokens = text.lower().split()
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)


    # 2. Sentiment Features
    X_train_features = np.array([
        np.concatenate([
            get_w2v_features(text),
            [analyzer.polarity_scores(text)['compound']]
        ]) for text in X_train
    ])

    X_test_features = np.array([
        np.concatenate([
            get_w2v_features(text),
            [analyzer.polarity_scores(text)['compound']]
        ]) for text in X_test
    ])

    # Handle Class Imbalance
    smote = SMOTE(random_state=i)
    X_resampled, y_resampled = smote.fit_resample(X_train_features, y_train)

    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=i),
        param_grid,
        cv=3,
        scoring='f1'
    )
    grid_search.fit(X_resampled, y_resampled)
    best_model = grid_search.best_estimator_

    # Optimal Threshold Finding
    train_probs = best_model.predict_proba(X_resampled)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_resampled, train_probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Final Prediction
    test_probs = best_model.predict_proba(X_test_features)[:, 1]
    y_pred = (test_probs >= optimal_threshold).astype(int)

    results.append([
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=1),
        recall_score(y_test, y_pred, zero_division=1),
        f1_score(y_test, y_pred, zero_division=1)
    ])

# Save results
results_df = pd.DataFrame(results, columns=["Accuracy", "Precision", "Recall", "F1-score"])
results_df.to_csv("tensorflow_improved_rf_results.csv", index=False)

print("Improved Random Forest results saved to tensorflow_improved_rf_results.csv")