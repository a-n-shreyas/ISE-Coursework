import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def cliffs_delta(x, y):
    """Effect size measurement for non-parametric distributions"""
    diff = np.array([xi - yj for xi in x for yj in y])
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y))

# Load results
nb_results = pd.read_csv("naive_bayes_results.csv")
rf_results = pd.read_csv("improved_rf_results.csv")

# Extract F1-scores
nb_f1 = nb_results["Recall"]
rf_f1 = rf_results["Recall"]

# Print summary statistics
print("\n📊 Recall Summary Statistics")
print(f"Naïve Bayes:\n{nb_f1.describe()}")
print(f"\nRandom Forest:\n{rf_f1.describe()}")

# Check unique values
print("\n🎯 Unique F1-score Values in Random Forest:")
print(rf_f1.unique())

# Normality tests
print("\n📊 Shapiro-Wilk Normality Tests:")
nb_shapiro = stats.shapiro(nb_f1)
rf_shapiro = stats.shapiro(rf_f1)
print(f"Naïve Bayes: W = {nb_shapiro.statistic:.3f}, p = {nb_shapiro.pvalue:.5f}")
print(f"Random Forest: W = {rf_shapiro.statistic:.3f}, p = {rf_shapiro.pvalue:.5f}")

# Statistical test selection
alpha = 0.05
if nb_shapiro.pvalue > alpha and rf_shapiro.pvalue > alpha:
    test_result = stats.ttest_rel(nb_f1, rf_f1)
    test_name = "Paired t-test"
else:
    test_result = stats.wilcoxon(nb_f1, rf_f1)
    test_name = "Wilcoxon Signed-Rank Test"

# Print test results
print(f"\n🧪 {test_name} Results:")
print(f"Test Statistic = {test_result.statistic:.5f}")
print(f"p-value = {test_result.pvalue:.5f}")

# Effect size analysis
print("\n📏 Effect Size Analysis:")
print(f"Mean Difference = {np.mean(rf_f1) - np.mean(nb_f1):.4f}")
print(f"Cohen's d = {stats.cohen_d(nb_f1, rf_f1):.3f}")
print(f"Cliff's Delta = {cliffs_delta(rf_f1, nb_f1):.3f}")

# Visualization
plt.figure(figsize=(8, 6))
sns.boxplot(data=[nb_f1, rf_f1],
            palette=["#1f77b4", "#2ca02c"],
            width=0.4)
plt.xticks([0, 1], ["Naïve Bayes", "Random Forest"])
plt.ylabel("Recall", fontsize=12)
plt.title("Model Comparison: Recall Distribution Across 30 Runs", pad=20)
sns.despine()
plt.tight_layout()
plt.show()