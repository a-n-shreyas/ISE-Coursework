import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
nb_results = pd.read_csv("naive_bayes_results.csv")
rf_results = pd.read_csv("random_forest_results.csv")

# Extract Accuracy scores
nb_accuracy = nb_results["F1-score"]
rf_accuracy = rf_results["F1-score"]

# Print summary statistics
print("\n📊 F1-score Summary Statistics")
print(f"Naïve Bayes:\n{nb_accuracy.describe()}")
print(f"\nRandom Forest:\n{rf_accuracy.describe()}")

# Check unique values in Random Forest accuracy
print("\n🎯 Unique F1-score Scores in Random Forest:")
print(rf_accuracy.unique())

# Shapiro-Wilk test (Normality test)
nb_shapiro_p = stats.shapiro(nb_accuracy).pvalue
rf_shapiro_p = stats.shapiro(rf_accuracy).pvalue

print("\n📊 Shapiro-Wilk Test (Normality Check)")
print(f"Naïve Bayes p-value: {nb_shapiro_p:.5f}")
print(f"Random Forest p-value: {rf_shapiro_p:.5f}")

# Choose statistical test based on normality
if nb_shapiro_p > 0.05 and rf_shapiro_p > 0.05:
    stat, p_value = stats.ttest_rel(nb_accuracy, rf_accuracy)
    test_name = "Paired t-test"
else:
    stat, p_value = stats.wilcoxon(nb_accuracy, rf_accuracy)
    test_name = "Wilcoxon Signed-Rank Test"

# Print test results
print(f"\n🧪 {test_name} Results")
print(f"Statistic: {stat:.5f}")
print(f"P-value: {p_value:.5f}")

# Box plot
plt.figure(figsize=(6, 5))
sns.boxplot(data=[nb_accuracy, rf_accuracy], palette=["blue", "green"])
plt.xticks([0, 1], ["Naïve Bayes", "Random Forest"])
plt.ylabel("F1-score")
plt.title("F1-score Comparison Across 30 Runs")
plt.show()
