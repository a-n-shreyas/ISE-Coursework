import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Style settings
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12

# Load data (replace paths as needed)
datasets = ["pytorch", "keras", "incubator-mxnet", "caffe", "tensorflow"]
metrics = ["Precision", "F1-score", "Recall"]

# Initialize results storage
results = {
    "Dataset": [],
    "Model": [],
    "Precision": [],
    "F1-score": [],
    "Recall": []
}

for dataset in datasets:
    # Load results
    nb = pd.read_csv(f"naive_bayes_results.csv")  # Naive Bayes
    rf = pd.read_csv(f"improved_rf_results.csv")  # Random Forest

    # Store aggregated metrics
    for model, df in zip(["Naive Bayes", "Random Forest"], [nb, rf]):
        results["Dataset"].extend([dataset] * len(df))
        results["Model"].extend([model] * len(df))
        for metric in metrics:
            results[metric].extend(df[metric])

results_df = pd.DataFrame(results)

# 1. Bar Chart: Mean Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=results_df.groupby(["Dataset", "Model"])[metric].mean().reset_index(),
        x="Dataset",
        y=metric,
        hue="Model",
        ax=ax
    )
    ax.set_title(f"Mean {metric} Comparison")
    ax.set_ylim(0, 0.35)
    ax.tick_params(axis="x", rotation=45)
    if i > 0:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig("mean_metrics_comparison.png")
plt.close()

# 2. Boxplot: F1-score Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=results_df[results_df["Model"] == "Random Forest"],
    x="Dataset",
    y="F1-score",
    palette="Blues"
)
plt.title("Random Forest F1-Score Distribution Across Datasets")
plt.xticks(rotation=45)
plt.savefig("f1_distribution_boxplot.png")
plt.close()


# 3. Precision-Recall Tradeoff (Example for Pytorch)
def plot_pr_curve(dataset):
    rf_probs = pd.read_csv(f"{dataset}_rf_probs.csv")  # Needs probability scores
    precision, recall, _ = precision_recall_curve(rf_probs["true"], rf_probs["probs"])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {dataset}")
    plt.savefig(f"pr_curve_{dataset}.png")
    plt.close()


# Generate for all datasets
for dataset in datasets:
    plot_pr_curve(dataset)  # Requires probability data

# 4. Effect Size Visualization
cliffs_delta = {
    "Dataset": datasets,
    "Effect Size": [0.91, 0.89, 0.90, 0.87, 0.85]  # From your analysis
}
effect_df = pd.DataFrame(cliffs_delta)

plt.figure(figsize=(10, 6))
sns.barplot(data=effect_df, x="Dataset", y="Effect Size", palette="viridis")
plt.axhline(0.474, color="red", linestyle="--", label="Large Effect Threshold")
plt.title("Cliff's Delta Effect Sizes Across Datasets")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("effect_sizes.png")
plt.close()