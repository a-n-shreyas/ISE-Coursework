import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Load results - use your actual filenames
    nb = pd.read_csv(f"naive_bayes_results.csv")
    rf = pd.read_csv(f"improved_rf_results.csv")

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
    color="blue"  # Fixed parameter
)
plt.title("Random Forest F1-Score Distribution Across Datasets")
plt.xticks(rotation=45)
plt.savefig("f1_distribution_boxplot.png")
plt.close()

# 4. Effect Size Visualization
cliffs_delta = {
    "Dataset": datasets,
    "Effect Size": [0.91, 0.89, 0.90, 0.87, 0.85]
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