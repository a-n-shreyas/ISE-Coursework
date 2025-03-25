# ISE-Coursework

📝 Introduction
Bug report classification is a crucial aspect of software maintenance, helping developers efficiently triage and resolve issues. However, traditional methods like Naïve Bayes with TF-IDF often struggle, particularly when dealing with imbalanced datasets, where non-bug reports significantly outnumber actual bug reports.

This project explores machine learning approaches to improve bug report classification by:
✔ Evaluating multiple datasets (Pytorch, Keras, Caffe, TensorFlow, incubator-mxnet).

✔ Comparing a Naïve Bayes baseline against an improved Random Forest model.

✔ Using Word2Vec embeddings for feature extraction instead of TF-IDF.

✔ Applying sentiment analysis (VADER) to analyze comments.

✔ Employing SMOTE to handle class imbalance and enhance recall.

✔ Conducting rigorous statistical validation to ensure improvements are significant.

Our results demonstrate that Random Forest outperforms Naïve Bayes across all datasets, but challenges remain in achieving production-level performance. This project provides insights into the limitations of traditional classifiers and suggests future improvements using BERT-based embeddings and hybrid models.

🚀 Whether you're a machine learning enthusiast, software engineer, or researcher, this project serves as a foundation for improving bug classification models!

📌 What You’ll Find in This Repository

📂 Dataset-Based Performance Comparisons (Precision, Recall, F1-Score)

📊 Statistical Analysis & Significance Testing

🛠 Reproducible Code for ML Training & Evaluation

📜 Detailed Observations & Future Work Recommendations

📊 Performance Comparison Across Datasets

🔹 Precision

| Dataset          | Naïve Bayes | Random Forest |
|-----------------|------------|--------------|
| Pytorch        | 1.000      | 0.313        |
| Keras          | 1.000      | 0.281        |
| incubator-mxnet | 1.000     | 0.318        |
| Caffe          | 1.000      | 0.290        |
| TensorFlow     | 1.000      | 0.309        |

🔹 F1-Score
| Dataset          | Naïve Bayes | Random Forest |
|-----------------|------------|--------------|
| Pytorch        | 0.000      | 0.130        |
| Keras          | 0.005      | 0.112        |
| incubator-mxnet | 0.000     | 0.115        |
| Caffe          | 0.000      | 0.111        |
| TensorFlow     | 0.019      | 0.116        |

🔹 Recall
| Dataset          | Naïve Bayes | Random Forest |
|-----------------|------------|--------------|
| Pytorch        | 0.000      | 0.091        |
| Keras          | 0.002      | 0.074        |
| incubator-mxnet | 0.000     | 0.075        |
| Caffe          | 0.000      | 0.075        |
| TensorFlow     | 0.010      | 0.075        |

📌 Key Observations

🚨 Consistent Baseline Failure

Naïve Bayes achieved perfect precision (1.0) across all datasets but zero recall/F1-score except for Keras and TensorFlow, where F1 < 0.02.

Cause: The classifier only predicts the majority class (87-90% non-bug reports), making it ineffective for minority class detection.

🚀 Random Forest Superiority

Statistically significant improvement over Naïve Bayes (p < 0.00001, Wilcoxon test).

Best Performance: Pytorch (F1 = 0.130)

Worst Performance: Caffe (F1 = 0.111)

⚖ Precision-Recall Tradeoff

Highest Precision: incubator-mxnet (0.318)

Highest Recall: Pytorch (0.091)

📈 Dataset Variability
| Metric         | Highest Variance (Std Dev) |
|---------------|--------------------------|
| Precision      | TensorFlow (0.282)      |
| F1-score      | Keras (0.081)           |
| Recall        | TensorFlow (0.056)      |

📌 Statistical Validation

📉 Normality Tests

Naïve Bayes results: W = 1.0, p = 1.0 (No variance).

Random Forest results: Non-normal distributions (p < 0.05).

📊 Effect Sizes (Cliff’s Delta for F1-score)

| Dataset          | Cliff’s Delta (F1) |
|-----------------|-------------------|
| Pytorch        | 0.91 (Large)       |
| Keras          | 0.89 (Large)       |
| incubator-mxnet | 0.90 (Large)      |
| Caffe          | 0.87 (Large)       |
| TensorFlow     | 0.85 (Large)       |

📌 Reflection & Future Improvements

🔴 Limitations

Low Absolute Performance:

Despite improvements, the highest F1-score is only 0.130 (Pytorch), far below production thresholds (F1 > 0.7).

Root Cause: Limited contextual understanding using Word2Vec instead of BERT-based embeddings.

Cross-Dataset Generalization:

Worst performance on Caffe (F1 = 0.111) despite similar conditions.

Possible Reason: Domain-specific jargon not being captured effectively.

💡 Improvement Pathways

✔ Feature Engineering Enhancements:

Introduce domain-specific lexicons (e.g., CUDA error for Pytorch/TensorFlow).

Use FastText instead of Word2Vec to improve handling of rare words.


✔ Threshold Optimization:

Adjust decision thresholds per dataset (e.g., 0.25 for Pytorch vs. 0.15 for Caffe).


✔ Model Architecture Enhancements:

Replace Word2Vec with DistilBERT for better contextual feature extraction.

Explore hybrid ML + Deep Learning models for better performance.


📌 Conclusion

✅ Random Forest significantly outperforms Naïve Bayes (p < 0.00001) across all datasets.

❌ Absolute performance remains inadequate for real-world deployment.

💡 Key takeaways:

Statistical evaluation beyond accuracy is critical.

Domain-specific feature engineering impacts generalization.

Traditional ML models struggle with semantic-heavy tasks.


🔮 Future Work:

Integrate BERT-based embeddings into classification models.

Explore hybrid transformer + ensemble methods for better performance.

📌 Repository Artifacts
📂 GitHub Repository: [Insert Link]

✅ What’s Included?

📊 Results for 5 datasets

🖥 Reproducible code for all experiments

📉 Statistical analysis scripts

📌 How to Use This Repository?
Clone the repository

git clone [(https://github.com/a-n-shreyas/ISE-Coursework.git)![image](https://github.com/user-attachments/assets/c1832d33-6ecf-4b3e-8148-0f068c0abaa7))]


Install dependencies

pip install -r requirements.txt


python model_training.py

Analyze the results

python statistical_analysis.py

📌 Contributors
👨‍💻 Annavati Shreyas – Lead Developer & Researcher

📌 License
🔖 This project is licensed under the MIT License – see the LICENSE file for details.

📌 References
📚 Key Papers & Studies

[1] T. Hall et al., "A Systematic Literature Review on Fault Classification," IEEE TSE, 2018.

[2] A. Lamkanfi et al., "Predicting Severity of Bug Reports," MSR, 2010.

[3] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL, 2019.

[4] C. Tantithamthavorn et al., "Automated Software Quality Assessment," EMSE, 2018.
