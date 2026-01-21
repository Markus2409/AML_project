### The `notebooks/` Directory detail
This directory contains the sequential Jupyter Notebooks that constitute the project pipeline. They are numbered to indicate the execution order:

* `01_EDA_and_Data_Preprocessing.ipynb`
  Covers **Section 3.1** of the report. It performs the initial data cleaning (KNN Imputation), detects and removes data leakage features (`N_Days`, `Status`), and visualizes the class imbalance and feature distributions via Exploratory Data Analysis.

* `02_Feature_Selection_and_Modeling.ipynb`
  Covers **Section 3.2**. It implements the **Stability Selection Protocol**. It includes the comparative analysis between Standard and Balanced Random Forest, executes the recursive feature ranking across Stratified K-Folds, and identifies the final consensus subset of 11 biomarkers.

* `03_Optimization_and_Benchmark_Testing.ipynb`
  Covers **Section 3.3**. It handles the **Bayesian Optimization** loop (`BayesSearchCV`) to fine-tune the hyperparameters ($C$, $\gamma$, etc.) for Logistic Regression, SVM, and Balanced RF. It concludes by training the final models and generating predictions on the held-out Benchmark Set.

* `04_Analysis_Results.ipynb`
  Covers **Section 4**. The final evaluation suite. It loads the benchmark predictions to generate technical metrics (MCC, F1-Score), visualizes Confusion Matrices, and performs the deep-dive **Error Analysis** on severe misclassifications (e.g., Outlier Patient 275).
