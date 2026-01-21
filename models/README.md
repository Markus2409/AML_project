### The `models/` Directory detail
This directory contains the **serialized machine learning models** saved in `.pkl` format using `joblib`. These models have been trained on the full **Development Set** and hyperparameter-tuned using **Bayesian Optimization**. They are ready for immediate inference (e.g., on the Benchmark Set) without the need for retraining.

* `lr_model.pkl`
  The optimized **Logistic Regression** pipeline, tuned for inverse regularization strength ($C$).
  * **Best Parameters:** Inverse Regularization Strength $C \approx 46.79$.

* `svm_model.pkl`
  The optimized **Support Vector Machine** (RBF Kernel) pipeline, with tuned Regularization ($C$) and Kernel coefficient ($\gamma$).
  * **Best Parameters:** Regularization $C \approx 3.15$, Kernel Coefficient $\gamma \approx 0.046$.

* `tree_model.pkl`
  The optimized **Balanced Random Forest** pipeline, tuned for ensemble size, depth, and leaf split criteria to handle the specific class imbalance of the dataset.
  * **Configuration:** An ensemble of **1000 trees** with `max_depth=35`.
  * **Split Criteria:** `min_samples_leaf=2`, `min_samples_split=2`, using `log2` features per split.

