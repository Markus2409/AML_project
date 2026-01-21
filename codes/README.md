### The `codes/` Directory detail
This folder contains modular Python scripts designed to streamline the analysis and ensure reproducibility.

* `confusion_matrix.py`
  A visualization utility containing the `plot_cm` function. It wraps Seaborn and Matplotlib to generate standardized **Confusion Matrices**, supporting custom color palettes, automatic figure saving, and consistent labeling for the 4 PBC stages.

* `fpr_fnr.py`
  Dedicated to **Clinical Error Analysis**. It contains:
  * `plot_fn_rate`: Calculates and visualizes the **False Negative Rate (Miss Rate)** per stage.
  * `plot_fp_rate`: Calculates and visualizes the **False Positive Rate (False Alarm)** per stage.
  These functions generate grouped bar charts to allow direct comparison of error patterns across multiple models.

* `stability_feature_sel.py`
  The core implementation of the **Stability Selection Protocol**.
  * `feat_sel`: The main driver function. It ranks features using Gini Importance (via a user-defined Tree-based model like *Balanced Random Forest*) and iteratively evaluates the MCC score of top-k subsets to identify the optimal feature count.
  * `performance_on_subset`: A helper function that trains and validates the pipeline on restricted feature subsets to drive the selection process.

* `requirements.txt`
  Lists all the external libraries and specific versions required to reproduce the environment.
