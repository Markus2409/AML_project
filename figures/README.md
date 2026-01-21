### The `figures/` Directory detail
This directory stores high-quality **vector graphics (.svg)** generated throughout the analysis. To facilitate navigation, the filenames follow a strict naming convention based on the pipeline stage:

* **Naming Convention:** `XX_description.svg`
  The prefix `XX` corresponds to the number of the **Jupyter Notebook** that generated the figure.

* **Content Overview:**
  * `01_*.svg`: **Exploratory Data Analysis**. Includes distributions, boxplots before/after scaling, and the correlation matrix.
  * `02_*.svg`: **Feature Selection**. Visualizations of the Stability Selection process, including Gini Importance rankings and feature consensus stability across folds.
  * `04_*.svg`: **Clinical Evaluation**. The final performance charts, including Confusion Matrices (`cm`), False Negative/Positive Rates (`fnr`, `fpr`), and the specific error analysis on severe misclassifications.

> **Note:** For a detailed interpretation of these plots, please refer to the markdown cells within the corresponding notebooks.
