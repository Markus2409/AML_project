# Multi-Class Classification of Liver Cirrhosis Stages: *An Evaluation of Random Forest, SVM, and Logistic Regression for the staging of Primary Biliary Cholangitis*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](codes/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Pipeline-orange?logo=jupyter)](notebooks/)
[![Dataset](https://img.shields.io/badge/Dataset-PBC_Liver_Stages-28acb3?logo=kaggle)](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)
[![Report](https://img.shields.io/badge/Read_Final_Report-PDF-b31b1b?logo=adobe-acrobat-reader&logoColor=white)](AML_BASIC__Marco_Cuscuna_Report.pdf)
[![Course](https://img.shields.io/badge/Course-AML_Basic_2026-880808?logo=unibo)](https://www.unibo.it/it/studiare/insegnamenti-competenze-trasversali-moocs/insegnamenti/insegnamento/2025/524694)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn+Imbalanced--Learn-yellowgreen)
![Optimization](https://img.shields.io/badge/Optimization-Bayesian_(Scikit--Optimize)-red)
![Technique](https://img.shields.io/badge/Technique-Stability_Selection-teal)
[![Model](https://img.shields.io/badge/Model-Balanced_Random_Forest-2ea44f)]()
[![Model](https://img.shields.io/badge/Model-SVM_(RBF_Kernel)-2ea44f)]()
[![Model](https://img.shields.io/badge/Model-Logistic_Regression-2ea44f)]()
<br>

<img width="1534" height="601" alt="project-logo" src="https://github.com/user-attachments/assets/7225d649-401c-47fd-a999-74bc2279aafd" />



## Table of Contents

- [Introduction & State of the Art](#introduction--state-of-the-art)
- [Repository Structure](#repository-structure)
- [Software & Libraries](#software--libraries)
- [Methodological Pipeline](#methodological-pipeline)
- [Key Results](#5-key-results-and-conclusions)
- [References](#references)
- [Contacts](#contacts)

---

## Introduction & State of the Art

This repository documents the **Applied Machine Learning Project** developed during the *Applied Machine Learning (Basic)* course of the Master’s Degree in **Bioinformatics** (University of Bologna, 2026). <br>
The project addresses the **Multi-Class Classification** of histological stages (1, 2, 3, 4) in patients affected by **Primary Biliary Cholangitis (PBC)**. Accurate staging is critical for determining prognosis and therapeutic strategies, yet it remains challenging due to the overlapping biochemical profiles of intermediate stages.<br>
The main goal was to design a **robust predictive pipeline** capable of handling typical clinical data challenges: **severe class imbalance** (Stage 1 ≈ 5%) and **limited sample size** ($N=312$). Unlike standard approaches, this study implements a **Stability Selection** protocol to identify reliable biomarkers and employs **Bayesian Optimization** to fine-tune three distinct architectures: Logistic Regression, SVM, and Balanced Random Forest.<br>
This repository serves as a container for the source code, the serialized models, and the final academic report.<br>

---

## Repository Structure

The repository is organized into the following folders:<br>

- [`AML_BASIC_Marco_Cuscuna_Report.pdf`](AML_BASIC_Marco_Cuscuna_Report.pdf) The final academic report detailing the study's background, methodology, results, and clinical interpretation.

- [`notebooks/`](./notebooks) 
  The directory containing the four sequential Jupyter Notebooks that constitute the project pipeline, guiding the user from Exploratory Data Analysis to the final Clinical Evaluation.

- [`codes/`](./codes)
  A folder containing custom Python utility scripts. It includes the core implementation of the Stability Selection algorithm and modular functions for visualization (e.g., Confusion Matrices, FPR/FNR plots) to keep the main notebooks readable, and the requirements.txt file to installation of the required packages.

- [`data/`](./data) 
  Stores the dataset at various stages of the pipeline: raw data, preprocessed versions (imputed and scaled), selected feature subsets, and the logged performance metrics from the benchmark testing.

- [`models/`](./models) 
  Stores the serialized trained models (`.pkl` format) for Logistic Regression, SVM, and Balanced Random Forest. These allow for the immediate loading of optimized classifiers without retraining.

- [`figures/`](./figures) 
  Contains all static visualizations generated during the analysis, including correlation heatmaps, clinical feature boxplots, and specific error analysis charts.
---

## Software & Libraries

The project relies on the following scientific libraries:<br><br>


| **Library** | **Purpose** | **Reference** |
|:---|:---|:---|
| **Scikit-learn** | Core machine learning framework for preprocessing, model training (LR, SVM), and validation metrics. | [Pedregosa 2011](#ref-sklearn) |
| **Imbalanced-learn** | Implementation of **Balanced Random Forest** (undersampling within ensemble) to handle class skew. | [Lemaître 2017](#ref-imblearn) |
| **Scikit-optimize** | Sequential model-based optimization using **Bayesian Optimization** (`BayesSearchCV`) for hyperparameter tuning. | [Head 2018](#ref-skopt) |
| **Pandas / NumPy** | Efficient data manipulation, vector operations, and handling of clinical dataframe structures. | [Reback 2020](#ref-pandas); [Harris 2020](#ref-numpy) |
| **Matplotlib / Seaborn** | Statistical data visualization for EDA, correlation heatmaps, and confusion matrices. | [Hunter 2007](#ref-matplotlib); [Waskom 2021](#ref-seaborn) |
> **Table 1:** All required packages can be installed easily via the command: `pip install -r codes/requirements.txt`

<br>

---

## Methodological Pipeline

### 1. Data Preprocessing & Integrity
The analysis started with a raw cohort of 418 patients. To ensure the reliability of the clinical models, a strict **Data Quality Control** protocol was applied. First, 6 samples were discarded due to missing target labels, followed by a "noise reduction" step that filtered out 106 samples from the observational cohort presenting excessive missing data (>2 null features).

For the remaining dataset ($N=312$), we implemented a hybrid imputation strategy: **KNN Imputer** ($k=5$) was used for numerical variables to preserve local multivariate structures, while Mode Imputation was applied to categorical features. Crucially, the variables `N_Days` and `Status` were identified as high-correlation features representing future outcomes; they were strictly removed to prevent **Data Leakage** and ensure the model's applicability in a real-world diagnostic setting. Finally, all numerical predictors underwent **Z-score Normalization** to prevent magnitude bias and facilitate the convergence of distance-based algorithms like SVM.

### 2. Validation Framework
To guarantee a robust evaluation, the dataset was partitioned into a **Development Set (80%)** used for training and hyperparameter tuning, and a held-out **Benchmark Set (20%)** reserved exclusively for the final testing. Within the Development Set, we employed a **Stratified K-Fold Cross-Validation** ($k=5$) to maintain the proportion of biological stages across folds, preventing bias against the underrepresented minority classes.

### 3. Stability Feature Selection
We implemented a **Stability Selection Protocol**. A preliminary analysis identified the **Balanced Random Forest** as the superior ranking engine compared to standard RF, due to its ability to handle class imbalance.
The protocol aggregated feature importance rankings across the 5 cross-validation folds. A "Consensus Filter" was applied, retaining only those biomarkers selected in more than **80% of the iterations** (4 out of 5 folds). This rigorous process converged to a robust subset of **11 features** (including *Bilirubin, Copper, Prothrombin, and Age*), effectively discarding noise variables such as *Sex* and *Ascites*.

### 4. Benchmark Testing and Result Analysis
The three candidate architectures—**Logistic Regression**, **SVM (RBF Kernel)**, and **Balanced Random Forest**—underwent a fine-tuning phase using **Bayesian Optimization** (`BayesSearchCV`, 60 iterations) to maximize the **Matthews Correlation Coefficient (MCC)**.
The final evaluation on the Benchmark Set focused on clinical safety, analyzing the **False Negative Rates (FNR)** per stage and performing a deep-dive error analysis on severe misclassifications. This included the profiling of outliers such as **Patient ID 275**, a cirrhotic subject misclassified as early-stage due to an atypical "pseudo-healthy" phenotype (normal Albumin levels and absence of Hepatomegaly).

### 5. Key Results and Conclusions

The following table presents the detailed performance metrics for each model on the independent **Benchmark Set**. The **Logistic Regression** achieved the highest overall MCC (**0.312**), balancing the trade-off between identifying early-stage patients and overall classification stability.

| Model | Stage | Support | Precision | Recall | F1-Score | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Balanced**<br>**Random**<br>**Forest** | 1 | 3 | 0.12 | 0.33 | 0.18 | |
| | 2 | 14 | 0.31 | 0.36 | 0.33 | |
| | 3 | 24 | 🔴 **0.53** | 🔴 **0.38** | 🔴 **0.44** | 0.269 |
| | 4 | 22 | 0.68 | 🔴 **0.68** | 0.68 | |
| | *W. Avg* | *63* | *0.52* | *0.48* | *0.49* | |
| | | | | | | |
| **SVM**<br>**(RBF Kernel)** | 1 | 3 | 0.11 | 0.33 | 0.17 | |
| | 2 | 14 | 🔴 **0.41** | 0.50 | 0.45 | |
| | 3 | 24 | 🔴 **0.53** | 0.33 | 0.41 | 0.302 |
| | 4 | 22 | 0.68 | 🔴 **0.68** | 0.68 | |
| | *W. Avg* | *63* | *0.54* | *0.49* | *0.50* | |
| | | | | | | |
| **Logistic**<br>**Regression** | 1 | 3 | 🔴 **0.17** | 🔴 **0.67** | 🔴 **0.27** | |
| | 2 | 14 | 0.40 | 🔴 **0.57** | 🔴 **0.47** | |
| | 3 | 24 | 0.50 | 0.21 | 0.29 | 🔴 **0.312** |
| | 4 | 22 | 🔴 **0.71** | 🔴 **0.68** | 🔴 **0.70** | |
| | *W. Avg* | *63* | *0.54* | *0.48* | *0.47* | |

> **Table 2:** Model-wise Performance Summary. The 🔴 icon indicates the best result for each specific metric across the three models.

* **Stage 1 Detection:** Logistic Regression significantly outperformed other models in detecting early-stage disease (Recall **67%** vs 33%), making it the most suitable candidate for screening purposes where minimizing False Negatives is crucial.
* **Intermediate Stages:** SVM showed the most consistent behavior for the "grey area" stages (2 and 3), managing the feature overlap slightly better than the linear model.
* **Severe Disease (Stage 4):** All models performed best on the terminal stage.

The study concludes that while the overall accuracy is constrained by the limited sample size and the significant biological overlap between intermediate stages (2 and 3), the **Logistic Regression** and **SVM** models proved to be the most capable of generalizing across the disease spectrum.
Specifically, **Logistic Regression** excelled in recognizing the **extreme stages** of the disease (1 and 4). However, its high sensitivity for Stage 1 must be interpreted with caution due to the extremely limited sample size of this subgroup ($N=16$). Conversely, the **SVM** demonstrated greater consistency in classifying **mid-stage** and Stage 4 patients, whereas it struggled to detect early-stage cases.
Notably, all models successfully minimized **severe misclassification errors** (confusing Stage 1 with Stage 4), ensuring a baseline of clinical safety, with the exception of the single atypical outlier identified in the analysis.
The general inability to reliably distinguish intermediate stages and the impact of atypical outliers indicate that classical ML approaches may have reached a **performance ceiling** on this dataset. Future iterations could benefit from **Deep Learning** techniques or the inclusion of additional histological features to capture non-linear nuances that current biomarkers fail to represent.

---

## References

- <a id="ref-dickson"></a>Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). *Prognosis in primary biliary cirrhosis: Model for decision making*. **Hepatology**, 10. DOI: 10.1002/hep.1840100102 <br>
- <a id="ref-numpy"></a>Harris, C. R., et al. (2020). *Array programming with NumPy*. **Nature**, 585(7825), 357–362. DOI: 10.1038/s41586-020-2649-2 <br>
- <a id="ref-skopt"></a>Head, T., et al. (2018). *Scikit-optimize/scikit-optimize*. **Zenodo**. DOI: 10.5281/zenodo.1157319 <br>
- <a id="ref-matplotlib"></a>Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. **Computing in Science & Engineering**, 9(3), 90–95. DOI: 10.1109/MCSE.2007.55 <br>
- <a id="ref-laschtowitz"></a>Laschtowitz, A., de Veer, R. C., der Meer, A. J. V., & Schramm, C. (2020). *Diagnosis and treatment of primary biliary cholangitis*. <br>
- <a id="ref-imblearn"></a>Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). *Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning*. **Journal of Machine Learning Research**, 18(17), 1-5. <br>
- <a id="ref-markus"></a>Markus, B. H., et al. (1989). *Efficacy of liver transplantation in patients with primary biliary cirrhosis*. **New England Journal of Medicine**, 320. DOI: 10.1056/nejm198906293202602 <br>
- <a id="ref-pandas"></a>The pandas development team. (2020). *pandas-dev/pandas: Pandas*. **Zenodo**. DOI: 10.5281/zenodo.3509134 <br>
- <a id="ref-sklearn"></a>Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. **Journal of Machine Learning Research**, 12, 2825–2830. <br>
- <a id="ref-szeghalmy"></a>Szeghalmy, S. & Fazekas, A. (2023). *A comparative study of the use of stratified cross-validation and distribution-balanced stratified cross-validation in imbalanced learning*. **Sensors**, 23. DOI: 10.3390/s23042333 <br>
- <a id="ref-trivella"></a>Trivella, J., John, B. V., & Levy, C. (2023). *Primary biliary cholangitis: Epidemiology, prognosis, and treatment*. <br>
- <a id="ref-seaborn"></a>Waskom, M. L. (2021). *seaborn: statistical data visualization*. **Journal of Open Source Software**, 6(60), 3021. DOI: 10.21105/joss.03021 <br>
- <a id="ref-you"></a>You, H., et al. (2023). *Guidelines on the diagnosis and management of primary biliary cholangitis (2021)*. **Journal of Clinical and Translational Hepatology**, 11. DOI: 10.14218/JCTH.2022.00347 <br>

---

## Contacts

For any questions, suggestions, or contributions, feel free to open an issue or contact the maintainer:<br><br>

**Marco Cuscunà** <br>
- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Seal_of_the_University_of_Bologna.svg/1200px-Seal_of_the_University_of_Bologna.svg.png" width="16"/> [marco.cuscuna@studio.unibo.it](mailto:marco.cuscuna@studio.unibo.it)
- [![GitHub](https://img.shields.io/badge/GitHub-Markus2409-181717?style=flat&logo=github)](https://github.com/Markus2409)
