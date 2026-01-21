# Multi-Class Classification of Liver Cirrhosis Stages: An Evaluation of Random Forest, SVM, and Logistic Regression
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](codes/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Pipeline-orange?logo=jupyter)](notebooks/)
[![Dataset](https://img.shields.io/badge/Dataset-PBC_Liver_Stages-28acb3?logo=kaggle)](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)
[![Report](https://img.shields.io/badge/Read_Final_Report-PDF-b31b1b?logo=adobe-acrobat-reader&logoColor=white)](AML_BASIC__Marco_Cuscuna_Report.pdf)
[![Course](https://img.shields.io/badge/Course-AML_Basic_2026-880808?logo=unibo)](https://www.unibo.it/it/studiare/insegnamenti-competenze-trasversali-moocs/insegnamenti/insegnamento/2025/524694)

[![Model](https://img.shields.io/badge/Model-Balanced_Random_Forest-2ea44f)]()
[![Model](https://img.shields.io/badge/Model-SVM_(RBF_Kernel)-2ea44f)]()
[![Model](https://img.shields.io/badge/Model-Logistic_Regression-2ea44f)]()

<img width="517" height="352" alt="ChatGPT Image 21 gen 2026, 13_36_19" src="https://github.com/user-attachments/assets/ce5f76bb-cb74-4ac2-883c-f59a8223a8cd" />

## Introduction

This repository contains the implementation of a Machine Learning pipeline designed for the **Multi-Class Classification of Primary Biliary Cholangitis (PBC) stages**. PBC is a progressive autoimmune liver disease where accurate histological staging is critical for determining patient prognosis and defining appropriate therapeutic strategies. However, distinguishing between disease stages—particularly intermediate ones—remains a significant challenge due to the overlap of biochemical markers and clinical symptoms.

Utilizing the **Cirrhosis Prediction Dataset** (sourced from Kaggle and originating from the Mayo Clinic's longitudinal study), this project addresses real-world data complexities, including severe class imbalance and missing clinical values. To ensure reliability, a rigorous **Stability Selection** protocol was implemented to identify a robust subset of predictive biomarkers. Three distinct supervised learning architectures—**Logistic Regression**, **Support Vector Machines (SVM)**, and **Balanced Random Forest**—were then trained, optimized, and compared to evaluate their efficacy in stratifying patients across the four disease stages.

This project was developed as part of the **Applied Machine Learning** course assignment during my **MSc in Bioinformatics** (at **University of Bologna**), with the goal of integrating rigorous data preprocessing, feature selection stability, and comparative model evaluation in a clinical context.
