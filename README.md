# Classifier-Showdown-Customer-Churn-Prediction

### 📋 Overview

A comparative machine learning project that trains and evaluates five distinct classification architectures to predict customer churn, then selects and fine-tunes the best-performing model using cross-validation and hyperparameter search.

### 🎯 Objectives


Preprocess the customer_churn_dataset for classification (handle missing values, encode categorical features, scale numeric features)
Train and benchmark five classifiers using Stratified 3-Fold Cross-Validation
Apply GridSearchCV hyperparameter tuning on the top two candidate models
Select the best model based on cross-validated F1-score and evaluate it on a held-out test set
Translate model results into a business recommendation


### 🛠️ Tools & Libraries


Python, Pandas, NumPy
Matplotlib, Seaborn — visualization
Scikit-learn — Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder, GridSearchCV, cross_val_score, classification metrics


### 🧠 Models Compared

#Classifier1Logistic Regression2K-Nearest Neighbors3Decision Tree4Random Forest5Support Vector Machine (SVM)

### 🔍 Methodology


Data Exploration & Preprocessing — Loaded the dataset, checked for missing values, dropped the non-predictive CustomerID column, and built a ColumnTransformer pipeline applying StandardScaler to numeric features (for distance-sensitive models) and one-hot encoding to categorical features
Baseline Cross-Validation — Evaluated all five classifiers using Stratified 3-Fold CV to establish baseline performance before tuning
Hyperparameter Tuning — Ran GridSearchCV on Logistic Regression (tuning C, penalty) and Random Forest to extract peak performance
Model Comparison — Compiled and visualized all cross-validated F1-scores in a ranked bar chart
Final Evaluation — Tested the winning model (Tuned Random Forest) on the untouched test set, generating a full classification report and confusion matrix
Business Recommendation — Translated technical results into a stakeholder-facing recommendation


### 📊 Results

MetricScoreWinning ModelTuned Random Forest ClassifierCross-Validated F1-Score~0.999Test Set Accuracy0.9981Test Set ROC-AUC1.0000Test Set Precision / Recall / F1 (both classes)1.00 / 1.00 / 1.00

### 💼 Business Recommendation

Recommended model: Tuned Random Forest Classifier

In a churn-prediction context, missing a churning customer (a False Negative) is far more costly than mistakenly flagging a loyal customer (a False Positive) — a false negative means direct revenue loss as the customer leaves, while a false positive only costs a minor retention incentive. The Tuned Random Forest achieves near-zero false negatives while maintaining high precision, allowing a retention team to accurately target at-risk accounts without wasting budget on false alarms.

### 📁 Repository Structure

├── ClassifierShowdown_RafiaRameen.ipynb           # Main analysis & modeling notebook
├── customer_churn_dataset-testing-master.csv       # Dataset
└── README.md

### 🚀 How to Run


Clone this repository
Open ClassifierShowdown_RafiaRameen.ipynb in Jupyter Notebook or Google Colab
Ensure customer_churn_dataset-testing-master.csv is in the same directory
Run all cells sequentially to reproduce preprocessing, training, tuning, and evaluation



Part of my AI/ML internship assignments — focused on comparative model evaluation and translating ML results into business decisions.
