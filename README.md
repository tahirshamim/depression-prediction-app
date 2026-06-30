# Depression Prediction using Machine Learning

## Project Objective

Predict **Depression** based on demographic, academic/work, lifestyle, and family history features using machine learning models. The dataset contains **140,700 entries** and **20 features**.

## Data Preparation & Exploration (EDA)

- Mixed numerical and categorical data types were observed.
- Significant missing values in **Profession, Academic/Work Pressure, CGPA, Study/Job Satisfaction**.
- Visualized distributions using histograms, box plots, and count plots.
- Target variable **Depression** is imbalanced, with majority class = 0 (No Depression).

## Data Cleaning

- Imputed missing values based on relevance to **Students vs. Working Professionals**.
- Filled remaining nulls in **Profession** with `Not Specified`; dropped few remaining nulls.
- Removed irrelevant columns (`id`, `Name`, `City`).
- Standardized and cleaned **Sleep Duration** and **Dietary Habits**; rare categories in **Profession** and **Degree** grouped as `Other`.

## Encoding

- Binary features Label Encoded (**Gender, Student/Professional, Suicidal Thoughts, Family History**).
- Ordinal feature **Sleep Duration** mapped to numeric.
- One-Hot Encoded **Dietary Habits**, **Profession**, and **Degree**; converted resulting boolean columns to integers.

## Scaling

Numerical features (**Age, Academic/Work Pressure, CGPA, Study/Job Satisfaction, Work/Study Hours, Financial Stress**) scaled to a **0-1 range**.

## Model Training & Evaluation

- Split data: **75% train / 25% test** with stratification.
- Applied **SMOTE** to handle class imbalance.
- Trained models:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - CatBoost
  - Logistic Regression
  - Ridge Classifier
  - Gaussian Naive Bayes
- Evaluation metrics: confusion matrix and classification report.
- Best-performing models: **Random Forest, XGBoost, CatBoost** (Accuracy 93–94%, F1-score for Depression=1: 0.81–0.83).

## Feature Importance

Extracted and visualized for **Random Forest** and **XGBoost**, highlighting the most influential features for predicting depression.
