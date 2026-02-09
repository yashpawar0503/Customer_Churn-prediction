# Customer_Churn-prediction
Implemented and compared Logistic Regression and XGBoost models for customer churn prediction, conducted cross-validation, hyperparameter tuning, feature analysis, and threshold optimization, resulting in a robust model with ROC-AUC of 0.83.

## Project Overview

Customer churn is a critical business problem where companies aim to identify customers who are likely to discontinue their services. This project focuses on building an end-to-end **customer churn prediction system** using supervised machine learning techniques, with an emphasis on **model comparison, proper evaluation, and decision-oriented threshold optimization**.

The project compares a linear model (Logistic Regression) with a tree-based ensemble model (XGBoost) and selects the final model based on generalization performance and business-aligned metrics.

---

## Objectives

* Predict whether a customer is likely to churn (binary classification)
* Compare Logistic Regression and XGBoost models fairly
* Use appropriate evaluation metrics for imbalanced data
* Optimize classification threshold based on precision–recall trade-offs
* Select and justify the final model

---

## Dataset Description

The dataset contains customer-level information including:

* Demographic details (gender, senior citizen, dependents)
* Account information (tenure, contract type, payment method)
* Service usage (internet service, phone service, add-on services)
* Billing details (monthly charges)
* Target variable: **Churn_binary** (1 = churn, 0 = no churn)

---

## Data Preprocessing

* Removed non-informative identifiers (CustomerID)
* Handled missing values using **median imputation** (train-only)
* Encoded categorical variables using:

  * Binary encoding for Yes/No features
  * One-hot encoding for multi-class categorical features
* Eliminated redundant and collinear features
* Ensured consistent feature alignment between training and test sets

---

## Models Used

### 1. Logistic Regression

* Used as a strong baseline model
* Evaluated using ROC-AUC
* Hyperparameter tuning performed using GridSearchCV

### 2. XGBoost (Final Model)

* Tree-based ensemble model capable of capturing nonlinear relationships
* Hyperparameter tuning performed using GridSearchCV
* Selected as the final model based on superior test performance

---

## Model Evaluation Metrics

Prior to model training, an exploratory correlation analysis was performed using a Seaborn heatmap to understand linear relationships between numerical features and identify potential multicollinearity. This analysis helped guide feature selection decisions (e.g., removal of highly correlated features such as redundant charge-related variables). Given the class imbalance in churn prediction, the following metrics were used:

* **ROC-AUC** (primary metric for model selection)
* Precision, Recall, and F1-score
* Confusion Matrix

Accuracy was deliberately avoided as the primary metric due to class imbalance.

---

## Threshold Optimization

* Default threshold (0.5) resulted in low recall for churners
* Model outputs probabilities rather than fixed decisions
* Classification threshold was optimized by:

  * Evaluating F1-scores across a range of thresholds
  * Selecting the threshold that maximized F1-score

**Final chosen threshold ≈ 0.33**, resulting in:

* Significantly improved recall for churners
* Acceptable precision
* Better balance between false positives and false negatives

---

## Results Summary

| Model                          | Test ROC-AUC |
| ------------------------------ | ------------ |
| Logistic Regression (Baseline) | ~0.826       |
| Logistic Regression (Tuned)    | ~0.828       |
| XGBoost (Baseline)             | ~0.814       |
| **XGBoost (Tuned)**            | **~0.835** ✅ |

The tuned XGBoost model demonstrated the best generalization performance.

---

## Final Conclusion

After comparing Logistic Regression and XGBoost models, performing hyperparameter tuning, and evaluating multiple metrics including ROC-AUC, precision, recall, and F1-score, **XGBoost was selected as the final model**. The tuned XGBoost model achieved an ROC-AUC of approximately **0.83** and demonstrated a strong balance between precision and recall after threshold optimization, making it suitable for real-world churn prediction tasks.

---

## Skills Demonstrated

* End-to-end machine learning pipeline design
* Data preprocessing and feature engineering
* Model comparison and hyperparameter tuning
* Evaluation using ROC-AUC and precision–recall analysis
* Business-aligned decision making in ML

---

## Tools & Libraries

* Python
* pandas, numpy
* scikit-learn
* XGBoost
* matplotlib, seaborn


