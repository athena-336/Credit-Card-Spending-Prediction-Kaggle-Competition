# Credit Card Spending Prediction

Kaggle competition predicting monthly credit card spending using 
regression models with feature engineering and hyperparameter tuning.

Columbia University · APAN · Individual Project · 2026

## Problem

Predict `monthly_spend` for credit card holders based on demographic 
and behavioral features including credit score, travel frequency, 
annual income, and transaction history.

## Models Benchmarked

| Model | Test RMSE |
|---|---|
| ElasticNet (final) | **248.75** |
| Gradient Boosting | 252.90 |
| XGBoost | 253.75 |
| Linear Regression | — |
| Random Forest | 275.11 |

## Final Pipeline
IQR Outlier Removal
↓
One-Hot Encoding
↓
RobustScaler
↓
PolynomialFeatures (degree=2)
↓
ElasticNet + 5-fold GridSearchCV (18 combinations)
## Why ElasticNet Won

- Polynomial degree=2 captured mild non-linear relationships
- RobustScaler stabilized training after outlier removal
- ElasticNet regularization prevented overfitting on small dataset
- XGBoost overfit due to unscaled features and no outlier removal

## Tech Stack

Python · Scikit-learn · XGBoost · Pandas · NumPy

## Competition

Kaggle · [PAC Competition](https://www.kaggle.com/competitions/n463372)
