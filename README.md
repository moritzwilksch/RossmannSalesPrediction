# RossmannSalesPrediction
Rossmann sales prediction project from kaggle

<img width=200 src="https://user-images.githubusercontent.com/58488209/136593674-d9b51420-a110-4f30-a57e-5709818be3e3.png">



# Repo Organization
- `data`: all data sets as csv
- `helpers`: helper functions for data prep, model eval and feature engineering
- `modelling`: code for building models & tuning HP for main model
    - `baseline`: baseline models
    - `_multivariate_prediction`: trying a multivariate output model for data reshaped as single time series (failes to learn anything)
    - `univariate_prediction`: building the "real" model predicting each single sales data point separately (works great)

- `models`: model artifacts to easily load
- `notebooks`: quick visuals for presentation
    - `eda`: Data exploration
    - `ML2_RossmannCOLAB`: Large notbook to run code on google colab (messy, contains different code parts)
    - `model_eval`: model eval plots


# Model Stats
- Root Means Squared Percentage Error

|Model | Validation | Test |
| ----- | ----------| -----|
|Mean | 0.536 | 0.533 |
|Mean per store| 0.307 | 0.25425 |
|Random Forest| 0.317 | 0.20 |
|Univariate output 2-layer NN (tuned)| 0.157 | 0.138 |
